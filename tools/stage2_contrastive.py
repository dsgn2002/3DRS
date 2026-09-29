"""Stage 2: contrastive geometric alignment between a frozen VLM and a frozen 3D model.

Both encoders stay frozen; only small projectors train, on the Stage 0 cache.
At inference only the VLM branch g(z) is used -- VGGT is a training-time target,
never an input -- so every metric below is computed on g(z) alone.

The ladder (each rung adds one idea, everything else held fixed):

  regress          3DRS-style: g(z) regresses the VGGT token by cosine.
  global           frame-pooled symmetric InfoNCE between g(z) and h(t).
  token            token-level symmetric InfoNCE; positive = same patch.
  token_xview      + positives = every token in the same voxel, in any frame,
                   cross-modal and VLM<->VLM (fixes token's false negatives).
  token_xview_hard + an instance term whose denominator is restricted to hard
                   negatives: tokens the *frozen* VLM finds most similar but that
                   sit >= hard_margin metres away in the same scene. Label-free
                   stand-in for "same category, different instance": GT category
                   labels only joined to 39% of boxes on this slice.
  *_depth          any contrastive rung above plus lam_depth * a depth term:
                   a linear head on the UNNORMALISED projection g(z) regresses
                   standardised log sensor depth (smooth-L1). The contrastive
                   losses only see g(z)/|g(z)|, so cross-view invariance and
                   per-view depth need not compete. Near and far tokens can be
                   re-weighted with --depth-weight inverse_freq.
  depth_only       ceiling control: g trained on the depth term alone.

Controls, because a trained projector can raise any metric it is trained on:

  raw              no projector; the frozen features themselves.
  random           untrained projector (same architecture).
  token_shuffled   token InfoNCE with VGGT tokens permuted within each frame, so
                   training is identical but patch correspondence is destroyed.

Metrics on held-out scenes (anisotropy-corrected where it matters):
  corr_gap         cross-view same-voxel cosine minus random-pair cosine.
  depth_r2/height  held-out ridge probes fitted on train-scene projections.
  hard_gap         cosine of hard pairs minus random pairs; LOWER = the space
                   separates look-alike objects at different locations (H3).
  retention        mutual-kNN overlap with the frozen features (H2 proxy: how
                   much of the VLM's own neighbourhood structure survives).
  cka_vggt         linear CKA with the teacher.
  retrieval@1      within-frame patch retrieval g(z_i) -> teacher token i
                   (chance ~0.5%).
  eff_rank         effective rank; collapse shows up here first.

Overfitting checks (all optional):
  --eval-train     also score the same metrics on the TRAIN scenes; compare the
                   train-minus-held-out gap against raw/random, which have no
                   trained projector and so show the gap the probes alone cause.
  --eval-at N ...  evaluate at these steps too (learning curve on both splits).
  --train-scenes K train on the first K scenes of a fixed permutation of the
                   training split, so smaller sets are nested in larger ones.
  --fold k         rotate which third of the 24 scenes is held out; fold 0 is
                   the original split.

    python tools/stage2_contrastive.py --cache cache/stage0 --layer 8 --seeds 0 1 2
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from llava.analysis.feature_gap import (  # noqa: E402
    linear_cka,
    multiview_correspondence_score,
    mutual_knn_alignment,
    random_pair_similarity,
)
from stage1_probe import effective_rank, ridge_probe, standardise  # noqa: E402
from llava.model.three_d_supervision import (  # noqa: E402
    hard_negative_mask,
    supcon,
    sym_infonce,
)

TRAINED = ["regress", "global", "token", "token_xview", "token_xview_hard", "token_shuffled",
           "token_xview_depth", "token_xview_hard_depth", "depth_only"]
DEPTH_BINS = 16


def uses_depth(method):
    return method == "depth_only" or method.endswith("_depth")


def base_method(method):
    return method[: -len("_depth")] if method.endswith("_depth") else method


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_scenes(cache, ids, layer, device):
    scenes = []
    for vid in ids:
        d = np.load(os.path.join(cache, vid.split("/")[-1] + ".npz"))
        valid = torch.from_numpy(d["valid"]).to(device)
        points = torch.from_numpy(d["points"]).to(device)
        V, P = valid.shape
        z = points[..., 2]
        scenes.append(dict(
            name=vid, V=V, P=P,
            vlm=torch.from_numpy(d[f"vlm_l{layer}"]).to(device),
            vggt=torch.from_numpy(d["vggt"]).to(device),
            points=points, valid=valid,
            logdepth=torch.log(torch.from_numpy(d["patch_depth"]).to(device).clamp_min(1e-3)),
            height=z - z[valid].median(),
        ))
    return scenes


def feature_stats(scenes, key):
    """Per-dimension mean/std over valid train tokens. Qwen2 hidden states carry
    massive outlier channels, so unnormalised inputs would let a few dims
    dominate every cosine."""
    n, s, s2 = 0, None, None
    for sc in scenes:
        x = sc[key].float()[sc["valid"].reshape(-1)]
        s = x.sum(0) if s is None else s + x.sum(0)
        s2 = (x * x).sum(0) if s2 is None else s2 + (x * x).sum(0)
        n += x.shape[0]
    mu = s / n
    return mu, (s2 / n - mu * mu).clamp_min(1e-6).sqrt()


def sample_batch(scenes, n_scenes, n_frames, gen, device):
    Z, T, PTS, SID, FID, D = [], [], [], [], [], []
    for k in torch.randperm(len(scenes), generator=gen)[:n_scenes].tolist():
        sc = scenes[k]
        frames = torch.randperm(sc["V"], generator=gen)[:n_frames]
        tok = (frames[:, None] * sc["P"] + torch.arange(sc["P"])[None]).reshape(-1).to(device)
        tok = tok[sc["valid"].reshape(-1)[tok]]
        Z.append(sc["vlm"][tok])
        T.append(sc["vggt"][tok])
        PTS.append(sc["points"].reshape(-1, 3)[tok])
        SID.append(torch.full((tok.numel(),), k, device=device))
        FID.append(tok // sc["P"])
        D.append(sc["logdepth"].reshape(-1)[tok])
    return (torch.cat(Z).float(), torch.cat(T).float(), torch.cat(PTS),
            torch.cat(SID), torch.cat(FID), torch.cat(D).float())


def depth_stats(scenes, max_weight=10.0):
    """Train-set log-depth mean/std, plus inverse-frequency weights over
    DEPTH_BINS equal-width bins of standardised log depth. Weights are capped
    and rescaled so the average token weight is 1, which keeps lam_depth
    comparable between weighting modes."""
    d = torch.cat([sc["logdepth"].reshape(-1)[sc["valid"].reshape(-1)] for sc in scenes]).float()
    mu, sd = d.mean(), d.std().clamp_min(1e-6)
    x = (d - mu) / sd
    edges = torch.linspace(x.min().item(), x.max().item(), DEPTH_BINS + 1, device=x.device)[1:-1]
    count = torch.bincount(torch.bucketize(x, edges), minlength=DEPTH_BINS).float()
    w = 1.0 / count.clamp_min(1.0)
    w = w / (w * count).sum() * count.sum()           # token-average weight = 1
    w = w.clamp(max=max_weight)                        # cap near-empty bins
    w = w / (w * count).sum() * count.sum()
    return mu, sd, edges, w


def depth_loss(pred, target_std, dstats, weighting):
    per = F.smooth_l1_loss(pred.squeeze(-1), target_std, reduction="none")
    if weighting == "inverse_freq":
        _, _, edges, w = dstats
        per = per * w[torch.bucketize(target_std, edges)]
    return per.mean()


def voxel_groups(points, sid, voxel):
    key = torch.cat([sid[:, None], torch.floor(points / voxel).long()], dim=1)
    return torch.unique(key, dim=0, return_inverse=True)[1]


# --------------------------------------------------------------------------- #
# Model and losses
# --------------------------------------------------------------------------- #
class Projector(nn.Module):
    def __init__(self, din, dout, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Linear(hidden, dout))

    def forward(self, x):
        return self.net(x)


def method_loss(method, g, h, z, t, pts, sid, fid, args, gen_device):
    method = base_method(method)
    if method == "regress":
        return (1 - F.cosine_similarity(g(z), t, dim=-1)).mean(), {}

    if method == "global":
        key = sid * 1000 + fid
        uniq, inv = torch.unique(key, return_inverse=True)
        count = torch.bincount(inv, minlength=len(uniq)).float()[:, None]
        gz, ht = g(z), h(t)
        zf = torch.zeros(len(uniq), gz.shape[1], device=z.device).index_add_(0, inv, gz) / count
        tf = torch.zeros(len(uniq), ht.shape[1], device=z.device).index_add_(0, inv, ht) / count
        return sym_infonce(zf, tf, args.tau), {}

    if method in ("token", "token_shuffled"):
        if method == "token_shuffled":
            key = (sid * 1000 + fid).double()
            base = torch.argsort(key)
            shuf = torch.argsort(key + torch.rand(key.shape, device=z.device, dtype=torch.double) * 0.5)
            perm = torch.empty_like(base)
            perm[base] = shuf
            t = t[perm]
        return sym_infonce(g(z), h(t), args.tau), {}

    gz, ht = g(z), h(t)
    grp = voxel_groups(pts, sid, args.voxel)
    same = grp[:, None] == grp[None, :]
    eye = torch.eye(len(grp), dtype=torch.bool, device=z.device)
    xview = same & (fid[:, None] != fid[None, :])

    l_cm = 0.5 * (supcon(gz, ht, same, args.tau) + supcon(ht, gz, same, args.tau))
    l_xv = supcon(gz, gz, xview, args.tau, allowed=~eye)
    parts = {"cm": l_cm.item(), "xv": l_xv.item()}
    loss = l_cm + args.lam_xview * l_xv

    if method == "token_xview_hard":
        hard = hard_negative_mask(z, pts, sid, args.hard_margin, args.hard_k) & ~eye
        l_inst = supcon(gz, gz, xview, args.tau, allowed=hard)
        parts["inst"] = l_inst.item()
        loss = loss + args.lam_inst * l_inst
    return loss, parts


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def embed(x_std, g, chunk=16384):
    if g is None:
        return x_std
    return torch.cat([g(x_std[i:i + chunk]) for i in range(0, x_std.shape[0], chunk)])


@torch.no_grad()
def evaluate(method, g, h, train, test, stats, args, seed):
    (zmu, zsd), (tmu, tsd) = stats
    rng = torch.Generator(device="cpu").manual_seed(1000 + seed)
    out = {}

    # Probes: fit on train-scene projections, score on held-out scenes.
    Xtr, Ydtr, Yhtr = [], [], []
    for sc in train:
        v = sc["valid"].reshape(-1)
        idx = torch.nonzero(v).flatten()
        idx = idx[torch.randperm(idx.numel(), generator=rng)[: args.probe_tokens].to(idx.device)]
        f = embed(((sc["vlm"].float() - zmu) / zsd)[idx], g)
        Xtr.append(f.float()); Ydtr.append(sc["logdepth"].reshape(-1)[idx, None])
        Yhtr.append(sc["height"].reshape(-1)[idx, None])
    Xtr, Ydtr, Yhtr = torch.cat(Xtr), torch.cat(Ydtr), torch.cat(Yhtr)

    Xte, Ydte, Yhte = [], [], []
    corr, cka, ret, rank_feats, hard, retr = [], [], [], [], [], []
    for sc in test:
        v = sc["valid"].reshape(-1)
        z_std = (sc["vlm"].float() - zmu) / zsd
        feat = embed(z_std, g).float()
        Xte.append(feat[v]); Ydte.append(sc["logdepth"].reshape(-1)[v, None])
        Yhte.append(sc["height"].reshape(-1)[v, None])
        rank_feats.append(feat[v])

        corr.append((multiview_correspondence_score(feat, sc["points"], sc["valid"], voxel_size=args.voxel)
                     - random_pair_similarity(feat)).item())

        idx = torch.nonzero(v).flatten()
        sub = idx[torch.randperm(idx.numel(), generator=rng)[:4096].to(idx.device)]
        t_std = (sc["vggt"].float() - tmu) / tsd
        cka.append(linear_cka(feat[sub], t_std[sub]).item())
        ret.append(mutual_knn_alignment(feat[sub[:2048]], z_std[sub[:2048]], k=10).item())

        # Hard-pair separation: pairs the frozen VLM finds alike but >= margin apart.
        anc = sub[:512]
        zn = F.normalize(z_std[idx], dim=-1)
        sim = F.normalize(z_std[anc], dim=-1) @ zn.t()
        pts = sc["points"].reshape(-1, 3)
        far = torch.cdist(pts[anc], pts[idx]) >= args.hard_margin
        sim = sim.masked_fill(~far, float("-inf"))
        top = sim.topk(8, dim=1)
        fn = F.normalize(feat, dim=-1)
        pair_cos = (fn[anc][:, None, :] * fn[idx[top.indices]]).sum(-1)
        pair_cos = pair_cos[torch.isfinite(top.values)]
        hard.append((pair_cos.mean() - random_pair_similarity(feat)).item())

        # Within-frame patch retrieval into the teacher (branches that have one).
        if h is not None or method == "regress":
            keys = embed(t_std, h) if h is not None else t_std
            q, kk = F.normalize(feat, dim=-1), F.normalize(keys.float(), dim=-1)
            hits, total = 0, 0
            for fr in range(sc["V"]):
                s0 = fr * sc["P"]
                m = sc["valid"][fr]
                if m.sum() < 2:
                    continue
                qi, ki = q[s0:s0 + sc["P"]][m], kk[s0:s0 + sc["P"]][m]
                hits += ((qi @ ki.t()).argmax(1) == torch.arange(qi.shape[0], device=qi.device)).sum().item()
                total += qi.shape[0]
            retr.append(hits / max(total, 1))

    Xte, Ydte, Yhte = torch.cat(Xte), torch.cat(Ydte), torch.cat(Yhte)
    a, b = standardise(Xtr, Xte)
    out["depth_r2"] = ridge_probe(a, Ydtr, b, Ydte)
    out["height_r2"] = ridge_probe(a, Yhtr, b, Yhte)
    out["corr_gap"] = float(np.mean(corr))
    out["hard_gap"] = float(np.mean(hard))
    out["retention"] = float(np.mean(ret))
    out["cka_vggt"] = float(np.mean(cka))
    out["retrieval@1"] = float(np.mean(retr)) if retr else float("nan")
    out["eff_rank"] = effective_rank(torch.cat(rank_feats))
    return out


# --------------------------------------------------------------------------- #
def train_method(method, train, stats, args, seed, device, eval_cb=None):
    torch.manual_seed(seed)
    (zmu, zsd), (tmu, tsd) = stats
    dz, dt = train[0]["vlm"].shape[1], train[0]["vggt"].shape[1]
    if method == "regress":
        g, h = Projector(dz, dt).to(device), None
    elif method == "depth_only":
        g, h = Projector(dz, args.dim).to(device), None
    else:
        g, h = Projector(dz, args.dim).to(device), Projector(dt, args.dim).to(device)
    params = list(g.parameters()) + (list(h.parameters()) if h is not None else [])
    n_params = sum(p.numel() for p in params)
    # The depth head is a training-time auxiliary; evaluation still uses g(z)
    # alone, and the held-out ridge probe is refitted from scratch.
    dhead = nn.Linear(args.dim, 1).to(device) if uses_depth(method) else None
    if dhead is not None:
        params += list(dhead.parameters())
        dstats = depth_stats(train)
    if method == "random":
        return g, None, n_params, {}

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    warm = max(1, args.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n_scenes, n_frames = (8, 4) if method == "global" else (args.batch_scenes, args.batch_frames)

    log = {}
    for step in range(args.steps):
        z, t, pts, sid, fid, logd = sample_batch(train, n_scenes, n_frames, gen, device)
        z, t = (z - zmu) / zsd, (t - tmu) / tsd
        if method == "depth_only":
            loss, parts = torch.zeros((), device=device), {}
        else:
            loss, parts = method_loss(method, g, h, z, t, pts, sid, fid, args, device)
        if dhead is not None:
            l_d = depth_loss(dhead(g(z)), (logd - dstats[0]) / dstats[1], dstats, args.depth_weight)
            parts["depth"] = l_d.item()
            loss = loss + (1.0 if method == "depth_only" else args.lam_depth) * l_d
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{method} seed {seed}: non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step in (0, args.steps // 2, args.steps - 1):
            log[step] = {"loss": loss.item(), **parts}
        if eval_cb is not None and (step + 1) in args.eval_at and (step + 1) != args.steps:
            g.eval()
            if h is not None:
                h.eval()
            eval_cb(step + 1, g, h)
            g.train()
            if h is not None:
                h.train()
    g.eval()
    if h is not None:
        h.eval()
    return g, h, n_params, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/stage0")
    ap.add_argument("--out", default="stage2_report.json")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--methods", nargs="+", default=["raw", "random"] + TRAINED)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--voxel", type=float, default=0.2)
    ap.add_argument("--lam-xview", dest="lam_xview", type=float, default=1.0)
    ap.add_argument("--lam-inst", dest="lam_inst", type=float, default=0.5)
    ap.add_argument("--lam-depth", dest="lam_depth", type=float, default=1.0)
    ap.add_argument("--depth-weight", dest="depth_weight", default="none",
                    choices=["none", "inverse_freq"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-train", dest="eval_train", action="store_true")
    ap.add_argument("--eval-at", dest="eval_at", type=int, nargs="*", default=[])
    ap.add_argument("--train-scenes", dest="train_scenes", type=int, default=0,
                    help="0 = all training scenes")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--hard-margin", dest="hard_margin", type=float, default=1.0)
    ap.add_argument("--hard-k", dest="hard_k", type=int, default=16)
    ap.add_argument("--batch-scenes", dest="batch_scenes", type=int, default=2)
    ap.add_argument("--batch-frames", dest="batch_frames", type=int, default=8)
    ap.add_argument("--probe-tokens", dest="probe_tokens", type=int, default=4000)
    args = ap.parse_args()

    device = args.device
    split = json.load(open(os.path.join(args.cache, "split.json")))
    if args.fold:
        pool = split["heldout"] + split["train"]
        n = len(pool) // args.folds
        held = pool[args.fold * n:(args.fold + 1) * n]
        split = {"heldout": held, "train": [v for v in pool if v not in held]}
    if args.train_scenes:
        order = np.random.default_rng(12345).permutation(len(split["train"]))
        split = {**split, "train": [split["train"][i] for i in order[: args.train_scenes]]}
    train = load_scenes(args.cache, split["train"], args.layer, device)
    test = load_scenes(args.cache, split["heldout"], args.layer, device)
    stats = (feature_stats(train, "vlm"), feature_stats(train, "vggt"))
    print(f"layer {args.layer} | train {len(train)} / held-out {len(test)} scenes | "
          f"{args.steps} steps | seeds {args.seeds}\n", flush=True)

    results = {}
    for method in args.methods:
        runs = []
        seeds = [0] if method == "raw" else args.seeds
        for seed in seeds:
            t0 = time.time()
            if method == "raw":
                g, h, n_params, log = None, None, 0, {}
            else:
                curve = {}

                def cb(step, g_, h_):
                    curve[step] = {"heldout": evaluate(method, g_, h_, train, test, stats, args, seed)}
                    if args.eval_train:
                        curve[step]["train"] = evaluate(method, g_, h_, train, train, stats, args, seed)
                    print(f"{method:>17} s{seed} step {step}: held-out depth "
                          f"{curve[step]['heldout']['depth_r2']:+.3f} corr {curve[step]['heldout']['corr_gap']:+.3f}"
                          + (f" | train depth {curve[step]['train']['depth_r2']:+.3f} "
                             f"corr {curve[step]['train']['corr_gap']:+.3f}" if args.eval_train else ""),
                          flush=True)

                g, h, n_params, log = train_method(method, train, stats, args, seed, device,
                                                   eval_cb=cb if args.eval_at else None)
                log = {"loss": log, "curve": curve}
            metrics = evaluate(method, g, h, train, test, stats, args, seed)
            if args.eval_train:
                metrics["train"] = evaluate(method, g, h, train, train, stats, args, seed)
            metrics.update(seed=seed, params=n_params, seconds=round(time.time() - t0, 1),
                           train_log=log)
            runs.append(metrics)
            print(f"{method:>17} s{seed}  corr_gap {metrics['corr_gap']:+.3f}  "
                  f"depth {metrics['depth_r2']:+.3f}  height {metrics['height_r2']:+.3f}  "
                  f"hard_gap {metrics['hard_gap']:+.3f}  ret {metrics['retention']:.3f}  "
                  f"cka {metrics['cka_vggt']:.3f}  r@1 {metrics['retrieval@1']:.3f}  "
                  f"rank {metrics['eff_rank']:.0f}  ({metrics['seconds']}s)", flush=True)
            del g, h
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        keys = ["corr_gap", "depth_r2", "height_r2", "hard_gap", "retention",
                "cka_vggt", "retrieval@1", "eff_rank"]
        results[method] = {
            "runs": runs,
            "mean": {k: float(np.nanmean([r[k] for r in runs])) for k in keys},
            "std": {k: float(np.nanstd([r[k] for r in runs])) for k in keys},
            "params": runs[0]["params"],
        }
        if args.eval_train:
            results[method]["train_mean"] = {k: float(np.nanmean([r["train"][k] for r in runs])) for k in keys}
            results[method]["train_std"] = {k: float(np.nanstd([r["train"][k] for r in runs])) for k in keys}

    print("\n=== summary (mean ± std over seeds; held-out scenes) ===")
    print(f"{'method':>17} {'corr_gap':>13} {'depth_r2':>13} {'height_r2':>13} "
          f"{'hard_gap↓':>13} {'retention':>11} {'r@1':>7} {'rank':>6}")
    for m, r in results.items():
        mu, sd = r["mean"], r["std"]
        print(f"{m:>17} {mu['corr_gap']:+.3f}±{sd['corr_gap']:.3f} "
              f"{mu['depth_r2']:+.3f}±{sd['depth_r2']:.3f} {mu['height_r2']:+.3f}±{sd['height_r2']:.3f} "
              f"{mu['hard_gap']:+.3f}±{sd['hard_gap']:.3f} {mu['retention']:.3f}±{sd['retention']:.2f} "
              f"{mu['retrieval@1']:.3f} {mu['eff_rank']:6.0f}")
        if "train_mean" in r:
            tm = r["train_mean"]
            print(f"{'  (train)':>17} {tm['corr_gap']:+.3f}       {tm['depth_r2']:+.3f}       "
                  f"{tm['height_r2']:+.3f}")

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "split": split, "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
