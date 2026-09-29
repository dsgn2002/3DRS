"""Stage 1: is metric 3D recoverable from a *frozen* LLaVA-Video's visual tokens?

Nothing downstream is trained here. We fit probes on cached features from the
training scenes and score them on held-out scenes, which answers the
precondition for the whole geometry-token idea: if geometry is not in the
frozen representation at all, no projector can surface it and the plan stops.

Two things make this an honest measurement rather than a flattering one:

**Targets are scene-independent.** This is subtler than it looks. Centring and
scaling world XYZ per scene is *not* enough: ScanNet's axis alignment leaves an
arbitrary yaw and handedness per scene, so "2m along +x" means a different
physical direction in every scene and cross-scene x/y regression is unlearnable
in principle -- an earlier version of this script scored R^2 = -0.49 on VGGT's
own features, which is how the bug announced itself. Two targets survive that
objection: log camera depth (per patch, frame-free) and height along the
gravity axis (world z is the one axis the alignment does fix), centred per
scene.

**Every number has a control**, and the controls have to actually control for
something. Three are reported:

  * *shuffled targets* -- breaks the feature/target pairing, keeps both
    marginals. Should land at 0.
  * *position only* -- predicts the target from the patch's (frame, row, column)
    alone. This is the one that matters: indoor depth is strongly predictable
    from image position (floor at the bottom of the frame), so a probe can score
    well while knowing nothing about the scene. Any claim of 3D awareness has to
    clear this bar, not the shuffled one.
  * *random square projection* -- included only as a sanity check that ridge is
    invariant to a change of basis. It is NOT an information-destroying control:
    a square Gaussian matrix is invertible, so it must reproduce the real score,
    and it does.

Effective rank is reported because a collapsed representation flatters
similarity metrics.

    python tools/stage1_probe.py --cache cache/stage0 --out stage1_report.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llava.analysis.feature_gap import (  # noqa: E402
    linear_cka,
    multiview_correspondence_score,
    mutual_knn_alignment,
    random_pair_similarity,
)


def load_split(cache_dir):
    with open(os.path.join(cache_dir, "split.json")) as f:
        return json.load(f)


def scene_file(cache_dir, video_id):
    return os.path.join(cache_dir, f"{video_id.split('/')[-1]}.npz")


def gather(cache_dir, video_ids, key, device, max_tokens_per_scene=None, seed=0):
    """Stack one feature key plus its geometry targets over a set of scenes."""
    rng = np.random.default_rng(seed)
    X, Yd, Yx, Ypos, keep_scenes = [], [], [], [], []
    for vid in video_ids:
        path = scene_file(cache_dir, vid)
        if not os.path.exists(path):
            continue
        data = np.load(path)
        if key not in data:
            continue
        feats = torch.from_numpy(data[key]).float()
        valid_2d = torch.from_numpy(data["valid"])
        valid = valid_2d.reshape(-1)
        points = torch.from_numpy(data["points"])
        depth = torch.from_numpy(data["patch_depth"]).reshape(-1)
        grid = int(data["grid"]) if "grid" in data.files else 14
        V, P = valid_2d.shape

        # Height along the gravity axis. ScanNet's axis alignment fixes z (up)
        # but leaves yaw free, so z is the only world coordinate comparable
        # across scenes. Centred on this scene's own valid median.
        height = points[..., 2].reshape(-1)
        if valid.any():
            height = height - height[valid].median()
        height = height.unsqueeze(-1)

        # Position-only features: where the patch sits in the frame. Tokens are
        # ordered frame-major then row-major, so the index gives this for free.
        t = torch.arange(V * P)
        fr, rem = t // P, t % P
        row, col = rem // grid, rem % grid
        pos = torch.stack([
            fr.float() / max(V - 1, 1), row.float() / (grid - 1), col.float() / (grid - 1)
        ], dim=-1)
        pos = torch.cat([pos] + [
            f(np.pi * k * pos) for k in (1, 2, 4) for f in (torch.sin, torch.cos)
        ], dim=-1)

        idx = torch.nonzero(valid).flatten()
        if max_tokens_per_scene and idx.numel() > max_tokens_per_scene:
            pick = rng.choice(idx.numel(), max_tokens_per_scene, replace=False)
            idx = idx[torch.from_numpy(np.sort(pick))]
        if idx.numel() == 0:
            continue
        X.append(feats[idx])
        Yd.append(torch.log(depth[idx].clamp_min(1e-3)).unsqueeze(-1))
        Yx.append(height[idx])
        Ypos.append(pos[idx])
        keep_scenes.append(vid)
    if not X:
        return None
    return (torch.cat(X).to(device), torch.cat(Yd).to(device),
            torch.cat(Yx).to(device), torch.cat(Ypos).to(device), keep_scenes)


def standardise(train, *others):
    mu, sd = train.mean(0, keepdim=True), train.std(0, keepdim=True).clamp_min(1e-6)
    return [(t - mu) / sd for t in (train, *others)]


def r2(pred, target):
    resid = ((pred - target) ** 2).sum()
    total = ((target - target.mean(0, keepdim=True)) ** 2).sum().clamp_min(1e-12)
    return (1 - resid / total).item()


def ridge_probe(Xtr, Ytr, Xte, Yte, alpha=1e-2):
    n, d = Xtr.shape
    Xtr = torch.cat([Xtr, torch.ones(n, 1, device=Xtr.device)], 1)
    Xte = torch.cat([Xte, torch.ones(Xte.shape[0], 1, device=Xte.device)], 1)
    gram = Xtr.T @ Xtr + alpha * n * torch.eye(d + 1, device=Xtr.device)
    w = torch.linalg.solve(gram.double(), (Xtr.T @ Ytr).double()).float()
    return r2(Xte @ w, Yte)


def mlp_probe(Xtr, Ytr, Xte, Yte, hidden=512, steps=1500, lr=1e-3, seed=0, device="cuda"):
    """Nonlinear counterpart: geometry may be present but not linearly decodable."""
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(Xtr.shape[1], hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, Ytr.shape[1]),
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    n = Xtr.shape[0]
    batch = min(8192, n)
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), device=device)
        loss = F.mse_loss(net(Xtr[idx]), Ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        preds = torch.cat([net(Xte[i:i + 16384]) for i in range(0, Xte.shape[0], 16384)])
    return r2(preds, Yte)


def effective_rank(X, max_tokens=8192):
    """exp(entropy of the normalised spectrum): how many directions are really used."""
    if X.shape[0] > max_tokens:
        X = X[torch.randperm(X.shape[0], device=X.device)[:max_tokens]]
    X = X - X.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(X.double())
    p = sv / sv.sum().clamp_min(1e-12)
    p = p[p > 0]
    return torch.exp(-(p * p.log()).sum()).item()


def per_scene_metrics(cache_dir, video_ids, key, device, voxel_size=0.2):
    """3DRS's correspondence score, and how far the space sits from VGGT's."""
    corr, rnd, cka, knn = [], [], [], []
    for vid in video_ids:
        path = scene_file(cache_dir, vid)
        if not os.path.exists(path):
            continue
        data = np.load(path)
        if key not in data:
            continue
        feats = torch.from_numpy(data[key]).float().to(device)
        points = torch.from_numpy(data["points"]).to(device)
        valid = torch.from_numpy(data["valid"]).to(device)
        corr.append(multiview_correspondence_score(feats, points, valid, voxel_size=voxel_size).item())
        rnd.append(random_pair_similarity(feats).item())
        if "vggt" in data and key != "vggt":
            teacher = torch.from_numpy(data["vggt"]).float().to(device)
            n = min(4096, feats.shape[0])
            sel = torch.randperm(feats.shape[0], device=device)[:n]
            cka.append(linear_cka(feats[sel], teacher[sel]).item())
            knn.append(mutual_knn_alignment(feats[sel][:1024], teacher[sel][:1024], k=10).item())
    out = {"corr_score": float(np.mean(corr)) if corr else float("nan"),
           "corr_random_pairs": float(np.mean(rnd)) if rnd else float("nan")}
    if cka:
        out["cka_vs_vggt"] = float(np.mean(cka))
        out["mutual_knn_vs_vggt"] = float(np.mean(knn))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/stage0")
    ap.add_argument("--out", default="stage1_report.json")
    ap.add_argument("--max-tokens-per-scene", type=int, default=6272)
    ap.add_argument("--mlp-steps", type=int, default=1500)
    ap.add_argument("--voxel-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = load_split(args.cache)
    train_ids, test_ids = split["train"], split["heldout"]

    probe = np.load(scene_file(args.cache, train_ids[0]))
    keys = [k for k in probe.files if k.startswith("vlm_l")]
    keys.sort(key=lambda k: int(k.split("l")[-1]))
    if "vggt" in probe.files:
        keys.append("vggt")     # reference ceiling: a real 3D model's features
    print(f"features: {keys}")
    print(f"train {len(train_ids)} scenes | held-out {len(test_ids)} scenes\n")

    report = {}
    for key in keys:
        tr = gather(args.cache, train_ids, key, device, args.max_tokens_per_scene, args.seed)
        te = gather(args.cache, test_ids, key, device, args.max_tokens_per_scene, args.seed)
        if tr is None or te is None:
            print(f"{key}: no data, skipped")
            continue
        Xtr, Ydtr, Yhtr, Ptr, _ = tr
        Xte, Ydte, Yhte, Pte, _ = te
        Xtr, Xte = standardise(Xtr, Xte)

        # Controls. Random projection keeps the token count and dimensionality
        # but destroys learned structure; shuffled targets break the pairing
        # while keeping both marginals intact.
        g = torch.Generator(device=device).manual_seed(args.seed)
        proj = torch.randn(Xtr.shape[1], Xtr.shape[1], generator=g, device=device) / Xtr.shape[1] ** 0.5
        perm = torch.randperm(Ydtr.shape[0], generator=g, device=device)
        perm_te = torch.randperm(Ydte.shape[0], generator=g, device=device)

        entry = {
            "n_train_tokens": int(Xtr.shape[0]),
            "n_test_tokens": int(Xte.shape[0]),
            "dim": int(Xtr.shape[1]),
            "effective_rank": effective_rank(Xtr),
            "depth_r2_linear": ridge_probe(Xtr, Ydtr, Xte, Ydte),
            "depth_r2_mlp": mlp_probe(Xtr, Ydtr, Xte, Ydte, steps=args.mlp_steps,
                                      seed=args.seed, device=device),
            "height_r2_linear": ridge_probe(Xtr, Yhtr, Xte, Yhte),
            "height_r2_mlp": mlp_probe(Xtr, Yhtr, Xte, Yhte, steps=args.mlp_steps,
                                       seed=args.seed, device=device),
            "ctrl_depth_r2_shuffled": ridge_probe(Xtr, Ydtr[perm], Xte, Ydte[perm_te]),
            "ctrl_depth_r2_position": ridge_probe(Ptr, Ydtr, Pte, Ydte),
            "ctrl_depth_r2_position_mlp": mlp_probe(Ptr, Ydtr, Pte, Ydte,
                                                    steps=args.mlp_steps, seed=args.seed,
                                                    device=device),
            "ctrl_height_r2_position": ridge_probe(Ptr, Yhtr, Pte, Yhte),
            "sanity_depth_r2_randproj": ridge_probe(Xtr @ proj, Ydtr, Xte @ proj, Ydte),
        }
        entry.update(per_scene_metrics(args.cache, test_ids, key, device, args.voxel_size))
        report[key] = entry

        gap = entry["corr_score"] - entry["corr_random_pairs"]
        print(f"{key:>10}  rank {entry['effective_rank']:7.1f} | "
              f"depth R2 {entry['depth_r2_linear']:+.3f}/{entry['depth_r2_mlp']:+.3f} | "
              f"height R2 {entry['height_r2_linear']:+.3f}/{entry['height_r2_mlp']:+.3f} | "
              f"corr {entry['corr_score']:+.3f} (rand {entry['corr_random_pairs']:+.3f}, "
              f"gap {gap:+.3f})"
              + (f" | CKA {entry['cka_vs_vggt']:.3f}" if "cka_vs_vggt" in entry else ""))
        print(f"{'':>10}  vs position-only: depth {entry['ctrl_depth_r2_position']:+.3f}"
              f"/{entry['ctrl_depth_r2_position_mlp']:+.3f}  "
              f"height {entry['ctrl_height_r2_position']:+.3f}   "
              f"shuffled {entry['ctrl_depth_r2_shuffled']:+.3f}")
        del Xtr, Xte
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump({"train_scenes": train_ids, "heldout_scenes": test_ids,
                   "results": report}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
