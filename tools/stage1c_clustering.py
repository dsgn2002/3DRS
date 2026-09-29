"""Stage 1c-b: do density-based clusters of each feature space follow 3D structure?

Unsupervised and training-free. Per held-out scene, the 6272 tokens (32 frames x
196 patches) of each feature space are clustered jointly across frames, and the
clusters are scored against the scene's measured 3D structure.

Clustering (settings fixed before looking at any result):
  * hdbscan_full   HDBSCAN* (Campello et al. 2013: the DBSCAN* hierarchy over all
                   density levels, so no eps to hand-tune), cosine distance on ALL
                   standardised dims. min_cluster_size=20, min_samples=10.
  * hdbscan_pca50  the same on the top-50 PCs. Stage 1b showed LLaVA's geometry
                   is NOT in its leading PCs, so this is the usual reduce-then-
                   cluster recipe, kept only to measure what it throws away.
  * dbscan_full    DBSCAN, eps at the knee of the sorted 10-NN distance curve.

Scores (valid-depth tokens; noise tokens excluded; every score is paired with
a chance level from the same labels permuted over the same tokens):
  * xview_same     P(same cluster | two different frames see the same 0.2 m voxel).
  * object_nmi     NMI with ground-truth object instances (token -> smallest GT box
                   containing its 3D point; instance identity only, no categories).
  * frame_nmi      NMI with the frame index. High means clusters are just views -- a
                   confound, not 3D structure. Lower is better.
  * compact_ratio  size-weighted mean distance of cluster members to their 3D
                   centroid, divided by the same for permuted labels. <1 = clusters
                   are tighter in the room than chance.
  * noise_frac, n_clusters.

    OMP_NUM_THREADS=2 python tools/stage1c_clustering.py --jobs 24
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage1b_pca_geometry as s1b  # noqa: E402

CONFIGS = ["hdbscan_full", "hdbscan_pca50", "dbscan_full"]
CONFIG_LABEL = {"hdbscan_full": "HDBSCAN*, all dims (cosine)",
                "hdbscan_pca50": "HDBSCAN*, top-50 PCs",
                "dbscan_full": "DBSCAN, all dims (cosine, knee eps)"}
MIN_CLUSTER, MIN_SAMPLES, VOXEL = 20, 10, 0.2
OTHER_GREY, NOISE_GREY = "#a8a7a2", "#ecebe7"
CLUSTER_COLOURS = s1b.SERIES + ["#e87ba4", "#008300", "#4a3aa7"]     # validated slots 1-7


# --------------------------------------------------------------------------- #
# Scene-level references (independent of any feature space)
# --------------------------------------------------------------------------- #
def cross_view_pairs(points, valid, P, rng, voxel=VOXEL, max_pairs=200_000):
    idx = np.flatnonzero(valid)
    frame = idx // P
    inv = np.unique(np.floor(points[idx] / voxel).astype(np.int64), axis=0, return_inverse=True)[1].reshape(-1)
    order = np.argsort(inv, kind="stable")
    inv, idx, frame = inv[order], idx[order], frame[order]
    cuts = np.flatnonzero(np.diff(inv)) + 1
    I, J = [], []
    for s, e in zip(np.r_[0, cuts], np.r_[cuts, len(inv)]):
        if e - s < 2:
            continue
        a, b = np.triu_indices(e - s, 1)
        keep = frame[s:e][a] != frame[s:e][b]
        I.append(idx[s:e][a[keep]]); J.append(idx[s:e][b[keep]])
    I, J = np.concatenate(I), np.concatenate(J)
    if len(I) > max_pairs:
        pick = rng.choice(len(I), max_pairs, replace=False)
        I, J = I[pick], J[pick]
    return I, J


def instance_ids(points, valid, boxes):
    ids = np.full(len(points), -1)
    if not boxes:
        return ids
    B = np.asarray(boxes, dtype=np.float64)
    inside = np.all(np.abs(points[:, None, :] - B[None, :, :3]) <= B[None, :, 3:6] / 2, axis=2) & valid[:, None]
    vol = np.where(inside, np.prod(B[:, 3:6], axis=1)[None], np.inf)
    has = inside.any(1)
    ids[has] = vol.argmin(1)[has]
    return ids


# --------------------------------------------------------------------------- #
# Clustering + scoring (runs in joblib workers, numpy only)
# --------------------------------------------------------------------------- #
def knee_eps(kdist):
    k = np.sort(kdist)
    y = (k - k.min()) / max(np.ptp(k), 1e-12)
    x = np.linspace(0, 1, len(k))
    return float(k[np.argmax(x - y)])


def score(labels, ref, rng, n_perm=5):
    pairs_i, pairs_j = ref["pairs"]
    valid, inst, frame, pts = ref["valid"], ref["inst"], ref["frame"], ref["points"]
    clustered = labels >= 0

    def xview(l):
        m = (l[pairs_i] >= 0) & (l[pairs_j] >= 0)
        return float((l[pairs_i][m] == l[pairs_j][m]).mean()) if m.any() else np.nan

    def obj(l):
        m = (inst >= 0) & (l >= 0)
        if m.sum() < 2:
            return np.nan, np.nan, np.nan
        return (float(normalized_mutual_info_score(inst[m], l[m])), float(adjusted_rand_score(inst[m], l[m])),
                float(adjusted_mutual_info_score(inst[m], l[m])))

    def frame_nmi(l):
        m = l >= 0
        return float(normalized_mutual_info_score(frame[m], l[m])) if m.sum() > 1 else np.nan

    def frame_ami(l):
        m = l >= 0
        return float(adjusted_mutual_info_score(frame[m], l[m])) if m.sum() > 1 else np.nan

    def dispersion(l):
        m = (l >= 0) & valid
        if not m.any():
            return np.nan
        tot, n = 0.0, 0
        for c in np.unique(l[m]):
            p = pts[m & (l == c)]
            tot += np.linalg.norm(p - p.mean(0), axis=1).sum()
            n += len(p)
        return tot / n

    def permuted():
        l = labels.copy()
        l[clustered] = rng.permutation(labels[clustered])
        return l

    ids, counts = np.unique(labels[clustered], return_counts=True)
    out = {"n_clusters": int(len(ids)), "noise_frac": float(1 - clustered.mean()),
           # A single dominant cluster makes "same cluster across views" trivially ~1.
           "largest_cluster_share": float(counts.max() / counts.sum()) if len(ids) else np.nan,
           "xview_same": xview(labels), "frame_nmi": frame_nmi(labels), "frame_ami": frame_ami(labels)}
    out["object_nmi"], out["object_ari"], out["object_ami"] = obj(labels)
    disp = dispersion(labels)
    perms = [permuted() for _ in range(n_perm)]
    out["chance_xview_same"] = float(np.nanmean([xview(p) for p in perms]))
    out["chance_object_nmi"] = float(np.nanmean([obj(p)[0] for p in perms]))
    out["chance_frame_nmi"] = float(np.nanmean([frame_nmi(p) for p in perms]))
    # Chance-corrected agreement (Cohen-kappa form): 0 = what these cluster sizes
    # give by luck, 1 = perfect. Comparable across configs with very different
    # cluster-size distributions, which the raw rate is not.
    p, pc = out["xview_same"], out["chance_xview_same"]
    out["xview_kappa"] = float((p - pc) / (1 - pc)) if np.isfinite(p) and pc < 1 - 1e-9 else np.nan
    out["compact_ratio"] = float(disp / np.nanmean([dispersion(p) for p in perms])) if np.isfinite(disp) else np.nan
    return out


def cluster_job(scene_name, key, Z, P50, ref, seed):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    D = np.clip(1.0 - Zn @ Zn.T, 0.0, 2.0).astype(np.float64)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2

    labels = {
        "hdbscan_full": HDBSCAN(min_cluster_size=MIN_CLUSTER, min_samples=MIN_SAMPLES,
                                metric="precomputed").fit_predict(D),
        "hdbscan_pca50": HDBSCAN(min_cluster_size=MIN_CLUSTER, min_samples=MIN_SAMPLES).fit_predict(P50),
    }
    eps = knee_eps(np.partition(D, MIN_SAMPLES, axis=1)[:, MIN_SAMPLES])
    labels["dbscan_full"] = DBSCAN(eps=max(eps, 1e-6), min_samples=MIN_SAMPLES,
                                   metric="precomputed").fit_predict(D)

    result = {cfg: score(l, ref, rng) for cfg, l in labels.items()}
    result["dbscan_full"]["eps"] = eps
    return {"scene": scene_name, "key": key, "metrics": result,
            "labels": {k: v.astype(np.int32) for k, v in labels.items()}, "seconds": time.time() - t0}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def colourise(labels):
    """Largest 7 clusters get the validated categorical slots; the rest share one
    grey; noise is near-surface. Colours are consistent across views of a scene
    (one joint clustering), not across rows (different clusterings)."""
    rgb = np.zeros((len(labels), 3))
    from matplotlib.colors import to_rgb
    rgb[:] = to_rgb(OTHER_GREY)
    rgb[labels < 0] = to_rgb(NOISE_GREY)
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    for colour, c in zip(CLUSTER_COLOURS, ids[np.argsort(-counts)]):
        rgb[labels == c] = to_rgb(colour)
    return rgb


def render_maps(scene, frames, views, ref, job_by_key, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    g, V = scene["grid"], scene["V"]
    keys = s1b.feature_keys()
    rows = ["RGB (model input)", "GT objects (boxes)"] + keys
    fig, axes = plt.subplots(len(rows), len(views), figsize=(1.75 * len(views) + 1.9, 1.75 * len(rows)),
                             facecolor=s1b.SURFACE)
    inst_rgb = colourise(np.where(ref["inst"] >= 0, ref["inst"], -1)).reshape(V, g, g, 3)
    maps = {k: colourise(job_by_key[k]["labels"]["hdbscan_full"]).reshape(V, g, g, 3) for k in keys}
    for c, fi in enumerate(views):
        axes[0, c].imshow(np.asarray(Image.open(frames[fi]).convert("RGB").resize((384, 384)))[:378, :378])
        axes[0, c].set_title(f"view {c + 1} (frame {fi + 1}/32)", fontsize=8, color=s1b.INK_2)
        axes[1, c].imshow(inst_rgb[fi], interpolation="nearest")
        for r, k in enumerate(keys, start=2):
            axes[r, c].imshow(maps[k][fi], interpolation="nearest")
    axes[0, 0].set_ylabel(rows[0], fontsize=8, color=s1b.INK, rotation=0, ha="right", va="center", labelpad=6)
    axes[1, 0].set_ylabel("GT objects\n(largest 7 coloured)", fontsize=8, color=s1b.INK, rotation=0, ha="right",
                          va="center", labelpad=6)
    for r, k in enumerate(keys, start=2):
        m = job_by_key[k]["metrics"]["hdbscan_full"]
        axes[r, 0].set_ylabel(f"{s1b.label(k)}\n{m['n_clusters']} clusters, {m['noise_frac']:.0%} noise",
                              fontsize=8, color=s1b.INK, rotation=0, ha="right", va="center", labelpad=6)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle(f"{scene['name']} (held-out): HDBSCAN* clusters over all 32 frames, all dims, cosine\n"
                 "same colour across views = same cluster; largest 7 coloured, other clusters grey, noise pale",
                 fontsize=9, color=s1b.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out_path, dpi=120, facecolor=s1b.SURFACE)
    plt.close(fig)


def render_summary(rows, n_scenes, out_path):
    """Metrics for the configuration that actually clusters (HDBSCAN*, all dims),
    plus a separate count of collapsed solutions for all three. Chance-corrected
    scores are meaningless when one cluster holds (nearly) every token -- plotting
    them next to real clusterings is how the first version of this chart misled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = s1b.feature_keys()
    xs = np.array(list(range(len(s1b.VLM_LAYERS))) + [len(s1b.VLM_LAYERS) + 1 + i for i in range(len(s1b.VGGT_LAYERS))],
                  dtype=float)
    primary = "hdbscan_full"
    panels = [("xview_kappa", "cross-view agreement (chance-corrected κ) ↑", 0.0),
              ("object_ami", "AMI with GT objects ↑", 0.0),
              ("frame_ami", "AMI with frame index (view grouping)", 0.0),
              ("compact_ratio", "3D spread of clusters vs chance ↓", 1.0),
              ("noise_frac", "tokens left unclustered (noise)", None)]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.4), facecolor=s1b.SURFACE)
    axes = axes.ravel()

    def style(ax, title):
        ax.set_title(title, fontsize=9, color=s1b.INK)
        ax.set_xticks(xs, [s1b.label(k).replace("LLaVA ", "").replace("VGGT ", "") for k in keys],
                      fontsize=7, color=s1b.INK_2)
        ax.text(2, -0.13, "LLaVA-Video", ha="center", fontsize=8, color=s1b.INK_2, transform=ax.get_xaxis_transform())
        ax.text(len(s1b.VLM_LAYERS) + 2.5, -0.13, "VGGT", ha="center", fontsize=8, color=s1b.INK_2,
                transform=ax.get_xaxis_transform())
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c3")
        ax.tick_params(length=0)

    for ax, (metric, title, ref) in zip(axes, panels):
        mean = np.array([rows[(k, primary)][metric][0] for k in keys])
        std = np.array([rows[(k, primary)][metric][1] for k in keys])
        if ref is not None:
            ax.axhline(ref, color="#c9c8c3", linewidth=1)
        ax.errorbar(xs, mean, yerr=std, fmt="o", color=s1b.SERIES[0], markersize=6, markeredgecolor=s1b.SURFACE,
                    markeredgewidth=1.2, elinewidth=1.2, capsize=0)
        if metric != "noise_frac":                      # label only points scored on fewer scenes
            for x, k, m_, sd_ in zip(xs, keys, mean, std):
                n_ok = int(rows[(k, primary)]["scored_scenes"][0])
                if n_ok < n_scenes and np.isfinite(m_):
                    ax.annotate(f"n={n_ok}", (x, m_ + sd_), textcoords="offset points", xytext=(0, 4),
                                ha="center", fontsize=7, color=s1b.INK_2)
        style(ax, title)

    ax = axes[-1]
    offsets = {"hdbscan_full": -0.25, "hdbscan_pca50": 0.0, "dbscan_full": 0.25}
    for cfg, colour, marker in zip(CONFIGS, s1b.SERIES, s1b.MARKERS):
        ax.plot(xs + offsets[cfg], [rows[(k, cfg)]["degenerate_scenes"][0] for k in keys], marker, color=colour,
                markersize=6, markeredgecolor=s1b.SURFACE, markeredgewidth=1.2, linestyle="none",
                label=CONFIG_LABEL[cfg])
    ax.set_ylim(-0.5, n_scenes + 0.5)
    style(ax, f"scenes collapsed to one cluster (of {n_scenes}) ↓")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)  # clear of the data
    fig.suptitle(f"HDBSCAN*, all dims (cosine), mean ± sd over held-out scenes (n={n_scenes} unless marked: "
                 "scenes that collapsed to one cluster are excluded from scores); grey line = chance", fontsize=9,
                 color=s1b.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130, facecolor=s1b.SURFACE, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/stage0")
    ap.add_argument("--vggt-dir", default="/diskSamsung/dinu/3DRS_data/scannet/vggt_multilayer")
    ap.add_argument("--embodiedscan-dir", default="data/embodiedscan")
    ap.add_argument("--boxes", nargs="+", default=["data/metadata/scannet_train_gt_box.json",
                                                   "data/metadata/scannet_val_gt_box.json"])
    ap.add_argument("--out", default="results/stage1c")
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--vis-scenes", type=int, default=2)
    ap.add_argument("--views", type=int, nargs="+", default=[0, 10, 21, 31])
    ap.add_argument("--limit-scenes", type=int, default=None, help="smoke test")
    ap.add_argument("--limit-keys", nargs="+", default=None, help="smoke test")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    keys = args.limit_keys or s1b.feature_keys()
    split = json.load(open(os.path.join(args.cache, "split.json")))
    heldout = split["heldout"][: args.limit_scenes] if args.limit_scenes else split["heldout"]
    boxes = {}
    for p in args.boxes:
        boxes.update(json.load(open(p)))

    # Train-split standardisation and PCA-50 per feature space.
    train = [s1b.load_scene(args.cache, args.vggt_dir, v) for v in split["train"]]
    stats = {}
    for k in keys:
        X = torch.cat([s["feats"][k] for s in train]).cuda().double()
        mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp_min(1e-6)
        Z = (X - mu) / sd
        _, evecs = s1b.pca(Z)
        stats[k] = (mu.float().cpu().numpy(), sd.float().cpu().numpy(),
                    Z.mean(0, keepdim=True).float().cpu().numpy(), evecs[:, :50].float().cpu().numpy())
        del X, Z
        torch.cuda.empty_cache()
    del train
    print(f"standardisation + PCA-50 fitted on {len(split['train'])} train scenes for {len(keys)} feature spaces",
          flush=True)

    def jobs_for(scene_ids, seed_base):
        scenes, specs = [], []
        for si, sid in enumerate(scene_ids):
            s = s1b.load_scene(args.cache, args.vggt_dir, sid)
            pts, valid = s["points"].numpy().astype(np.float64), s["valid"].numpy()
            rng = np.random.default_rng(seed_base + si)
            ref = {"points": pts, "valid": valid, "frame": np.arange(s["V"] * s["P"]) // s["P"],
                   "inst": instance_ids(pts, valid, boxes.get(sid, [])),
                   "pairs": cross_view_pairs(pts, valid, s["P"], rng)}
            scenes.append((sid, s, ref))
            for ki, k in enumerate(keys):
                mu, sd, zmean, comps = stats[k]
                Z = ((s["feats"][k].float().numpy() - mu) / sd).astype(np.float32)
                specs.append((s["name"], k, Z, ((Z - zmean) @ comps).astype(np.float32), ref,
                              seed_base * 100 + si * 10 + ki))
        return scenes, specs

    t0 = time.time()
    all_results = []
    # Phase 1: the pre-registered visualisation scenes, rendered as soon as they finish.
    vis_ids = heldout[: args.vis_scenes]
    scenes, specs = jobs_for(vis_ids, 1)
    results = Parallel(n_jobs=args.jobs, backend="loky")(delayed(cluster_job)(*sp) for sp in specs)
    all_results += results
    for sid, s, ref in scenes:
        by_key = {r["key"]: r for r in results if r["scene"] == s["name"]}
        if len(by_key) == len(s1b.feature_keys()):
            frames = s1b.frame_paths(sid, args.embodiedscan_dir, "data", s["V"])
            render_maps(s, frames, args.views, ref, by_key, os.path.join(args.out, f"clusters_{s['name']}.png"))
            print(f"rendered clusters_{s['name']}.png ({time.time() - t0:.0f}s)", flush=True)

    # Phase 2: the remaining held-out scenes, for the summary statistics.
    rest = heldout[args.vis_scenes:]
    if rest:
        _, specs = jobs_for(rest, 2)
        all_results += Parallel(n_jobs=args.jobs, backend="loky")(delayed(cluster_job)(*sp) for sp in specs)

    for r in all_results:
        m = r["metrics"]["hdbscan_full"]
        print(f"{r['scene']:>13} {s1b.label(r['key']):>10} k={m['n_clusters']:3d} noise {m['noise_frac']:4.0%} "
              f"xview kappa {m['xview_kappa']:.2f} objAMI {m['object_ami']:.2f} frameAMI {m['frame_ami']:.2f} "
              f"compact {m['compact_ratio']:.2f} largest {m['largest_cluster_share']:.2f}  [{r['seconds']:.0f}s]",
              flush=True)

    metrics = ["xview_kappa", "xview_same", "chance_xview_same", "object_ami", "object_nmi", "chance_object_nmi",
               "object_ari", "frame_ami", "frame_nmi", "chance_frame_nmi", "compact_ratio",
               "largest_cluster_share", "noise_frac", "n_clusters"]
    rows = {}
    for k in keys:
        for cfg in CONFIGS:
            vals = [r["metrics"][cfg] for r in all_results if r["key"] == k]
            # Collapsed = one cluster holds >= 90% of clustered tokens (or nothing clustered).
            # Agreement scores are undefined or unstable there, so they are averaged over
            # the remaining scenes only; noise and collapse counts use every scene.
            def is_collapsed(v):
                return not np.isfinite(v["largest_cluster_share"]) or v["largest_cluster_share"] >= 0.9
            ok = [v for v in vals if not is_collapsed(v)]
            rows[(k, cfg)] = {}
            for m in metrics:
                src = vals if m in ("noise_frac", "n_clusters", "largest_cluster_share") else ok
                xs_ = [v[m] for v in src]
                rows[(k, cfg)][m] = ((float(np.nanmean(xs_)), float(np.nanstd(xs_))) if xs_
                                     else (float("nan"), float("nan")))
            rows[(k, cfg)]["degenerate_scenes"] = (float(len(vals) - len(ok)), 0.0)
            rows[(k, cfg)]["scored_scenes"] = (float(len(ok)), 0.0)
    with open(os.path.join(args.out, "cluster_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "config", "n_scenes", "collapsed_scenes"]
                   + [f"{m}_{s}" for m in metrics for s in ("mean", "std")])
        for (k, cfg), v in rows.items():
            w.writerow([s1b.label(k), cfg, len(heldout), int(v["degenerate_scenes"][0])]
                       + [f"{x:.4f}" for m in metrics for x in v[m]])
    json.dump([{kk: vv for kk, vv in r.items() if kk != "labels"} for r in all_results],
              open(os.path.join(args.out, "cluster_metrics_per_scene.json"), "w"), indent=2)
    if len(keys) == len(s1b.feature_keys()):
        render_summary(rows, len(heldout), os.path.join(args.out, "cluster_summary.png"))
    print(f"\nwrote {args.out}/ in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
