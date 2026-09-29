"""Stage 1b: what do the principal components of each feature space encode?

Pure analysis on frozen, cached features -- nothing is trained except closed-form
linear read-outs. Fitted on the train split, scored on held-out scenes.

For every feature space (LLaVA-Video hidden layers 0/8/16/22/28, VGGT aggregator
layers 4/11/17/23 -- the four VGGT's own DPT depth/point heads read):

  * PCA on per-dimension standardised features. Qwen2 hidden states carry a few
    enormous outlier channels; unstandardised, they own PC1 and the spectrum
    says nothing about content. The raw PC1 variance share is reported so the
    size of that effect is visible rather than assumed.
  * Pearson correlation of each of the top-10 PCs with three geometry targets
    that are comparable across scenes: log camera depth, height along gravity
    (centred per scene), and verticality |n_z| of the surface normal (1 = floor
    / table top, 0 = wall). World x/y are excluded: ScanNet's axis alignment
    leaves an arbitrary yaw per scene.
  * Held-out R^2 of a linear read-out from the top-3 / 10 / 50 PCs versus from
    all dimensions. This is the check that stops a wrong conclusion: geometry
    that lives in low-variance directions is invisible in a PCA picture but
    still recoverable, so "not in the top PCs" must never be read as "absent".
  * PCA->RGB grids for pre-registered held-out scenes and views, with the basis
    fitted jointly over all 32 frames of the scene so colours are comparable
    across views (per-frame PCA would flip component order and sign between
    frames and fake view-inconsistency). Colours are NOT comparable across rows:
    PC sign and order are arbitrary per feature space.

    python tools/stage1b_pca_geometry.py --cache cache/stage0 \
        --vggt-dir /diskSamsung/dinu/3DRS_data/scannet/vggt_multilayer --out results/stage1b
"""

import argparse
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("tds", ROOT / "llava/model/three_d_supervision.py")
tds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tds)

VLM_LAYERS = [0, 8, 16, 22, 28]
VGGT_LAYERS = [4, 11, 17, 23]
TARGETS = ["log_depth", "height", "verticality"]
TARGET_LABEL = {"log_depth": "log depth", "height": "height (gravity axis)",
                "verticality": "verticality |n_z|"}
TOPK = [3, 10, 50]

# Palette (dataviz reference instance): diverging blue<->red with a neutral
# gray midpoint; sequential single-hue blue; categorical slots 1-4.
DIV_NEG, DIV_MID, DIV_POS = "#1c5cab", "#f0efec", "#e34948"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]            # secondary encoding: identity is never colour alone
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def feature_keys():
    return [f"vlm_l{l}" for l in VLM_LAYERS] + [f"vggt_l{l}" for l in VGGT_LAYERS]


def label(key):
    model, layer = key.split("_l")
    return f"{'LLaVA' if model == 'vlm' else 'VGGT'} L{layer}"


def load_scene(cache, vggt_dir, video_id, grid=14):
    name = video_id.split("/")[-1]
    d = np.load(os.path.join(cache, name + ".npz"))
    v = np.load(os.path.join(vggt_dir, name, "vggt_layers.npz"))
    valid = torch.from_numpy(d["valid"])
    points = torch.from_numpy(d["points"]).float()
    V, P = valid.shape
    z = points[..., 2]
    height = (z - z[valid].median()).reshape(-1)

    normals, nmask = tds.grid_normals(points, valid, grid)          # (V, g-2, g-2, 3)
    vert = torch.full((V, grid, grid), float("nan"))
    vert[:, 1:-1, 1:-1] = torch.where(nmask, normals[..., 2].abs(), torch.tensor(float("nan")))

    feats = {f"vlm_l{l}": torch.from_numpy(d[f"vlm_l{l}"]) for l in VLM_LAYERS}
    feats.update({f"vggt_l{l}": torch.from_numpy(v[f"feature_l{l}"]).reshape(V * P, -1)
                  for l in VGGT_LAYERS})
    targets = {
        "log_depth": torch.where(valid.reshape(-1),
                                 torch.log(torch.from_numpy(d["patch_depth"]).reshape(-1).clamp_min(1e-3)),
                                 torch.tensor(float("nan"))),
        "height": torch.where(valid.reshape(-1), height, torch.tensor(float("nan"))),
        "verticality": vert.reshape(-1),
    }
    return {"name": name, "V": V, "P": P, "grid": grid, "feats": feats, "targets": targets,
            "points": points.reshape(-1, 3), "valid": valid.reshape(-1)}


def pca(X):
    """Top eigenvectors of the covariance of X (rows = tokens), float64."""
    Xc = X.double() - X.double().mean(0, keepdim=True)
    cov = Xc.T @ Xc / (X.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    return evals[order].clamp_min(0), evecs[:, order]


def ridge_r2(Xtr, ytr, Xte, yte, alpha=1e-2):
    n, d = Xtr.shape
    A = torch.cat([Xtr, torch.ones(n, 1, device=Xtr.device, dtype=Xtr.dtype)], 1)
    B = torch.cat([Xte, torch.ones(Xte.shape[0], 1, device=Xte.device, dtype=Xte.dtype)], 1)
    w = torch.linalg.solve(A.T @ A + alpha * n * torch.eye(d + 1, device=A.device, dtype=A.dtype), A.T @ ytr)
    pred = B @ w
    return (1 - ((pred - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()).item()


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum().item() / max((a.norm() * b.norm()).item(), 1e-12)


def analyse(train, test, device):
    report = {}
    for key in feature_keys():
        Xtr = torch.cat([s["feats"][key] for s in train]).to(device).float()
        Xte = torch.cat([s["feats"][key] for s in test]).to(device).float()

        raw_evals, _ = pca(Xtr)
        mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
        Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
        evals, evecs = pca(Ztr)
        comps = evecs[:, :max(TOPK)].float()
        Str = (Ztr - Ztr.mean(0, keepdim=True)) @ comps
        Ste = (Zte - Ztr.mean(0, keepdim=True)) @ comps

        entry = {
            "dim": int(Xtr.shape[1]),
            "raw_pc1_variance_share": (raw_evals[0] / raw_evals.sum()).item(),
            "std_explained_variance_top10": (evals[:10] / evals.sum()).tolist(),
            "std_cumulative_variance": {k: (evals[:k].sum() / evals.sum()).item() for k in TOPK},
            "corr_top10": {}, "r2": {},
        }
        for t in TARGETS:
            ytr = torch.cat([s["targets"][t] for s in train]).to(device)
            yte = torch.cat([s["targets"][t] for s in test]).to(device)
            mtr, mte = torch.isfinite(ytr), torch.isfinite(yte)
            entry["corr_top10"][t] = [pearson(Ste[mte, i].double(), yte[mte].double()) for i in range(10)]
            entry["r2"][t] = {f"top{k}": ridge_r2(Str[mtr, :k].double(), ytr[mtr].double(),
                                                  Ste[mte, :k].double(), yte[mte].double(), alpha=1e-6)
                              for k in TOPK}
            entry["r2"][t]["all"] = ridge_r2(Ztr[mtr].double(), ytr[mtr].double(),
                                             Zte[mte].double(), yte[mte].double())
        report[key] = entry
        best = {t: int(np.argmax(np.abs(entry["corr_top10"][t]))) for t in TARGETS}
        print(f"{label(key):>10}  raw PC1 {entry['raw_pc1_variance_share']:5.1%} | "
              + " | ".join(f"{t}: best PC{best[t]+1} r={entry['corr_top10'][t][best[t]]:+.2f} "
                           f"R2 top3 {entry['r2'][t]['top3']:+.2f} top10 {entry['r2'][t]['top10']:+.2f} "
                           f"all {entry['r2'][t]['all']:+.2f}" for t in TARGETS), flush=True)
        del Xtr, Xte, Ztr, Zte, Str, Ste
        torch.cuda.empty_cache()
    return report


def frame_paths(video_id, embodiedscan_dir, video_folder, n_frames):
    """The exact frames the VLM saw: VideoProcessor.sample_frame_files, force_sample."""
    for split in ("train", "val", "test"):
        path = os.path.join(embodiedscan_dir, f"embodiedscan_infos_{split}.pkl")
        if not os.path.exists(path):
            continue
        for item in pickle.load(open(path, "rb"))["data_list"]:
            if item["sample_idx"] == video_id:
                files = [os.path.join(video_folder, img["img_path"]) for img in item["images"]]
                idx = np.linspace(0, len(files) - 1, n_frames, dtype=int)
                return [files[i] for i in idx]
    raise KeyError(video_id)


def scene_pca_rgb(feats, stats):
    """Joint PCA over one scene's frames -> per-token RGB in [0, 1]."""
    mu, sd = stats
    Z = (feats.float() - mu) / sd
    _, evecs = pca(Z)
    S = (Z - Z.mean(0, keepdim=True)) @ evecs[:, :3].float()
    lo = torch.quantile(S, 0.02, dim=0)
    hi = torch.quantile(S, 0.98, dim=0)
    return ((S - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)


def render_grid(scene, frames, views, stats, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from PIL import Image

    g, P = scene["grid"], scene["P"]
    seq = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    seq.set_bad("#dcdbd6")
    rows = ["RGB (model input)"] + [TARGET_LABEL[t] for t in TARGETS] + [label(k) for k in feature_keys()]
    rgb = {k: scene_pca_rgb(scene["feats"][k], stats[k]).reshape(scene["V"], g, g, 3).numpy()
           for k in feature_keys()}

    fig, axes = plt.subplots(len(rows), len(views), figsize=(1.75 * len(views) + 1.4, 1.75 * len(rows)),
                             facecolor=SURFACE)
    for c, fi in enumerate(views):
        img = Image.open(frames[fi]).convert("RGB").resize((384, 384))
        axes[0, c].imshow(np.asarray(img)[:378, :378])
        axes[0, c].set_title(f"view {c + 1} (frame {fi + 1}/32)", fontsize=8, color=INK_2)
        for r, t in enumerate(TARGETS, start=1):
            vals = scene["targets"][t].reshape(scene["V"], g, g)[fi].numpy()
            finite = scene["targets"][t][torch.isfinite(scene["targets"][t])]
            axes[r, c].imshow(np.ma.masked_invalid(vals), cmap=seq, interpolation="nearest",
                              vmin=float(finite.quantile(0.02)), vmax=float(finite.quantile(0.98)))
        for r, k in enumerate(feature_keys(), start=1 + len(TARGETS)):
            axes[r, c].imshow(rgb[k][fi], interpolation="nearest")
    for r, name in enumerate(rows):
        axes[r, 0].set_ylabel(name, fontsize=8, color=INK, rotation=0, ha="right", va="center", labelpad=6)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle(f"{scene['name']}: joint PCA over 32 frames, top-3 PCs as RGB\n"
                 "rows are separate feature spaces - colours are comparable across views, not across rows",
                 fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def render_correlations(report, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    div = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
    keys = feature_keys()
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(4.0 * len(TARGETS) + 1.2, 4.2), facecolor=SURFACE,
                             gridspec_kw={"wspace": 0.06})
    for i, (ax, t) in enumerate(zip(axes, TARGETS)):
        M = np.array([report[k]["corr_top10"][t] for k in keys])
        im = ax.imshow(M, cmap=div, vmin=-1, vmax=1, aspect="auto")
        for r in range(M.shape[0]):                     # label only the strongest cell per row
            c = int(np.argmax(np.abs(M[r])))
            ax.text(c, r, f"{M[r, c]:+.2f}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if abs(M[r, c]) > 0.55 else INK)
        ax.axhline(len(VLM_LAYERS) - 0.5, color=SURFACE, linewidth=3)
        ax.set_xticks(range(10), [f"PC{i+1}" for i in range(10)], fontsize=7, color=INK_2)
        ax.set_yticks(range(len(keys)), [label(k) for k in keys], fontsize=8, color=INK)
        if i > 0:                                        # one set of row labels, not three colliding ones
            ax.tick_params(labelleft=False)
        ax.set_title(f"held-out correlation with {TARGET_LABEL[t]}", fontsize=9, color=INK)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("Pearson r", fontsize=8, color=INK_2)
    cb.outline.set_visible(False)
    fig.savefig(out_path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def render_r2(report, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = feature_keys()
    xs = list(range(len(VLM_LAYERS))) + [len(VLM_LAYERS) + 1 + i for i in range(len(VGGT_LAYERS))]
    series = [("top3", "top-3 PCs"), ("top10", "top-10 PCs"), ("top50", "top-50 PCs"), ("all", "all dims")]
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(4.4 * len(TARGETS), 3.6), sharey=True, facecolor=SURFACE)
    for ax, t in zip(axes, TARGETS):
        for (sk, sl), col, mk in zip(series, SERIES, MARKERS):
            ys = [report[k]["r2"][t][sk] for k in keys]
            for seg in (slice(0, len(VLM_LAYERS)), slice(len(VLM_LAYERS), None)):
                ax.plot(xs[seg], ys[seg], color=col, linewidth=2, marker=mk, markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, label=sl if seg.start == 0 else None)
        ax.axhline(0, color="#c9c8c3", linewidth=1)
        ax.set_xticks(xs, [label(k).replace("LLaVA ", "").replace("VGGT ", "") for k in keys], fontsize=7, color=INK_2)
        ax.text(2, -0.16, "LLaVA-Video layers", ha="center", fontsize=8, color=INK_2, transform=ax.get_xaxis_transform())
        ax.text(len(VLM_LAYERS) + 2.5, -0.16, "VGGT layers", ha="center", fontsize=8, color=INK_2,
                transform=ax.get_xaxis_transform())
        ax.set_title(f"held-out R², {TARGET_LABEL[t]}", fontsize=9, color=INK)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c3")
        ax.tick_params(length=0)
    axes[0].set_ylabel("R² (held-out scenes)", fontsize=8, color=INK_2)
    lowest = min(report[k]["r2"][t][sk] for k in keys for t in TARGETS for sk, _ in series)
    axes[0].set_ylim(min(-0.05, lowest - 0.05), 1.0)        # never clip a data point off the bottom
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, fontsize=8, loc="upper center", ncol=len(series),
               bbox_to_anchor=(0.5, 1.06))                    # above the plots, not over the data
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/stage0")
    ap.add_argument("--vggt-dir", default="/diskSamsung/dinu/3DRS_data/scannet/vggt_multilayer")
    ap.add_argument("--embodiedscan-dir", default="data/embodiedscan")
    ap.add_argument("--out", default="results/stage1b")
    ap.add_argument("--vis-scenes", type=int, default=2, help="first N held-out scenes (pre-registered)")
    ap.add_argument("--replot", action="store_true",
                    help="re-render the correlation and R^2 charts from the saved JSON only")
    ap.add_argument("--views", type=int, nargs="+", default=[0, 10, 21, 31],
                    help="frame indices to show (pre-registered, evenly spaced)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.replot:
        report = json.load(open(os.path.join(args.out, "pca_geometry.json")))["results"]
        render_correlations(report, os.path.join(args.out, "pc_geometry_correlation.png"))
        render_r2(report, os.path.join(args.out, "pc_geometry_r2.png"))
        print("re-rendered charts")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = json.load(open(os.path.join(args.cache, "split.json")))
    train = [load_scene(args.cache, args.vggt_dir, v) for v in split["train"]]
    test = [load_scene(args.cache, args.vggt_dir, v) for v in split["heldout"]]
    print(f"train {len(train)} / held-out {len(test)} scenes, "
          f"{sum(s['V'] * s['P'] for s in train)} / {sum(s['V'] * s['P'] for s in test)} tokens\n", flush=True)

    report = analyse(train, test, device)
    json.dump({"train_scenes": split["train"], "heldout_scenes": split["heldout"], "results": report},
              open(os.path.join(args.out, "pca_geometry.json"), "w"), indent=2)

    stats = {}
    for k in feature_keys():
        X = torch.cat([s["feats"][k] for s in train]).float()
        stats[k] = (X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp_min(1e-6))
    render_correlations(report, os.path.join(args.out, "pc_geometry_correlation.png"))
    render_r2(report, os.path.join(args.out, "pc_geometry_r2.png"))
    for scene_id, scene in zip(split["heldout"][: args.vis_scenes], test[: args.vis_scenes]):
        frames = frame_paths(scene_id, args.embodiedscan_dir, "data", scene["V"])
        render_grid(scene, frames, args.views, stats, os.path.join(args.out, f"pca_rgb_{scene['name']}.png"))
    print(f"\nwrote {args.out}/", flush=True)


if __name__ == "__main__":
    main()
