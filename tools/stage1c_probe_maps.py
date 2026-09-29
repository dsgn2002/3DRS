"""Stage 1c-a: show the geometry that PCA pictures hide.

Stage 1b found LLaVA-Video's top principal components carry almost no geometry
while the full feature space still predicts it (held-out depth R^2 ~0.4). This
renders that directly: a closed-form ridge read-out from ALL standardised
dimensions, fitted on the train scenes, predicting log depth, height and
verticality on held-out scenes, next to the ground truth. No network is trained.

Each target row shares one colour scale (the ground truth's 2-98th percentile
over the scene), so predicted and true maps are directly comparable.

    python tools/stage1c_probe_maps.py --keys vlm_l8 vggt_l11
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage1b_pca_geometry as s1b  # noqa: E402


def fit_ridge(X, y, alpha=1e-2):
    n, d = X.shape
    A = torch.cat([X, torch.ones(n, 1, device=X.device, dtype=X.dtype)], 1)
    return torch.linalg.solve(A.T @ A + alpha * n * torch.eye(d + 1, device=X.device, dtype=X.dtype), A.T @ y)


def predict(X, w):
    return torch.cat([X, torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)], 1) @ w


def r2(pred, y):
    return (1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()).item()


def render(scene, frames, views, keys, preds, scene_r2, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from PIL import Image

    g = scene["grid"]
    seq = LinearSegmentedColormap.from_list("seq_blue", s1b.SEQ_BLUE)
    seq.set_bad("#dcdbd6")
    rows = [("rgb", None, None)]
    for t in s1b.TARGETS:
        rows.append((t, "gt", None))
        rows.extend((t, "pred", k) for k in keys)

    fig, axes = plt.subplots(len(rows), len(views), figsize=(1.75 * len(views) + 1.9, 1.75 * len(rows)),
                             facecolor=s1b.SURFACE)
    for c, fi in enumerate(views):
        img = Image.open(frames[fi]).convert("RGB").resize((384, 384))
        axes[0, c].imshow(np.asarray(img)[:378, :378])
        axes[0, c].set_title(f"view {c + 1} (frame {fi + 1}/32)", fontsize=8, color=s1b.INK_2)
    for r, (t, kind, k) in enumerate(rows):
        if t == "rgb":
            axes[r, 0].set_ylabel("RGB (model input)", fontsize=8, color=s1b.INK, rotation=0, ha="right",
                                  va="center", labelpad=6)
            continue
        gt = scene["targets"][t]
        finite = gt[torch.isfinite(gt)]
        vmin, vmax = float(finite.quantile(0.02)), float(finite.quantile(0.98))
        full = gt if kind == "gt" else preds[k][t]
        # Predictions exist for every token, but only tokens with ground truth are scored,
        # so mask the same tokens to keep the comparison honest.
        full = torch.where(torch.isfinite(gt), full, torch.tensor(float("nan")))
        maps = full.reshape(scene["V"], g, g)
        for c, fi in enumerate(views):
            axes[r, c].imshow(np.ma.masked_invalid(maps[fi].numpy()), cmap=seq, vmin=vmin, vmax=vmax,
                              interpolation="nearest")
        name = (f"{s1b.TARGET_LABEL[t]}\nground truth" if kind == "gt"
                else f"{s1b.label(k)} prediction\nscene R² {scene_r2[k][t]:+.2f}")
        axes[r, 0].set_ylabel(name, fontsize=8, color=s1b.INK if kind == "gt" else s1b.INK_2, rotation=0,
                              ha="right", va="center", labelpad=6)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle(f"{scene['name']} (held-out): ridge read-out from ALL feature dims vs ground truth\n"
                 "same colour scale within each target; grey = no valid depth", fontsize=9, color=s1b.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out_path, dpi=120, facecolor=s1b.SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/stage0")
    ap.add_argument("--vggt-dir", default="/diskSamsung/dinu/3DRS_data/scannet/vggt_multilayer")
    ap.add_argument("--embodiedscan-dir", default="data/embodiedscan")
    ap.add_argument("--out", default="results/stage1c")
    ap.add_argument("--keys", nargs="+", default=["vlm_l8", "vggt_l11"])
    ap.add_argument("--vis-scenes", type=int, default=2)
    ap.add_argument("--views", type=int, nargs="+", default=[0, 10, 21, 31])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    split = json.load(open(os.path.join(args.cache, "split.json")))
    train = [s1b.load_scene(args.cache, args.vggt_dir, v) for v in split["train"]]
    vis_ids = split["heldout"][: args.vis_scenes]
    vis = [s1b.load_scene(args.cache, args.vggt_dir, v) for v in vis_ids]

    preds = {i: {} for i in range(len(vis))}
    scene_r2 = {i: {} for i in range(len(vis))}
    for k in args.keys:
        Xtr = torch.cat([s["feats"][k] for s in train]).to(device).double()
        mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
        Xtr = (Xtr - mu) / sd
        for t in s1b.TARGETS:
            ytr = torch.cat([s["targets"][t] for s in train]).to(device).double()
            m = torch.isfinite(ytr)
            w = fit_ridge(Xtr[m], ytr[m])
            for i, s in enumerate(vis):
                X = (s["feats"][k].to(device).double() - mu) / sd
                p = predict(X, w).float().cpu()
                preds[i].setdefault(k, {})[t] = p
                y = s["targets"][t]
                mm = torch.isfinite(y)
                scene_r2[i].setdefault(k, {})[t] = r2(p[mm], y[mm])
        del Xtr
        torch.cuda.empty_cache()
        print(f"{s1b.label(k):>10}: " + " | ".join(
            f"{vis_ids[i].split('/')[-1]} " + " ".join(f"{t} {scene_r2[i][k][t]:+.2f}" for t in s1b.TARGETS)
            for i in range(len(vis))), flush=True)

    for i, (sid, s) in enumerate(zip(vis_ids, vis)):
        frames = s1b.frame_paths(sid, args.embodiedscan_dir, "data", s["V"])
        render(s, frames, args.views, args.keys, preds[i], scene_r2[i],
               os.path.join(args.out, f"probe_maps_{s['name']}.png"))
    json.dump({vis_ids[i]: scene_r2[i] for i in range(len(vis))},
              open(os.path.join(args.out, "probe_maps_scene_r2.json"), "w"), indent=2)
    print(f"wrote {args.out}/", flush=True)


if __name__ == "__main__":
    main()
