"""VGGT geometry for ScanQA re-answering.

recon : run VGGT on 32 uniformly sampled frames (same selection as the
        Fable zero-shot run), save depth-unprojected point maps + cameras.
post  : estimate floor plane -> gravity-aligned frame, render top-down map
        with camera trajectory, write grid-overlaid frames for pixel picking.
query : lift (frame, u, v) clicks in ORIGINAL pixels to aligned 3D, render
        a labelled top-down map, print pairwise relations.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCANNET = Path("/diskSamsung/dinu/3DRS_data/scannet/posed_images")
VGGT_SRC = Path("/home/guest/dinu/3DRS_stage1/3DRS/vggt")
CKPT = Path("/home/guest/dinu/3DRS_stage1/3DRS/checkpoints/model.pt")
N_FRAMES = 32
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def select_frames(scene: str) -> list[Path]:
    files = sorted((SCANNET / scene).glob("*.jpg"))
    idx = np.linspace(0, len(files) - 1, N_FRAMES, dtype=int)
    return [files[i] for i in idx]


# ---------------------------------------------------------------- recon
def recon(scene: str, out: Path) -> None:
    import torch

    sys.path.insert(0, str(VGGT_SRC))
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    frames = select_frames(scene)
    model = VGGT()
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    model = model.cuda().eval()
    imgs = load_and_preprocess_images([str(p) for p in frames]).cuda()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = model(imgs)
        extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], imgs.shape[-2:])
    torch.cuda.synchronize()
    depth = pred["depth"].float().cpu().numpy().squeeze(0)          # S,H,W,1
    dconf = pred["depth_conf"].float().cpu().numpy().squeeze(0)     # S,H,W
    extr = extr.float().cpu().numpy().squeeze(0)                    # S,3,4
    intr = intr.float().cpu().numpy().squeeze(0)                    # S,3,3
    pts = unproject_depth_map_to_point_map(depth, extr, intr)       # S,H,W,3
    rgb = pred["images"].float().cpu().numpy().squeeze(0).transpose(0, 2, 3, 1)
    orig_w, orig_h = Image.open(frames[0]).size
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "vggt.npz",
        points=pts.astype(np.float32), conf=dconf.astype(np.float32),
        depth=depth[..., 0].astype(np.float32),
        extrinsic=extr, intrinsic=intr,
        rgb=(np.clip(rgb, 0, 1) * 255).astype(np.uint8),
        frame_names=np.array([p.name for p in frames]),
        orig_size=np.array([orig_w, orig_h]),
    )
    print(json.dumps({"scene": scene, "frames": len(frames),
                      "proc_hw": list(pts.shape[1:3]), "orig_wh": [orig_w, orig_h],
                      "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}))


# ---------------------------------------------------------------- post
def fit_floor(P: np.ndarray, cams: np.ndarray, iters: int = 400, seed: int = 0):
    """RANSAC plane on points below the camera centroid; returns (n, d) with n·x+d=0,
    n oriented so cameras are on the positive side (up)."""
    rng = np.random.default_rng(seed)
    best, best_in = None, 0
    tol = 0.02 * np.linalg.norm(P.std(axis=0))
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        d = -n @ s[0]
        dist = np.abs(P @ n + d)
        k = int((dist < tol).sum())
        if k > best_in:
            best_in, best = k, (n, d)
    n, d = best
    inl = P[np.abs(P @ n + d) < tol]
    c = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - c)
    n = vt[2] / np.linalg.norm(vt[2])
    d = -n @ c
    if (cams @ n + d).mean() < 0:      # cameras must be above the floor
        n, d = -n, -d
    return n, d, best_in / len(P)


def rot_to_z(n: np.ndarray) -> np.ndarray:
    z = np.array([0, 0, 1.0])
    v = np.cross(n, z); s = np.linalg.norm(v); c = n @ z
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


def load(out: Path):
    z = np.load(out / "vggt.npz")
    return {k: z[k] for k in z.files}


def cam_centers(extr: np.ndarray) -> np.ndarray:
    R, t = extr[:, :, :3], extr[:, :, 3]
    return -np.einsum("sji,sj->si", R, t)   # -R^T t


def aligned_points(d, A):
    R, t, s = A["R"], A["t"], A["scale"]
    P = d["points"].reshape(-1, 3)
    return (s * (P @ R.T + t)).reshape(d["points"].shape)


def post(out: Path) -> None:
    d = load(out)
    P = d["points"].reshape(-1, 3)
    C = d["conf"].reshape(-1)
    fin = np.isfinite(P).all(1)
    thr = np.quantile(C[fin], 0.3)
    keep = fin & (C >= thr)
    Pk = P[keep]
    cams = cam_centers(d["extrinsic"])
    # floor: candidate points in the lowest band along the axis of largest camera-relative spread
    # use camera principal "down" (mean of camera y axes in world) to pick the band
    down = d["extrinsic"][:, 1, :3].mean(0); down /= np.linalg.norm(down)
    h = Pk @ down
    band = Pk[h > np.quantile(h, 0.75)]           # farthest along "down" = floor candidates
    sub = band[np.random.default_rng(0).choice(len(band), min(60000, len(band)), replace=False)]
    n, dd, inl = fit_floor(sub, cams)
    R = rot_to_z(n)
    t = np.array([0, 0, dd])                        # after rotation floor is z=0: n·x+d=0 -> z' = n·x + d
    Pa = Pk @ R.T + t
    cams_a = cams @ R.T + t
    # scale: VGGT is scale-free. Use mean camera height above floor as 1 unit reference; report raw.
    cam_h = float(np.median(cams_a[:, 2]))
    # heuristic metric guess: handheld ScanNet camera ~1.4 m above floor
    scale = 1.4 / cam_h if cam_h > 0 else 1.0
    A = {"R": R.tolist(), "t": t.tolist(), "scale": scale, "floor_inlier_frac": inl,
         "median_cam_height_raw": cam_h,
         "note": "scale is a HEURISTIC (median camera height := 1.4 m); relations, not metres, are the output"}
    (out / "align.json").write_text(json.dumps(A, indent=1))
    render_topdown(out, Pa * scale, cams_a * scale, d["rgb"].reshape(-1, 3)[keep], labels=None)
    write_grid_frames(out, d)
    print(json.dumps({"scene": out.name, "floor_inliers": round(inl, 3), "cam_h_raw": round(cam_h, 3),
                      "cams_above_floor": int((cams_a[:, 2] > 0).sum())}))


def render_topdown(out: Path, Pa, cams_a, rgb, labels, fname="topdown.png", res=900):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # drop floor and ceiling bands so furniture shows; keep 0.1..2.0 m
    m = (Pa[:, 2] > 0.10) & (Pa[:, 2] < 2.0)
    Q = Pa[m]; col = rgb[m] / 255.0
    if len(Q) > 250000:
        i = np.random.default_rng(0).choice(len(Q), 250000, replace=False); Q, col = Q[i], col[i]
    fig, ax = plt.subplots(figsize=(9, 9), dpi=res // 9)
    ax.scatter(Q[:, 0], Q[:, 1], s=0.3, c=col, linewidths=0)
    ax.plot(cams_a[:, 0], cams_a[:, 1], "-", color="red", lw=0.8, alpha=0.6)
    for i, c in enumerate(cams_a):
        ax.text(c[0], c[1], str(i), color="red", fontsize=6, ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.1", fc="white", ec="red", lw=0.4, alpha=0.8))
    if labels:
        for lab in labels:
            x, y = lab["xyz"][:2]
            ax.plot(x, y, "o", color="blue", ms=5)
            ax.text(x, y, lab["name"], color="blue", fontsize=8, fontweight="bold",
                    ha="left", va="bottom", bbox=dict(fc="white", ec="blue", lw=0.5, alpha=0.85))
    ax.set_aspect("equal"); ax.grid(True, lw=0.3, alpha=0.5)
    ax.set_xlabel("x (heuristic m)"); ax.set_ylabel("y (heuristic m)")
    ax.set_title(f"{out.name} top-down (gravity-aligned, VGGT, heuristic scale); red = camera index")
    fig.tight_layout(); fig.savefig(out / fname); plt.close(fig)


def write_grid_frames(out: Path, d) -> None:
    gdir = out / "grid"; gdir.mkdir(exist_ok=True)
    font = ImageFont.truetype(FONT, 22)
    for i, name in enumerate(d["frame_names"]):
        im = Image.open(SCANNET / out.name / str(name)).convert("RGB")
        W, H = im.size
        dr = ImageDraw.Draw(im)
        for x in range(0, W, 100):
            dr.line([(x, 0), (x, H)], fill=(255, 255, 0), width=1)
            dr.text((x + 2, 2), str(x), fill=(255, 255, 0), font=font)
        for y in range(0, H, 100):
            dr.line([(0, y), (W, y)], fill=(0, 255, 255), width=1)
            dr.text((2, y + 2), str(y), fill=(0, 255, 255), font=font)
        dr.text((W - 120, H - 30), f"f{i:02d}", fill=(255, 0, 0), font=font)
        im.resize((W // 2, H // 2)).save(gdir / f"f{i:02d}.jpg", quality=80)


# ---------------------------------------------------------------- query
def query(out: Path, clicks_path: Path) -> None:
    d = load(out)
    A = json.loads((out / "align.json").read_text())
    R, t, s = np.array(A["R"]), np.array(A["t"]), A["scale"]
    S, H, W = d["conf"].shape
    ow, oh = d["orig_size"]
    Pa = (d["points"].reshape(-1, 3) @ R.T + t) * s
    cams = (cam_centers(d["extrinsic"]) @ R.T + t) * s
    # camera forward (+z in cam) in aligned frame, projected to ground
    fwd = (d["extrinsic"][:, 2, :3] @ R.T)          # row 2 of R_cw = z axis of cam in world
    clicks = json.loads(clicks_path.read_text())
    labels = []
    for c in clicks:
        f, u, v = c["frame"], c["u"], c["v"]
        x = int(round(u * W / ow)); y = int(round(v * H / oh))
        x0, x1 = max(0, x - 2), min(W, x + 3); y0, y1 = max(0, y - 2), min(H, y + 3)
        idx = (f * H + np.arange(y0, y1)[:, None]) * W + np.arange(x0, x1)[None, :]
        pts = Pa[idx.reshape(-1)]; cf = d["conf"][f, y0:y1, x0:x1].reshape(-1)
        ok = np.isfinite(pts).all(1)
        p = np.median(pts[ok], axis=0) if ok.any() else np.full(3, np.nan)
        labels.append({"name": c["name"], "frame": f, "uv": [u, v], "xyz": p.round(3).tolist(),
                       "conf": float(np.median(cf[ok])) if ok.any() else 0.0,
                       "dist_from_cam": float(np.linalg.norm(p - cams[f]))})
    render_topdown(out, Pa[np.isfinite(Pa).all(1) & (d["conf"].reshape(-1) >= np.quantile(d["conf"], 0.3))],
                   cams, d["rgb"].reshape(-1, 3)[np.isfinite(Pa).all(1) & (d["conf"].reshape(-1) >= np.quantile(d["conf"], 0.3))],
                   labels, fname=clicks_path.stem + "_map.png")
    # room extent from floor-band points
    fl = Pa[np.isfinite(Pa).all(1) & (Pa[:, 2] > 0.05) & (Pa[:, 2] < 1.8)]
    lo, hi = np.percentile(fl[:, :2], 2, axis=0), np.percentile(fl[:, :2], 98, axis=0)
    ctr = (lo + hi) / 2
    print("room_xy_extent", lo.round(2).tolist(), hi.round(2).tolist(), "centre", ctr.round(2).tolist())
    for L in labels:
        p = np.array(L["xyz"])
        rel = (p[:2] - ctr) / ((hi - lo) / 2)
        print(f"{L['name']:<22} f{L['frame']:02d} xyz={L['xyz']} conf={L['conf']:.2f} "
              f"height={p[2]:.2f} room_rel=({rel[0]:+.2f},{rel[1]:+.2f}) "
              f"wall_dist={np.min(np.r_[p[:2]-lo, hi-p[:2]]):.2f}")
    names = [L["name"] for L in labels]; X = np.array([L["xyz"] for L in labels])
    print("pairwise ground distances:")
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            print(f"  {names[i]} - {names[j]}: {np.linalg.norm(X[i,:2]-X[j,:2]):.2f}")
    print("camera positions/forward (ground):")
    for i in range(S):
        fw = fwd[i, :2] / (np.linalg.norm(fwd[i, :2]) + 1e-9)
        print(f"  f{i:02d} at {cams[i,:2].round(2).tolist()} fwd {fw.round(2).tolist()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["recon", "post", "query"])
    ap.add_argument("scene")
    ap.add_argument("--root", type=Path, default=Path("/home/guest/dinu/scanqa_vggt"))
    ap.add_argument("--clicks", type=Path)
    a = ap.parse_args()
    out = a.root / a.scene
    {"recon": lambda: recon(a.scene, out), "post": lambda: post(out),
     "query": lambda: query(out, a.clicks)}[a.cmd]()
