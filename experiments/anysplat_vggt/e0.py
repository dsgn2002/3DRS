"""E0: does anchor-aligned Gaussian placement (AnchorSplat, arXiv 2604.07053) help
without the learned decoder?

Per-scene optimisation proxy. One geometric prior per scene gives a dense,
pixel-aligned point pool from the 32 training views. Four arms share that pool,
the same GT cameras, the same loss and the same iteration budget:

  pixel        one Gaussian per pool point (pixel-aligned, like feed-forward
               pixel-aligned 3DGS), free positions, no densification
  anchor       FPS anchors (count = occupied voxels at --voxel), 4 Gaussians per
               anchor, centre = anchor + r * tanh(offset), no densification
  anchor_free  same init as anchor, unconstrained centres
  3dgs         one Gaussian per anchor, standard gsplat densification

Priors:
  gt    ScanNet sensor depth + GT poses (upper bound on prior quality)
  vggt  VGGT depth on the 32 training frames only, Sim(3)-aligned to the GT
        cameras (rotation from camera rotations, scale/shift from centres) (no test frames are seen by VGGT)

Evaluation on 8 held-out frames: PSNR / SSIM / LPIPS at 648x484, rendered
expected depth vs sensor depth (AbsRel, delta1, metric, no scale fitting) on
pixels with rendered alpha > 0.5, plus that coverage fraction.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SCANNET = Path("/diskSamsung/dinu/3DRS_data/scannet/posed_images")
VGGT_SRC = Path("/home/guest/dinu/3DRS_stage1/3DRS/vggt")
VGGT_CKPT = Path("/home/guest/dinu/3DRS_stage1/3DRS/checkpoints/model.pt")
ROOT = Path("/home/guest/dinu/anchorsplat_e0")
N_FRAMES, TEST_EVERY, TEST_OFFSET = 40, 5, 2
W, H = 648, 484  # half of 1296x968
DEV = "cuda"


# ---------------------------------------------------------------- data
def load_mat(p: Path) -> np.ndarray:
    return np.loadtxt(p).reshape(4, 4)


def select_frames(scene: str) -> tuple[list[str], list[str]]:
    d = SCANNET / scene
    ids = [p.stem for p in sorted(d.glob("*.jpg")) if np.isfinite(load_mat(d / f"{p.stem}.txt")).all()]
    pick = [ids[i] for i in np.linspace(0, len(ids) - 1, N_FRAMES, dtype=int)]
    test = pick[TEST_OFFSET::TEST_EVERY]
    train = [f for f in pick if f not in test]
    return train, test


def load_views(scene: str, ids: list[str]) -> dict:
    d = SCANNET / scene
    Kc = load_mat(d / "intrinsic.txt")[:3, :3].copy()
    Kd = load_mat(d / "depth_intrinsic.txt")[:3, :3].copy()
    w0, h0 = Image.open(d / f"{ids[0]}.jpg").size
    Kc[0] *= W / w0
    Kc[1] *= H / h0
    rgb = np.stack([np.asarray(Image.open(d / f"{i}.jpg").convert("RGB").resize((W, H), Image.BICUBIC)) for i in ids])
    depth = np.stack([np.asarray(Image.open(d / f"{i}.png"), dtype=np.float32) / 1000.0 for i in ids])
    c2w = np.stack([load_mat(d / f"{i}.txt") for i in ids])
    return dict(
        rgb=torch.from_numpy(rgb).float().div(255).to(DEV),           # V,H,W,3
        depth=torch.from_numpy(depth).to(DEV),                         # V,480,640
        c2w=torch.from_numpy(c2w).float().to(DEV),
        viewmat=torch.linalg.inv(torch.from_numpy(c2w).float()).to(DEV),
        Kc=torch.from_numpy(Kc).float().to(DEV),
        Kd=torch.from_numpy(Kd).float().to(DEV),
    )


def unproject(depth, K, c2w, stride):
    """depth [h,w] -> world points [n,3], camera depth [n], valid pixel coords."""
    h, w = depth.shape
    v, u = torch.meshgrid(torch.arange(0, h, stride, device=DEV), torch.arange(0, w, stride, device=DEV), indexing="ij")
    d = depth[v, u]
    ok = (d > 0.1) & (d < 10.0)
    u, v, d = u[ok].float(), v[ok].float(), d[ok]
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    pc = torch.stack([x, y, d], -1)
    return pc @ c2w[:3, :3].T + c2w[:3, 3], d


def sample_colors(pts, views):
    """Average colour of world points over the training images that see them."""
    acc = torch.zeros(len(pts), 3, device=DEV)
    cnt = torch.zeros(len(pts), 1, device=DEV)
    for i in range(len(views["rgb"])):
        pc = pts @ views["viewmat"][i, :3, :3].T + views["viewmat"][i, :3, 3]
        z = pc[:, 2]
        uv = pc[:, :2] / z.clamp(min=1e-3)[:, None] * views["Kc"][[0, 1], [0, 1]] + views["Kc"][:2, 2]
        ok = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
        g = torch.stack([uv[:, 0] / (W - 1) * 2 - 1, uv[:, 1] / (H - 1) * 2 - 1], -1)[None, :, None]
        c = F.grid_sample(views["rgb"][i].permute(2, 0, 1)[None], g, align_corners=True)[0, :, :, 0].T
        acc[ok] += c[ok]
        cnt[ok] += 1
    return (acc / cnt.clamp(min=1)).clamp(0, 1)


def align_sim3(Rv, cv, Rg, cg):
    """Sim(3) with gt ~= s * R @ vggt + t. Rotation from the camera rotations
    (centres alone leave it ~6 deg off on near-planar room trajectories), then
    scale and translation from the camera centres by least squares."""
    U, _, Vt = np.linalg.svd(sum(Rg[i] @ Rv[i].T for i in range(len(Rv))))
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    xv, xg = cv - cv.mean(0), cg - cg.mean(0)
    s = float((xg * (xv @ R.T)).sum() / (xv ** 2).sum())
    return s, R, cg.mean(0) - s * R @ cv.mean(0)


def build_pool(scene, prior, train_ids, views, conf_q=0.0) -> dict:
    """Dense pixel-aligned prior points (world frame) + per-point footprint."""
    if prior == "gt":
        pts, fp = [], []
        for i in range(len(train_ids)):
            p, d = unproject(views["depth"][i], views["Kd"], views["c2w"][i], stride=2)
            pts.append(p)
            fp.append(d * 2 / views["Kd"][0, 0])
        pts, fp = torch.cat(pts), torch.cat(fp)
        info = {}
    elif prior == "vggt":
        sys.path.insert(0, str(VGGT_SRC))
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        model = VGGT()
        model.load_state_dict(torch.load(VGGT_CKPT, map_location="cpu", weights_only=True))
        model = model.to(DEV).eval()
        imgs = load_and_preprocess_images([str(SCANNET / scene / f"{i}.jpg") for i in train_ids]).to(DEV)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(imgs)
            extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], imgs.shape[-2:])
        del model
        torch.cuda.empty_cache()
        depth = pred["depth"].float()[0, ..., 0]                       # S,h,w
        conf = pred["depth_conf"].float()[0]
        extr, intr = extr.float()[0], intr.float()[0]                  # S,3,4 (w2c), S,3,3
        c2w_v = torch.zeros(len(extr), 4, 4, device=DEV)
        c2w_v[:, :3, :3] = extr[:, :3, :3].transpose(1, 2)
        c2w_v[:, :3, 3] = -(extr[:, :3, :3].transpose(1, 2) @ extr[:, :3, 3:])[..., 0]
        c2w_v[:, 3, 3] = 1
        npd = lambda x: x.cpu().double().numpy()
        s, R, t = align_sim3(npd(c2w_v[:, :3, :3]), npd(c2w_v[:, :3, 3]), npd(views["c2w"][:, :3, :3]), npd(views["c2w"][:, :3, 3]))
        R_, t_ = torch.tensor(R, device=DEV).float(), torch.tensor(t, device=DEV).float()
        thr = torch.quantile(conf.flatten()[::97], conf_q) if conf_q > 0 else -float("inf")
        pts, fp = [], []
        for i in range(len(train_ids)):
            dm = torch.where(conf[i] > thr, depth[i], torch.zeros_like(depth[i]))
            p, d = unproject(dm, intr[i], c2w_v[i], stride=2)
            pts.append(s * p @ R_.T + t_)
            fp.append(s * d * 2 / intr[i, 0, 0])
        pts, fp = torch.cat(pts), torch.cat(fp)
        cam_err = (s * c2w_v[:, :3, 3] @ R_.T + t_ - views["c2w"][:, :3, 3]).norm(dim=1)
        rot_err = torch.rad2deg(torch.arccos(((torch.einsum("ij,njk,nlk->nil", R_, c2w_v[:, :3, :3], views["c2w"][:, :3, :3])
                                                .diagonal(dim1=1, dim2=2).sum(1) - 1) / 2).clamp(-1, 1)))
        info = dict(sim3_scale=float(s), cam_center_err_mean_m=float(cam_err.mean()), cam_center_err_max_m=float(cam_err.max()),
                    cam_rot_err_mean_deg=float(rot_err.mean()), cam_rot_err_max_deg=float(rot_err.max()))
    else:
        raise ValueError(prior)
    return dict(pts=pts, fp=fp, rgb=sample_colors(pts, views), info=info)


# ---------------------------------------------------------------- arms
def fps(x: torch.Tensor, n: int) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(0)
    idx = torch.empty(n, dtype=torch.long, device=DEV)
    dist = torch.full((len(x),), float("inf"), device=DEV)
    far = torch.randint(len(x), (1,), device=DEV, generator=g)[0]
    for i in range(n):
        idx[i] = far
        dist = torch.minimum(dist, ((x - x[far]) ** 2).sum(1))
        far = torch.argmax(dist)
    return idx


def knn_dist(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    out = torch.empty(len(x), device=DEV)
    for s in range(0, len(x), 4096):
        d = torch.cdist(x[s:s + 4096], x)
        out[s:s + 4096] = d.topk(k + 1, largest=False).values[:, 1:].mean(1)
    return out


def rgb_to_sh0(c):
    return (c - 0.5) / 0.28209479177387814


def make_params(means, scale, rgb):
    n = len(means)
    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means.clone()),
        "scales": torch.nn.Parameter(torch.log(scale.clamp(min=1e-4))[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0], device=DEV).repeat(n, 1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.1, device=DEV))),
        "sh0": torch.nn.Parameter(rgb_to_sh0(rgb)[:, None, :]),
    })


class Model:
    """Gaussian set; for the anchor arm `means` is derived from anchors + offsets."""

    def __init__(self, arm, pool, voxel, radius, scene_scale):
        self.arm, self.radius = arm, radius
        self.anchors = None
        if arm == "pixel":
            self.params = make_params(pool["pts"], pool["fp"], pool["rgb"])
            self.n_anchor = None
            return
        n = int(torch.unique(torch.floor(pool["pts"] / voxel).long(), dim=0).shape[0])
        idx = fps(pool["pts"], n)
        A, C = pool["pts"][idx], pool["rgb"][idx]
        self.n_anchor = n
        if arm == "3dgs":
            self.params = make_params(A, knn_dist(A), C)
            return
        g = torch.Generator(device=DEV).manual_seed(1)
        off = torch.randn(n, 4, 3, device=DEV, generator=g) * 0.5      # breaks symmetry of the 4 children
        A4, C4 = A[:, None].expand(n, 4, 3).reshape(-1, 3), C[:, None].expand(n, 4, 3).reshape(-1, 3)
        scale = torch.full((4 * n,), voxel / 2, device=DEV)
        if arm == "anchor":
            self.anchors = A4
            self.params = make_params(torch.zeros_like(A4), scale, C4)
            self.params["means"].data = off.reshape(-1, 3)              # raw offsets
        elif arm == "anchor_free":
            self.params = make_params(A4 + radius * torch.tanh(off.reshape(-1, 3)), scale, C4)
        else:
            raise ValueError(arm)

    def means(self):
        if self.anchors is not None:
            return self.anchors + self.radius * torch.tanh(self.params["means"])
        return self.params["means"]

    def render(self, viewmats, Ks, w, h, mode="RGB+ED"):
        from gsplat import rasterization

        p = self.params
        return rasterization(
            self.means(), p["quats"], torch.exp(p["scales"]), torch.sigmoid(p["opacities"]), p["sh0"],
            viewmats, Ks, w, h, sh_degree=0, render_mode=mode, packed=False,
        )


def optimizers_for(model, scene_scale, iters):
    mean_lr = 1.6e-4 * scene_scale
    if model.arm == "anchor":
        mean_lr = mean_lr / model.radius                             # same metric step size through tanh
    lrs = dict(means=mean_lr, scales=5e-3, quats=1e-3, opacities=5e-2, sh0=2.5e-3)
    opts = {k: torch.optim.Adam([model.params[k]], lr=lr, eps=1e-15) for k, lr in lrs.items()}
    sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1.0 / iters))
    return opts, sched


# ---------------------------------------------------------------- fit / eval
def fit(model, train, scene_scale, iters, log_every=1000):
    from gsplat.strategy import DefaultStrategy
    from pytorch_msssim import ssim

    opts, sched = optimizers_for(model, scene_scale, iters)
    strategy = state = None
    if model.arm == "3dgs":
        strategy = DefaultStrategy(refine_stop_iter=int(iters * 0.75), verbose=False)
        strategy.check_sanity(model.params, opts)
        state = strategy.initialize_state(scene_scale=scene_scale)
    g = torch.Generator().manual_seed(0)
    V = len(train["rgb"])
    t0 = time.time()
    for step in range(iters):
        i = int(torch.randint(V, (1,), generator=g))
        out, _, info = model.render(train["viewmat"][i:i + 1], train["Kc"][None], W, H, mode="RGB")
        pred, gt = out[0], train["rgb"][i]
        if strategy:
            strategy.step_pre_backward(model.params, opts, state, step, info)
        loss = 0.8 * (pred - gt).abs().mean() + 0.2 * (1 - ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], data_range=1.0))
        loss.backward()
        if strategy:
            strategy.step_post_backward(model.params, opts, state, step, info, packed=False)
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        sched.step()
        if log_every and step % log_every == 0:
            print(f"  [{model.arm}] step {step} loss {loss.item():.4f} n={len(model.params['means'])}", flush=True)
    torch.cuda.synchronize()
    return time.time() - t0


@torch.no_grad()
def evaluate(model, views, lpips_fn, save_png=None):
    from pytorch_msssim import ssim

    res = dict(psnr=[], ssim=[], lpips=[], absrel=[], delta1=[], coverage=[])
    tiles = []
    for i in range(len(views["rgb"])):
        out, _, _ = model.render(views["viewmat"][i:i + 1], views["Kc"][None], W, H, mode="RGB")
        pred, gt = out[0].clamp(0, 1), views["rgb"][i]
        res["psnr"].append(float(-10 * torch.log10(((pred - gt) ** 2).mean())))
        a, b = pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]
        res["ssim"].append(float(ssim(a, b, data_range=1.0)))
        res["lpips"].append(float(lpips_fn(a * 2 - 1, b * 2 - 1)))
        dimg, alpha, _ = model.render(views["viewmat"][i:i + 1], views["Kd"][None], 640, 480, mode="ED")
        dp, al, dg = dimg[0, ..., 0], alpha[0, ..., 0], views["depth"][i]
        valid = (dg > 0.1) & (dg < 10.0)
        cov = valid & (al > 0.5)
        res["coverage"].append(float(cov.sum() / valid.sum().clamp(min=1)))
        if cov.any():
            r = (dp[cov] - dg[cov]).abs() / dg[cov]
            res["absrel"].append(float(r.mean()))
            res["delta1"].append(float((torch.maximum(dp[cov] / dg[cov], dg[cov] / dp[cov].clamp(min=1e-6)) < 1.25).float().mean()))
        if save_png is not None and i < 2:
            tiles.append(torch.cat([gt, pred], 1))
    if save_png is not None and tiles:
        Image.fromarray((torch.cat(tiles, 0).cpu().numpy() * 255).astype(np.uint8)).save(save_png)
    return {k: float(np.mean(v)) if v else None for k, v in res.items()}


CROP = 448


def load_crop(scene: str, ids: list[str], views: dict) -> dict:
    """AnySplat's view of a frame: resize short side to 448 and centre-crop
    448x448 (same PIL calls as anysplat src/utils/image.py:process_image).
    Returns the crops, their pinhole K, and sensor depth resampled into them
    (same camera centre, so only intrinsics differ)."""
    d = SCANNET / scene
    w0, h0 = Image.open(d / f"{ids[0]}.jpg").size
    assert w0 > h0
    nw = int(w0 * (CROP / h0))
    left = (nw - CROP) // 2
    sx, sy = nw / w0, CROP / h0
    K0 = load_mat(d / "intrinsic.txt")[:3, :3]
    K = np.array([[K0[0, 0] * sx, 0, (K0[0, 2] + 0.5) * sx - 0.5 - left],
                  [0, K0[1, 1] * sy, (K0[1, 2] + 0.5) * sy - 0.5],
                  [0, 0, 1]])
    rgb = np.stack([np.asarray(Image.open(d / f"{i}.jpg").convert("RGB").resize((nw, CROP)))[:, left:left + CROP] for i in ids])
    K = torch.from_numpy(K).float().to(DEV)
    Kd = views["Kd"]
    v, u = torch.meshgrid(torch.arange(CROP, device=DEV), torch.arange(CROP, device=DEV), indexing="ij")
    ud = ((u - K[0, 2]) / K[0, 0] * Kd[0, 0] + Kd[0, 2]).round().long()
    vd = ((v - K[1, 2]) / K[1, 1] * Kd[1, 1] + Kd[1, 2]).round().long()
    inside = (ud >= 0) & (ud < 640) & (vd >= 0) & (vd < 480)
    depth = views["depth"][:, vd.clamp(0, 479), ud.clamp(0, 639)] * inside
    return dict(rgb=torch.from_numpy(rgb).float().div(255).to(DEV), K=K, depth=depth)


@torch.no_grad()
def evaluate_crop(render_fn, crop, lpips_fn, save_png=None):
    """render_fn(i) -> (rgb [S,S,3], expected depth [S,S] in metres, alpha [S,S])."""
    from pytorch_msssim import ssim

    res = dict(psnr=[], ssim=[], lpips=[], absrel=[], delta1=[], coverage=[])
    tiles = []
    for i in range(len(crop["rgb"])):
        pred, dp, al = render_fn(i)
        pred, gt = pred.clamp(0, 1), crop["rgb"][i]
        res["psnr"].append(float(-10 * torch.log10(((pred - gt) ** 2).mean())))
        a, b = pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]
        res["ssim"].append(float(ssim(a, b, data_range=1.0)))
        res["lpips"].append(float(lpips_fn(a * 2 - 1, b * 2 - 1)))
        dg = crop["depth"][i]
        valid = (dg > 0.1) & (dg < 10.0)
        cov = valid & (al > 0.5)
        res["coverage"].append(float(cov.sum() / valid.sum().clamp(min=1)))
        if cov.any():
            res["absrel"].append(float(((dp[cov] - dg[cov]).abs() / dg[cov]).mean()))
            res["delta1"].append(float((torch.maximum(dp[cov] / dg[cov], dg[cov] / dp[cov].clamp(min=1e-6)) < 1.25).float().mean()))
        if save_png is not None and i < 2:
            tiles.append(torch.cat([gt, pred], 1))
    if save_png is not None and tiles:
        Image.fromarray((torch.cat(tiles, 0).cpu().numpy() * 255).astype(np.uint8)).save(save_png)
    return {k: float(np.mean(v)) if v else None for k, v in res.items()}


def so3_exp(w):
    th = w.norm().clamp(min=1e-12)
    k = w / th
    Kx = torch.zeros(3, 3, device=w.device)
    Kx[0, 1], Kx[0, 2], Kx[1, 0], Kx[1, 2], Kx[2, 0], Kx[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, device=w.device) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


def refine_pose(render_rgb, c2w0, gt, steps=300, lr=2e-3):
    """Test-time pose alignment (as in pose-free NVS evaluation): frozen Gaussians,
    optimise a 6-DoF correction of one camera against the held-out image (L1)."""
    w = torch.full((3,), 1e-6, device=DEV, requires_grad=True)
    t = torch.zeros(3, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([w, t], lr=lr)

    def compose():
        T = torch.eye(4, device=DEV)
        T = T.index_put((torch.arange(3)[:, None], torch.arange(3)[None]), so3_exp(w))
        T = T.index_put((torch.arange(3), torch.full((3,), 3)), t)
        return c2w0 @ T

    with torch.enable_grad():
        for _ in range(steps):
            loss = (render_rgb(compose()) - gt).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return compose()


def model_crop_renderer(model, views, crop, refine=False):
    def rgbd(c2w):
        out, alpha, _ = model.render(torch.linalg.inv(c2w)[None], crop["K"][None], CROP, CROP, mode="RGB+ED")
        return out[0, ..., :3], out[0, ..., 3], alpha[0, ..., 0]

    def fn(i):
        c2w = views["c2w"][i]
        if refine:
            c2w = refine_pose(lambda c: rgbd(c)[0], c2w, crop["rgb"][i])
        return rgbd(c2w)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--prior", choices=["gt", "vggt"], required=True)
    ap.add_argument("--arms", default="pixel,anchor,anchor_free,3dgs")
    ap.add_argument("--voxel", type=float, default=0.04)
    ap.add_argument("--radius_mult", type=float, default=2.0, help="anchor offset range r = radius_mult * voxel")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--conf_q", type=float, default=0.0, help="drop VGGT pixels below this depth-confidence quantile")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    import lpips

    torch.manual_seed(0)
    out = Path(args.out) / f"{args.scene}_{args.prior}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    train_ids, test_ids = select_frames(args.scene)
    train, test = load_views(args.scene, train_ids), load_views(args.scene, test_ids)
    test_crop = load_crop(args.scene, test_ids, test)
    scene_scale = float((train["c2w"][:, :3, 3] - train["c2w"][:, :3, 3].mean(0)).norm(dim=1).max() * 1.1)
    t0 = time.time()
    pool = build_pool(args.scene, args.prior, train_ids, train, args.conf_q)
    pool_time = time.time() - t0
    print(f"{args.scene} {args.prior}: pool {len(pool['pts'])} pts in {pool_time:.1f}s {pool['info']}", flush=True)
    lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(DEV)
    radius = args.radius_mult * args.voxel
    for arm in args.arms.split(","):
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        model = Model(arm, pool, args.voxel, radius, scene_scale)
        init_time = time.time() - t1
        n_init = len(model.params["means"])
        fit_time = fit(model, train, scene_scale, args.iters)
        rec = dict(
            conf_q=args.conf_q, scene=args.scene, prior=args.prior, arm=arm, voxel=args.voxel, radius=radius, iters=args.iters,
            train_ids=train_ids, test_ids=test_ids, scene_scale=scene_scale, pool_points=len(pool["pts"]),
            prior_info=pool["info"], n_anchor=model.n_anchor, n_gauss_init=n_init,
            n_gauss_final=len(model.params["means"]), init_s=init_time, fit_s=fit_time,
            peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9,
            test=evaluate(model, test, lpips_fn, save_png=out / f"{arm}_test.png"),
            train=evaluate(model, train, lpips_fn),
            test_crop448=evaluate_crop(model_crop_renderer(model, test, test_crop), test_crop, lpips_fn),
            test_crop448_refined=evaluate_crop(model_crop_renderer(model, test, test_crop, refine=True), test_crop, lpips_fn),
        )
        (out / f"{arm}.json").write_text(json.dumps(rec, indent=1))
        t = rec["test"]
        print(f"{args.scene} {args.prior} {arm}: N={rec['n_gauss_final']} PSNR {t['psnr']:.2f} SSIM {t['ssim']:.3f} "
              f"LPIPS {t['lpips']:.3f} AbsRel {t['absrel']:.3f} d1 {t['delta1']:.3f} cov {t['coverage']:.3f} "
              f"train PSNR {rec['train']['psnr']:.2f} fit {fit_time:.0f}s | crop448 PSNR {rec['test_crop448']['psnr']:.2f} "
              f"AbsRel {rec['test_crop448']['absrel']:.3f} | refined PSNR {rec['test_crop448_refined']['psnr']:.2f} "
              f"AbsRel {rec['test_crop448_refined']['absrel']:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
