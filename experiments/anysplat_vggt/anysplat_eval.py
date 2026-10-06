"""Zero-shot AnySplat (lhjiang/anysplat) on the E0 ScanNet split.

Context = the 32 E0 training frames, held-out = the 8 E0 test frames, all as
AnySplat 448x448 centre crops. Two protocols:

  A  official (anysplat src/eval_nvs.py): target poses from AnySplat's camera
     head run on context+target images, translation rescaled to the context
     poses; RGB metrics only.
  B  GT frame: Sim(3) from AnySplat's predicted context cameras to the GT ones
     (rotation averaging, then scale/shift from centres, as for the E0 VGGT
     prior). GT test cameras are mapped into AnySplat's frame and rendered with
     AnySplat's predicted intrinsics (its Gaussians are built for its own focal
     length; GT-intrinsics variant kept as protoB_test_gtK).
  Depth (both protocols) is scaled to metres with the Sim(3) scale and compared
  with sensor depth; metrics match E0's test_crop448 block.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ANYSPLAT = Path("/home/guest/dinu/anysplat")
sys.path.insert(0, str(ANYSPLAT))
sys.path.insert(0, str(Path(__file__).parent))
import e0  # noqa: E402
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402
from src.model.model.anysplat import AnySplat  # noqa: E402
from src.utils.image import process_image  # noqa: E402

DEV = "cuda"
S = e0.CROP


def to_c2w(extr34):
    pad = torch.tensor([0, 0, 0, 1.0], device=extr34.device).view(1, 1, 1, 4).expand(*extr34.shape[:2], 1, 4)
    return torch.cat([extr34, pad], 2).inverse()


def render(model, gaussians, c2w, Knorm):
    """AnySplat decoder; returns rgb [V,S,S,3], expected depth [V,S,S] (AnySplat units), alpha [V,S,S]."""
    V = len(c2w)
    out = model.decoder.forward(gaussians, c2w[None], Knorm[None], torch.full((1, V), 0.01, device=DEV),
                                torch.full((1, V), 100.0, device=DEV), (S, S))
    rgb = out.color[0].permute(0, 2, 3, 1)
    acc_d, alpha = out.depth[0].reshape(V, S, S), out.alpha[0].reshape(V, S, S)
    return rgb, acc_d / alpha.clamp(min=1e-6), alpha


@torch.no_grad()
def run_scene(model, scene, lpips_fn, out_dir):
    train_ids, test_ids = e0.select_frames(scene)
    train, test = e0.load_views(scene, train_ids), e0.load_views(scene, test_ids)
    crop_tr, crop_te = e0.load_crop(scene, train_ids, train), e0.load_crop(scene, test_ids, test)
    jpg = lambda i: str(e0.SCANNET / scene / f"{i}.jpg")
    ctx = (torch.stack([process_image(jpg(i)) for i in train_ids])[None].to(DEV) + 1) * 0.5
    tgt = (torch.stack([process_image(jpg(i)) for i in test_ids])[None].to(DEV) + 1) * 0.5
    assert torch.allclose(ctx[0].permute(0, 2, 3, 1), crop_tr["rgb"], atol=1 / 255 + 1e-6), "crop mismatch vs e0.load_crop"

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    gaussians, pred = model.inference(ctx)
    torch.cuda.synchronize()
    recon_s = time.time() - t0
    n_gauss = int(gaussians.means.shape[1])
    c2w_ctx = pred["extrinsic"][0].float()                                   # V,4,4 camera-to-world
    K_ctx = pred["intrinsic"][0].float()                                     # V,3,3 normalised

    # ---- protocol A (official eval_nvs.py)
    nc = ctx.shape[1]
    allimg = torch.cat([ctx, tgt], 1).to(torch.bfloat16)
    with torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
        toks, _ = model.encoder.aggregator(allimg, intermediate_layer_idx=model.encoder.cfg.intermediate_layer_idx)
    with torch.cuda.amp.autocast(enabled=False):
        enc = model.encoder.camera_head([t.float() for t in toks])[-1]
        extr, intr = pose_encoding_to_extri_intri(enc, allimg.shape[-2:])
    c2w_all = to_c2w(extr.float())
    intr = intr.float().clone()
    intr[:, :, 0] /= S
    intr[:, :, 1] /= S
    scale = c2w_ctx[None, :, :3, 3].mean() / c2w_all[:, :nc, :3, 3].mean()
    c2w_tgt_A = c2w_all[0, nc:].clone()
    c2w_tgt_A[:, :3, 3] *= scale
    rgbA, dA, aA = render(model, gaussians, c2w_tgt_A, intr[0, nc:])

    # ---- protocol B (GT frame)
    npd = lambda x: x.cpu().double().numpy()
    s, R, t = e0.align_sim3(npd(c2w_ctx[:, :3, :3]), npd(c2w_ctx[:, :3, 3]), npd(train["c2w"][:, :3, :3]), npd(train["c2w"][:, :3, 3]))
    R_, t_ = torch.tensor(R, device=DEV).float(), torch.tensor(t, device=DEV).float()
    cam_err = (s * c2w_ctx[:, :3, 3] @ R_.T + t_ - train["c2w"][:, :3, 3]).norm(dim=1)
    rot_err = torch.rad2deg(torch.arccos(((R_ @ c2w_ctx[:, :3, :3] @ train["c2w"][:, :3, :3].transpose(1, 2))
                                          .diagonal(dim1=1, dim2=2).sum(1) - 1).div(2).clamp(-1, 1)))
    Knorm = crop_te["K"].clone()
    Knorm[0] /= S
    Knorm[1] /= S

    def gt_to_any(c2w_gt):
        c = c2w_gt.clone()
        c[:, :3, :3] = R_.T @ c2w_gt[:, :3, :3]
        c[:, :3, 3] = (c2w_gt[:, :3, 3] - t_) @ R_ / s
        return c

    # A: AnySplat's own target poses; its depth in metres via the Sim(3) scale
    protoA = e0.evaluate_crop(lambda i: (rgbA[i], dA[i] * s, aA[i]), crop_te, lpips_fn, save_png=out_dir / f"{scene}_A.png")
    # B: GT poses. AnySplat built its Gaussians for its own focal length, so render
    # with its predicted intrinsics (mean over context views) as the main variant
    # and with GT intrinsics as a secondary one.
    K_pred = K_ctx.mean(0)
    res = {}
    for name, views, crop, K in (("test", test, crop_te, K_pred), ("train", train, crop_tr, K_pred), ("test_gtK", test, crop_te, Knorm)):
        rgbB, dB, aB = render(model, gaussians, gt_to_any(views["c2w"]), K.expand(len(views["c2w"]), 3, 3).contiguous())
        res[name] = e0.evaluate_crop(lambda i: (rgbB[i], dB[i] * s, aB[i]), crop, lpips_fn,
                                     save_png=out_dir / f"{scene}_B_{name}.png" if name.startswith("test") else None)

    # B_refined (main): Sim(3)-mapped GT pose + per-view test-time pose refinement,
    # AnySplat intrinsics. Same refinement as E0's test_crop448_refined.
    c2w_te = gt_to_any(test["c2w"])

    def refined(i):
        one = lambda c: render(model, gaussians, c[None], K_pred[None])
        c = e0.refine_pose(lambda c: one(c)[0][0], c2w_te[i], crop_te["rgb"][i])
        rgb, d, a = one(c)
        return rgb[0], d[0] * s, a[0]
    res["test_refined"] = e0.evaluate_crop(refined, crop_te, lpips_fn, save_png=out_dir / f"{scene}_B_refined.png")
    rec = dict(
        scene=scene, train_ids=train_ids, test_ids=test_ids, n_gauss=n_gauss, recon_s=recon_s,
        peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9,
        sim3_scale=float(s), cam_center_err_mean_m=float(cam_err.mean()), cam_center_err_max_m=float(cam_err.max()),
        cam_rot_err_mean_deg=float(rot_err.mean()), cam_rot_err_max_deg=float(rot_err.max()),
        pred_fx_over_gt=float(K_ctx[:, 0, 0].mean() * S / crop_te["K"][0, 0]),
        protoA_test=protoA, protoB_test=res["test"], protoB_train=res["train"], protoB_test_gtK=res["test_gtK"], protoB_test_refined=res["test_refined"],
    )
    (out_dir / f"{scene}.json").write_text(json.dumps(rec, indent=1))
    b = rec["protoB_test"]
    print(f"{scene}: N={n_gauss} recon {recon_s:.1f}s mem {rec['peak_mem_gb']:.1f}GB | A PSNR {protoA['psnr']:.2f} LPIPS {protoA['lpips']:.3f} AbsRel {protoA['absrel']:.3f} | "
          f"B PSNR {b['psnr']:.2f} SSIM {b['ssim']:.3f} LPIPS {b['lpips']:.3f} AbsRel {b['absrel']:.3f} d1 {b['delta1']:.3f} "
          f"cov {b['coverage']:.3f} train PSNR {res['train']['psnr']:.2f} gtK PSNR {res['test_gtK']['psnr']:.2f} | refined PSNR {res['test_refined']['psnr']:.2f} SSIM {res['test_refined']['ssim']:.3f} "
          f"LPIPS {res['test_refined']['lpips']:.3f} AbsRel {res['test_refined']['absrel']:.3f} | cam {rec['cam_center_err_mean_m']:.3f}m "
          f"{rec['cam_rot_err_mean_deg']:.1f}deg fx {rec['pred_fx_over_gt']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="scene0015_00,scene0256_00,scene0334_00,scene0338_00,scene0427_00")
    ap.add_argument("--out", default=str(e0.ROOT / "results_anysplat"))
    args = ap.parse_args()
    import lpips

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = AnySplat.from_pretrained("lhjiang/anysplat").to(DEV).eval()
    for p in model.parameters():
        p.requires_grad = False
    lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(DEV)
    for scene in args.scenes.split(","):
        run_scene(model, scene, lpips_fn, out)


if __name__ == "__main__":
    main()
