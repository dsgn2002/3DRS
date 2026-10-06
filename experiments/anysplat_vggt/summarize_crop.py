"""AnySplat (zero-shot, feed-forward) vs E0 per-scene arms on the same 448 crops.
RGB metrics at the test-time-refined pose; depth at the unrefined pose (GT pose for
E0, Sim(3)-mapped GT pose for AnySplat), since refinement moves the camera away
from where the sensor depth was captured."""
import json
from pathlib import Path

import numpy as np

R = Path("/home/guest/dinu/anchorsplat_e0")
rng = np.random.default_rng(0)
rows = {}
for p in (R / "results_anysplat").glob("scene*.json"):
    r = json.loads(p.read_text())
    rows.setdefault("AnySplat (official poses, A)", {})[r["scene"]] = dict(
        psnr=r["protoA_test"]["psnr"], ssim=r["protoA_test"]["ssim"], lpips=r["protoA_test"]["lpips"],
        absrel=r["protoA_test"]["absrel"], delta1=r["protoA_test"]["delta1"], n=r["n_gauss"], t=r["recon_s"])
    rows.setdefault("AnySplat (GT pose + refine, B)", {})[r["scene"]] = dict(
        psnr=r["protoB_test_refined"]["psnr"], ssim=r["protoB_test_refined"]["ssim"], lpips=r["protoB_test_refined"]["lpips"],
        absrel=r["protoB_test"]["absrel"], delta1=r["protoB_test"]["delta1"], n=r["n_gauss"], t=r["recon_s"])
for p in (R / "results_crop").glob("scene*/*.json"):
    r = json.loads(p.read_text())
    rf, ur = r["test_crop448_refined"], r["test_crop448"]
    rows.setdefault(f"E0 {r['prior']} {r['arm']}", {})[r["scene"]] = dict(
        psnr=rf["psnr"], ssim=rf["ssim"], lpips=rf["lpips"], absrel=ur["absrel"], delta1=ur["delta1"],
        n=r["n_gauss_final"], t=r["fit_s"], psnr_unrefined=ur["psnr"])

M = ["psnr", "ssim", "lpips", "absrel", "delta1"]
print(f"{'method':32s} {'n':>2s} " + " ".join(f"{m:>7s}" for m in M) + f" {'#GS':>9s} {'time_s':>7s}")
for k in sorted(rows):
    v = rows[k]
    print(f"{k:32s} {len(v):2d} " + " ".join(f"{np.mean([x[m] for x in v.values()]):7.3f}" for m in M)
          + f" {np.mean([x['n'] for x in v.values()]):9.0f} {np.mean([x['t'] for x in v.values()]):7.1f}")

ref = rows.get("AnySplat (GT pose + refine, B)", {})
print("\npaired vs AnySplat B (method - AnySplat), bootstrap 95% over scenes")
for k in sorted(rows):
    if not k.startswith("E0"):
        continue
    common = sorted(set(rows[k]) & set(ref))
    if len(common) < 2:
        continue
    parts = []
    for m in ("psnr", "lpips", "absrel"):
        d = np.array([rows[k][s][m] - ref[s][m] for s in common])
        bs = rng.choice(d, (10000, len(d))).mean(1)
        parts.append(f"{m} {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]")
    print(f"  {k:24s} n={len(common)}  " + "  ".join(parts))

print("\nper scene PSNR / AbsRel")
for k in sorted(rows):
    print(f"  {k:32s} " + "  ".join(f"{s[5:9]}:{x['psnr']:.2f}/{x['absrel']:.3f}" for s, x in sorted(rows[k].items())))
any_ = sorted((R / "results_anysplat").glob("scene*.json"))
if any_:
    print("\nAnySplat camera alignment:")
    for p in any_:
        r = json.loads(p.read_text())
        print(f"  {r['scene'][5:9]}: centre {r['cam_center_err_mean_m']:.3f} m, rot {r['cam_rot_err_mean_deg']:.1f} deg, "
              f"fx/gt {r['pred_fx_over_gt']:.3f}, unrefined B PSNR {r['protoB_test']['psnr']:.2f}, mem {r['peak_mem_gb']:.1f} GB")
