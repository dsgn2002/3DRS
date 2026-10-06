"""Aggregate E0 results: per (prior, arm) means over scenes and paired
scene-level differences vs the pixel arm with a bootstrap 95% CI."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/guest/dinu/anchorsplat_e0/results")
recs = [json.loads(p.read_text()) for p in root.glob("scene*/*.json")]
by = defaultdict(dict)
for r in recs:
    by[(r["prior"], r["arm"])][r["scene"]] = r

METRICS = ["psnr", "ssim", "lpips", "absrel", "delta1", "coverage"]
ARMS = ["pixel", "anchor", "anchor_free", "3dgs"]
rng = np.random.default_rng(0)


def boot(d):
    d = np.asarray(d)
    m = rng.choice(d, (10000, len(d))).mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


summary = {}
for prior in ["gt", "vggt"]:
    print(f"\n=== prior: {prior}")
    print(f"{'arm':12s} {'n':>2s} " + " ".join(f"{m:>8s}" for m in METRICS) + f" {'trainPSNR':>9s} {'#GS':>10s} {'fit_s':>6s}")
    for arm in ARMS:
        rs = by.get((prior, arm), {})
        if not rs:
            continue
        row = {m: float(np.mean([r["test"][m] for r in rs.values()])) for m in METRICS}
        row.update(train_psnr=float(np.mean([r["train"]["psnr"] for r in rs.values()])),
                   n_gauss=float(np.mean([r["n_gauss_final"] for r in rs.values()])),
                   fit_s=float(np.mean([r["fit_s"] for r in rs.values()])), n_scenes=len(rs))
        summary[f"{prior}/{arm}"] = row
        print(f"{arm:12s} {len(rs):2d} " + " ".join(f"{row[m]:8.3f}" for m in METRICS)
              + f" {row['train_psnr']:9.2f} {row['n_gauss']:10.0f} {row['fit_s']:6.0f}")
    base = by.get((prior, "pixel"), {})
    for arm in ARMS[1:]:
        rs = by.get((prior, arm), {})
        common = sorted(set(rs) & set(base))
        if len(common) < 2:
            continue
        parts = []
        for m in ["psnr", "lpips", "absrel"]:
            mu, lo, hi = boot([rs[s]["test"][m] - base[s]["test"][m] for s in common])
            summary[f"{prior}/{arm}-pixel/{m}"] = [mu, lo, hi]
            parts.append(f"{m} {mu:+.3f} [{lo:+.3f},{hi:+.3f}]")
        print(f"  {arm} - pixel (n={len(common)}): " + "  ".join(parts))

print("\nper-scene test PSNR / AbsRel")
for (prior, arm), rs in sorted(by.items()):
    print(f"{prior:5s} {arm:12s} " + "  ".join(f"{s[5:9]}:{r['test']['psnr']:.2f}/{r['test']['absrel']:.3f}" for s, r in sorted(rs.items())))
vg = by.get(("vggt", "pixel"), {})
if vg:
    print("\nVGGT alignment:", {s[5:9]: {k: round(v, 3) for k, v in r["prior_info"].items()} for s, r in sorted(vg.items())})
(root / "summary.json").write_text(json.dumps(summary, indent=1))
