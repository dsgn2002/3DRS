"""Is each arm's effect vs the pixel arm the same on training views as on held-out views?"""
import json, sys
from pathlib import Path
root = Path("/home/guest/dinu/anchorsplat_e0/results")
R = {}
for p in root.glob("scene*/*.json"):
    r = json.loads(p.read_text()); R[(r["scene"], r["prior"], r["arm"])] = r
print(f"{'scene':6s} {'prior':5s} {'arm':12s} | {'dPSNR tr':>8s} {'dPSNR te':>8s} | {'dAbsRel tr':>10s} {'dAbsRel te':>10s} | {'gap arm':>7s} {'gap pix':>7s}")
for (s, pr, a), r in sorted(R.items()):
    b = R.get((s, pr, "pixel"))
    if a == "pixel" or b is None: continue
    d = lambda split, m: r[split][m] - b[split][m]
    print(f"{s[5:9]:6s} {pr:5s} {a:12s} | {d('train','psnr'):+8.2f} {d('test','psnr'):+8.2f} | {d('train','absrel'):+10.3f} {d('test','absrel'):+10.3f} | "
          f"{r['train']['psnr']-r['test']['psnr']:7.2f} {b['train']['psnr']-b['test']['psnr']:7.2f}")
