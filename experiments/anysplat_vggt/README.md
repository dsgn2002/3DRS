# VGGT prior vs AnySplat on ScanNet (2026-10-06)

Question: how good is the 3D that a feed-forward splatting model (AnySplat) recovers,
compared with Gaussians fitted per scene on a VGGT (or sensor-depth) prior? This is the
reconstruction side of using either model as the 3D source for a VLM.

## Setup

- 5 ScanQA val scenes: 0015, 0256, 0334, 0338, 0427 (ScanNet `posed_images`).
- 40 uniformly spaced frames with finite poses per scene: 32 context, 8 held out (`pick[2::5]`).
- **E0** (`e0.py`): per-scene gsplat fit, 10k iterations, L1 + SSIM, SH degree 0. The prior is sensor depth (`gt`) or
  VGGT run on the 32 context frames, Sim(3)-aligned to the GT cameras by rotation averaging and then
  scale and shift. Arms:
  - `pixel`: one Gaussian per prior point.
  - `anchor`: FPS anchors with 4 offset Gaussians each.
  - `anchor_free`: as `anchor`, without the offset constraint.
  - `3dgs`: one Gaussian per anchor, with densification.
- **AnySplat** (`anysplat_eval.py`): `lhjiang/anysplat`, zero-shot from the 32 uncalibrated context frames
  at its 448x448 centre crops.
  - Protocol A: the official `eval_nvs.py` poses.
  - Protocol B: GT test poses mapped into AnySplat's frame by Sim(3), plus per-view test-time pose refinement.
    Without refinement it scores about 15 PSNR because of its pose and focal-length errors.
    E0 gets the same refinement for fairness.
- Depth is always scored at the unrefined pose against sensor depth (AbsRel). RGB is scored at the refined pose.
- `run_all.sh` runs E0 at full frame. `run_crop.sh` runs AnySplat and the E0 reruns on 448 crops (GPUs 2-3).
  The scripts use hard-coded cez078 paths. Result JSONs are kept on the server.

## Results (448 crops, held-out views, mean of 5 scenes)

| Method | PSNR | SSIM | LPIPS | AbsRel | #Gaussians | Time per scene |
|---|---|---|---|---|---|---|
| AnySplat B (GT pose + refine) | 21.96 | 0.749 | **0.408** | 0.128 | 4.13M | **2.3 s** |
| AnySplat A (official poses) | 19.17 | 0.702 | 0.425 | 0.103 | 4.13M | 2.3 s |
| E0 VGGT pixel | 22.80 | 0.733 | 0.452 | 0.123 | 1.62M | 142 s |
| E0 VGGT anchor | 22.98 | 0.760 | 0.457 | 0.153 | 417k | 139 s |
| E0 GT pixel | 23.53 | 0.751 | 0.426 | **0.060** | 2.27M | 138 s |
| E0 GT anchor | **23.90** | **0.778** | 0.436 | 0.085 | 383k | 143 s |

Paired difference against AnySplat B (bootstrap 95% CI over scenes):

| Arm | ΔPSNR | ΔLPIPS | ΔAbsRel |
|---|---|---|---|
| VGGT pixel | +0.85 [-0.51, +1.90] | +0.044 [+0.017, +0.072] | -0.005 [-0.040, +0.031] |
| VGGT anchor | +1.02 [+0.04, +1.75] | +0.049 [+0.025, +0.080] | +0.025 [-0.022, +0.074] |
| GT pixel | +1.57 [+0.33, +2.52] | +0.019 [-0.001, +0.040] | -0.068 [-0.076, -0.054] |

AnySplat's cameras compared with GT, after Sim(3) alignment:

- Centre error is 9-34 cm and rotation error 2.6-4.0°.
- Predicted fx is 11-13% short on every scene.

## Findings

1. **AnySplat geometry is about as good as VGGT geometry, and no better.** Its depth is equal to the VGGT-prior pixel fit
   (ΔAbsRel -0.005, interval spans 0). This is expected, because its encoder is initialised from VGGT and distilled
   from VGGT. Sensor depth is far better than both (AbsRel 0.060).
2. **AnySplat's advantage is appearance and speed.** It has the best LPIPS of all methods, including the GT-prior fits,
   and it builds a renderable scene in 2.3 s from uncalibrated images, against about 140 s for a per-scene fit
   that needs calibrated cameras.
3. **Its main weakness is cameras.** These errors matter for any VLM tool that maps answers back to the
   ScanNet/GT frame.
4. **Anchoring (E0, full frame) does not explain the AnchorSplat paper's depth claim.** Anchors regularise RGB
   (+0.4-0.5 dB on held-out views) but make depth worse in 10 of 10 runs. The offset constraint has no effect.

Implication for VLMs: rendered novel views (for example top-down or targeted views as a VLM tool) are where AnySplat
could beat a raw VGGT point cloud. Metric geometry (depth, distances, counting by dedup) should not improve.
