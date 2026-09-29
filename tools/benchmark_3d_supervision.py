"""Real-scale GPU benchmark for the 3DRS-G supervision terms.

The CPU self-test proves the maths; this proves the terms are affordable at the
shapes training actually uses (32 frames, 14x14 patches, 3584-d hidden states,
2048-d VGGT teacher, bf16) and reports what each one costs in time and memory.
Decide loss weights after seeing this, not before.

    python tools/benchmark_3d_supervision.py --frames 32 --iters 20
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tds = _load("three_d_supervision", "llava/model/three_d_supervision.py")

GRID = 14
TEACHER_TOKENS = 1036
TEACHER_DIM = 2048


def synth_scene(frames, res, device):
    """Slanted plane with a dropout rectangle, as ScanNet has."""
    v = torch.linspace(-1, 1, res, device=device)
    yy, xx = torch.meshgrid(v, v, indexing="ij")
    world, depth = [], []
    for f in range(frames):
        z = 3.0 / (0.2 * xx + 0.1 * yy + 1.0)
        pts = torch.stack([xx * z + 0.35 * f, yy * z, z], dim=-1)
        d = z.clone()
        d[100:160, 200:280] = 0.0
        world.append(pts)
        depth.append(d)
    return torch.stack(world), torch.stack(depth)


class Config:
    three_d_patch_grid = GRID
    three_d_align_weight = 1.0
    three_d_relational_weight = 0.5
    three_d_geo_weight = 1.0
    three_d_geo_normal_weight = 1.0
    three_d_geo_depth_weight = 0.5
    three_d_geo_confidence = False
    three_d_corr_weight = 0.5
    three_d_corr_voxel_size = 0.2
    three_d_corr_temperature = 0.07
    three_d_corr_anchors = 256


def build_model(hidden, device, dtype):
    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_3d = nn.Sequential(
                nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, TEACHER_DIM)
            )
            self.corres_linear = nn.Linear(hidden, hidden, bias=False)
            self.geo_heads = tds.GeometryHeads(hidden)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.config = Config()

    return Model().to(device=device, dtype=dtype)


def run(model, hs, video_dict, span, terms):
    for name in ("three_d_align_weight", "three_d_relational_weight",
                 "three_d_geo_weight", "three_d_corr_weight"):
        setattr(Config, name, 0.0)
    for name, weight in terms.items():
        setattr(Config, name, weight)
    losses = tds.compute_3d_supervision(model, video_dict, hs, [0], [span])
    total = sum(losses.values())
    total.backward()
    return {k: v.item() for k, v in losses.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=3584)   # Qwen2-7B
    parser.add_argument("--res", type=int, default=384)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("needs CUDA")
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)

    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | {args.dtype}")
    print(f"{args.frames} frames, {GRID}x{GRID} patches = "
          f"{args.frames * GRID * GRID} visual tokens, hidden {args.hidden}\n")

    model = build_model(args.hidden, device, dtype)
    world, depth = synth_scene(args.frames, args.res, device)
    span = args.frames * GRID * (GRID + 1)
    video_dict = {
        "feature_3d": torch.randn(1, args.frames, TEACHER_TOKENS, TEACHER_DIM,
                                  device=device, dtype=dtype),
        "world_coords": world.unsqueeze(0),
        "cam_depth": depth.unsqueeze(0).half(),
    }

    configs = {
        "align (baseline)": {"three_d_align_weight": 1.0},
        "+ relational": {"three_d_align_weight": 1.0, "three_d_relational_weight": 0.5},
        "+ geometry": {"three_d_align_weight": 1.0, "three_d_geo_weight": 1.0},
        "+ correspondence": {"three_d_align_weight": 1.0, "three_d_corr_weight": 0.5},
        "all terms": {"three_d_align_weight": 1.0, "three_d_relational_weight": 0.5,
                      "three_d_geo_weight": 1.0, "three_d_corr_weight": 0.5},
    }

    print(f"{'config':<20} {'ms/iter':>9} {'peak MB':>9}   terms")
    baseline_ms = None
    for label, terms in configs.items():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        hs = torch.randn(1, span, args.hidden, device=device, dtype=dtype, requires_grad=True)

        values = run(model, hs, video_dict, span, terms)          # warm-up
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iters):
            model.zero_grad(set_to_none=True)
            hs.grad = None
            values = run(model, hs, video_dict, span, terms)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) / args.iters * 1000
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        if baseline_ms is None:
            baseline_ms = ms
        delta = "" if label.startswith("align") else f"  (+{ms - baseline_ms:.1f}ms)"
        print(f"{label:<20} {ms:9.1f} {peak:9.0f}   "
              + " ".join(f"{k}={v:.3f}" for k, v in values.items()) + delta)

    print("\nA 7B training step is ~1-2s, so judge the deltas against that.")


if __name__ == "__main__":
    main()
