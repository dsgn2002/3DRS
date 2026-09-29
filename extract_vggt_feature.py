"""Extract frozen VGGT features as 3DRS distillation targets.

Changes from the released version:

  * **Pools to the student's 14x14 grid before saving.** The training loss runs
    ``adaptive_avg_pool2d(..., (14, 14))`` on these features as its first act,
    so storing all 1036 patch tokens keeps 5.3x more data than any loss will
    ever read. Pooling here is numerically identical and cuts the feature store
    from ~410GB to ~39GB over ScanNet, with proportionally less dataloader I/O.
  * **Frame selection comes from EmbodiedScan, not os.listdir.** The dataloader
    samples its 32 frames from ``meta_info["images"]``; the released extractor
    sampled from a sorted directory listing. If those two orders ever disagree,
    every feature is silently paired with the wrong frame and training still
    looks healthy. Using one source makes the alignment structural.
  * fp16 storage, resumable, and shards across all visible GPUs.

    torchrun --nproc_per_node=4 extract_vggt_feature.py     # or plain python
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# The repo vendors the whole VGGT *repository* at ./vggt, so the importable
# package is nested one level down at ./vggt/vggt. Without this the imports
# below only resolve when cwd happens to be ./vggt.
sys.path.insert(0, str(Path(__file__).resolve().parent / "vggt"))

from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402

PATCH_GRID = 14


def pool_to_grid(tokens, grid=PATCH_GRID):
    """(S, L, D) patch tokens -> (S, grid*grid, D), matching the training loss."""
    S, L, D = tokens.shape
    shapes = {1036: (28, 37), 768: (24, 32), 256: (16, 16)}
    if L not in shapes:
        raise NotImplementedError(f"unknown VGGT token count L={L}")
    h, w = shapes[L]
    x = tokens.view(S, h, w, D).permute(0, 3, 1, 2).float()
    x = F.adaptive_avg_pool2d(x, (grid, grid))
    return x.view(S, D, grid * grid).permute(0, 2, 1).contiguous()


def frame_lists(root_dir, num_frames, embodiedscan_dir, video_folder):
    """Map scene -> the exact frames the dataloader will sample.

    Falls back to a sorted directory listing only if EmbodiedScan metadata is
    unavailable, and says so loudly, because that path is the one that can
    silently misalign features and frames.
    """
    scenes = sorted(s for s in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, s)))
    infos = {}
    if embodiedscan_dir and os.path.isdir(embodiedscan_dir):
        import pickle
        for split in ("train", "val", "test"):
            path = os.path.join(embodiedscan_dir, f"embodiedscan_infos_{split}.pkl")
            if not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                for item in pickle.load(f)["data_list"]:
                    if item["sample_idx"].startswith("scannet"):
                        infos[item["sample_idx"].split("/")[-1]] = item
    else:
        print("WARNING: no EmbodiedScan metadata; falling back to directory order. "
              "Features may not correspond to the frames training samples.")

    out = {}
    for scene in scenes:
        info = infos.get(scene)
        if info is not None:
            files = [os.path.join(video_folder, img["img_path"]) for img in info["images"]]
        else:
            scene_dir = os.path.join(root_dir, scene)
            files = sorted(
                os.path.join(scene_dir, f) for f in os.listdir(scene_dir) if f.endswith(".jpg")
            )
        if not files:
            continue
        # Same rule as VideoProcessor.sample_frame_files with force_sample=True.
        idx = np.linspace(0, len(files) - 1, num=num_frames, dtype=int)
        out[scene] = [files[i] for i in idx]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="data/scannet/posed_images")
    parser.add_argument("--save-dir", default="data/scannet/posed_images_3d_feature_vggt")
    parser.add_argument("--embodiedscan-dir", default="data/embodiedscan")
    parser.add_argument("--video-folder", default="data")
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--patch-grid", type=int, default=PATCH_GRID)
    parser.add_argument("--no-pool", action="store_true",
                        help="store all VGGT tokens (the original behaviour, 5.3x the disk)")
    parser.add_argument("--scenes-file", default=None,
                        help="newline-separated scene ids; restricts extraction "
                             "to those (e.g. the Stage 0 slice)")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="aggregator layer indices to save (e.g. 4 11 17 23, the layers "
                             "VGGT's own DPT depth/point heads read). Default: last layer only, "
                             "stored under 'feature' as before")
    parser.add_argument("--out-name", dest="out_name", default="vggt.npz")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{rank % max(torch.cuda.device_count(), 1)}"
                          if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    model = VGGT()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model = model.to(device).eval()

    os.makedirs(args.save_dir, exist_ok=True)
    scenes = frame_lists(args.root_dir, args.num_frames, args.embodiedscan_dir, args.video_folder)
    if args.scenes_file:
        wanted = {l.strip() for l in open(args.scenes_file) if l.strip()}
        missing = wanted - set(scenes)
        if missing:
            print(f"WARNING: {len(missing)} requested scenes have no frames on disk")
        scenes = {k: v for k, v in scenes.items() if k in wanted}
    names = sorted(scenes)[rank::world_size]
    print(f"[rank {rank}/{world_size}] {len(names)} scenes on {device}")

    for scene in tqdm(names, disable=rank != 0):
        save_path = os.path.join(args.save_dir, scene, args.out_name)
        if os.path.exists(save_path) and not args.overwrite:
            continue
        files = [f for f in scenes[scene] if os.path.exists(f)]
        if len(files) != args.num_frames:
            print(f"skipping {scene}: {len(files)}/{args.num_frames} frames on disk")
            continue

        images = load_and_preprocess_images(files).to(device=device, dtype=dtype).unsqueeze(0)
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
            tokens, ps_idx = model.aggregator(images)

        def grab(layer_tokens):
            f = layer_tokens[:, :, ps_idx:, :].squeeze(0).float()        # (S, L, D)
            if not args.no_pool:
                f = pool_to_grid(f, args.patch_grid)
            return f.cpu().numpy().astype(np.float16)[None]               # (1, S, P, D)

        if args.layers is None:
            arrays = {"feature": grab(tokens[-1])}
        else:
            arrays = {f"feature_l{k}": grab(tokens[k]) for k in args.layers}

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(
            save_path,
            **arrays,
            ps_idx=np.array(0),        # already stripped; kept for loader compatibility
            pooled=np.array(not args.no_pool),
            patch_grid=np.array(args.patch_grid),
            layers=np.array(args.layers if args.layers is not None else [len(tokens) - 1]),
        )


if __name__ == "__main__":
    main()
