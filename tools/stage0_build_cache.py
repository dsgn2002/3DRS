"""Stage 0: cache frozen LLaVA-Video and VGGT features for a slice of ScanNet.

Both encoders are frozen for the whole contrastive-alignment study, so they are
run exactly once and their outputs cached. Everything downstream (probes,
projectors, contrastive ladders) then trains on tensors that fit in RAM, which
is what makes the study affordable on a single 4090.

Per scene it stores:
  * LLaVA-Video visual tokens at several depths. Layer 0 is the mm_projector
    output (what the LLM is handed); the rest are decoder layers. 3DRS
    supervises the last layer, but "is 3D recoverable from a frozen VLM" is a
    question about the whole stack, so we keep several.
  * VGGT tokens pooled to the same 14x14 grid.
  * Per-patch world XYZ, camera depth and a validity mask, from the ScanNet
    depth sensor and poses -- the ground truth every Stage 1 probe regresses.

    python tools/stage0_build_cache.py --num-scenes 24 --out cache/stage0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.model.three_d_supervision import (  # noqa: E402
    extract_visual_tokens,
    infer_patch_layout,
    patch_geometry,
    resample_teacher_features,
)
from llava.video_utils import VideoProcessor, merge_video_dict  # noqa: E402

# Neutral and identical for every scene: we are probing the visual tokens, so
# the text must not be a variable.
PROMPT = ("The video captures 3D spatial information of a scene. "
          "Describe the spatial layout of the scene.")


def select_scenes(video_processor, posed_root, num_scenes, seed=0):
    """Scenes that have frames on disk, EmbodiedScan poses, and box metadata."""
    on_disk = {d for d in os.listdir(posed_root) if os.path.isdir(os.path.join(posed_root, d))}
    usable = sorted(
        vid for vid in video_processor.scene
        if vid.split("/")[-1] in on_disk and vid in video_processor.scan2obj
    )
    rng = np.random.default_rng(seed)
    if len(usable) > num_scenes:
        usable = [usable[i] for i in sorted(rng.choice(len(usable), num_scenes, replace=False))]
    return usable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="data/models/LLaVA-Video-7B-Qwen2")
    ap.add_argument("--posed-root", default="data/scannet/posed_images")
    ap.add_argument("--embodiedscan-folder", default="data/embodiedscan")
    ap.add_argument("--vggt-dir", default="data/scannet/posed_images_3d_feature_vggt")
    ap.add_argument("--out", default="cache/stage0")
    ap.add_argument("--num-scenes", type=int, default=24)
    ap.add_argument("--held-out", type=int, default=8)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--patch-grid", type=int, default=14)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 8, 16, 22, 28],
                    help="hidden_states indices; 0 is the mm_projector output "
                         "handed to the LLM, 28 is the last decoder layer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-from", dest="split_from", default=None,
                    help="reuse the scenes and train/held-out split of an existing "
                         "split.json; random selection drifts as more scenes land on disk")
    ap.add_argument("--select-only", action="store_true",
                    help="write split.json and exit, without loading the 7B; "
                         "lets VGGT extraction run on just the chosen scenes")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    if args.select_only:
        video_processor = VideoProcessor(
            video_folder="data", annotation_dir=args.embodiedscan_folder,
            frame_sampling_strategy="uniform",
        )
        scenes = select_scenes(video_processor, args.posed_root, args.num_scenes, args.seed)
        split = {"train": scenes[args.held_out:], "heldout": scenes[: args.held_out]}
        with open(os.path.join(args.out, "split.json"), "w") as f:
            json.dump(split, f, indent=2)
        with open(os.path.join(args.out, "scenes.txt"), "w") as f:
            f.write("\n".join(s.split("/")[-1] for s in scenes) + "\n")
        print(f"selected {len(scenes)} scenes "
              f"(train {len(split['train'])} / held-out {len(split['heldout'])})")
        for s in scenes:
            print("  ", s)
        return

    print("loading LLaVA-Video (frozen)...")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, get_model_name_from_path(args.model_path)
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    video_processor = VideoProcessor(
        video_folder="data",
        annotation_dir=args.embodiedscan_folder,
        frame_sampling_strategy="uniform",
    )
    if args.split_from:
        split = json.load(open(args.split_from))
        scenes = split["heldout"] + split["train"]
        print(f"{len(scenes)} scenes reused from {args.split_from}")
    else:
        scenes = select_scenes(video_processor, args.posed_root, args.num_scenes, args.seed)
        print(f"{len(scenes)} scenes selected")
        split = {"train": scenes[args.held_out:], "heldout": scenes[: args.held_out]}
    with open(os.path.join(args.out, "split.json"), "w") as f:
        json.dump({k: v for k, v in split.items()}, f, indent=2)
    print(f"  train {len(split['train'])} | held-out {len(split['heldout'])}")

    done, skipped = 0, []
    for video_id in tqdm(scenes):
        scene = video_id.split("/")[-1]
        out_path = os.path.join(args.out, f"{scene}.npz")
        if os.path.exists(out_path):
            done += 1
            continue
        try:
            video_dict = video_processor.process_3d_video(
                video_id, image_processor, force_sample=True, frames_upbound=args.frames
            )
        except (FileNotFoundError, KeyError) as exc:
            skipped.append((scene, f"data: {exc}"))
            continue

        video_dict = merge_video_dict([video_dict])
        images = video_dict.pop("images").half().to(model.device)
        world_coords = video_dict["world_coords"][0].float()
        cam_depth = video_dict["cam_depth"][0].float()

        gpu_dict = {}
        for key in ("world_coords", "cam_depth", "objects", "feature_3d", "box_input"):
            if key in video_dict and torch.is_tensor(video_dict[key]):
                gpu_dict[key] = video_dict[key].half().to(model.device)

        prompt = f"{DEFAULT_IMAGE_TOKEN}\n{PROMPT}"
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model.device)

        with torch.inference_mode():
            (_, position_ids, attention_mask, _, inputs_embeds, _, _, _,
             img_pos_list, img_length_list) = model.prepare_inputs_labels_for_multimodal(
                input_ids, None, None, None, None, images, ["video"],
                image_sizes=None, video_dict=gpu_dict,
            )
            outputs = model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                img_pos_list=img_pos_list,
                img_length_list=img_length_list,
                output_hidden_states=True,
                return_dict=True,
            )

        # Derive the token layout from what the model actually produced rather
        # than assuming 3DRS's training config; grid drives the geometry
        # patchification and the VGGT pooling too, so it must be right.
        # world_coords is (V, H, W, 3), so V is the authoritative frame count;
        # `images` carries a batch dim from merge_video_dict and would give 1.
        n_frames = world_coords.shape[0]
        grid, newline = infer_patch_layout(int(img_length_list[0]), n_frames)
        if grid != args.patch_grid:
            print(f"  note: {scene} uses a {grid}x{grid} grid "
                  f"(newline_per_row={newline}), not {args.patch_grid}")

        payload = {}
        n_layers = len(outputs.hidden_states)
        for layer in args.layers:
            idx = layer if layer < n_layers else n_layers - 1
            tok = extract_visual_tokens(
                outputs.hidden_states[idx], img_pos_list, img_length_list, grid, newline
            )
            payload[f"vlm_l{layer}"] = tok.to(torch.float16).cpu().numpy()
        payload["grid"] = np.array(grid)

        geom = patch_geometry(world_coords, cam_depth, grid)
        payload["points"] = geom["points"].cpu().numpy().astype(np.float32)
        payload["patch_depth"] = geom["depth"].cpu().numpy().astype(np.float32)
        payload["valid"] = geom["valid"].cpu().numpy()

        vggt_path = os.path.join(args.vggt_dir, scene, "vggt.npz")
        if os.path.exists(vggt_path):
            data = np.load(vggt_path)
            feat = torch.from_numpy(data["feature"]).float()
            payload["vggt"] = (
                resample_teacher_features(feat, grid).to(torch.float16).cpu().numpy()
            )
        else:
            skipped.append((scene, "no vggt features"))

        np.savez_compressed(out_path, **payload)
        done += 1
        del outputs
        torch.cuda.empty_cache()

    print(f"\ncached {done} scenes into {args.out}")
    if skipped:
        print("skipped:")
        for scene, why in skipped:
            print(f"  {scene}: {why}")


if __name__ == "__main__":
    main()
