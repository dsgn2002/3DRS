"""Measure how far an MLLM's visual features sit from a 3D foundation model's.

3DRS reports one number for this relationship -- the value of the cosine
distillation loss -- which cannot distinguish "the student encodes the same
geometry in a different basis" from "the student does not encode geometry".
This script reports both, per layer, alongside the paper's own 3D-awareness
proxy and a linear geometry probe, so a training run can be judged on whether
the representation actually improved rather than on whether a loss went down.

Needs the full 3DRS environment (transformers/deepspeed) plus ScanNet, the
pre-extracted VGGT features, and a checkpoint. Run the baseline and the 3DRS-G
checkpoint through it and compare:

    python tools/measure_3d_gap.py \
        --model-path ckpt/llavanext-qwen-3drs \
        --num-scenes 50 --layers -1 -3 -5 \
        --output analysis/gap_baseline.json
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llava.analysis.feature_gap import feature_gap_report, geometry_probe_r2  # noqa: E402
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.model.three_d_supervision import (  # noqa: E402
    extract_visual_tokens,
    patch_geometry,
    resample_teacher_features,
)
from llava.video_utils import VideoProcessor, merge_video_dict  # noqa: E402

PROMPT = (
    "The video captures 3D spatial information of a scene. "
    "Describe the spatial layout of the scene."
)


def scene_hidden_states(model, tokenizer, image_processor, video_processor, video_id, args):
    """Run one scene through the MLLM and return (hidden_states_per_layer, video_dict)."""
    video_dict = video_processor.process_3d_video(
        video_id, image_processor, force_sample=True, frames_upbound=args.max_frame_num
    )
    video_dict = merge_video_dict([video_dict])
    images = video_dict.pop("images").half().to(model.device)
    for key in video_dict:
        video_dict[key] = video_dict[key].half().to(model.device)

    prompt = f"{DEFAULT_IMAGE_TOKEN}\n{PROMPT}"
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    with torch.inference_mode():
        # Mirrors generate()/predict_box(): the multimodal packing is what tells
        # us where the visual tokens landed in the sequence.
        (
            _, position_ids, attention_mask, _, inputs_embeds, _, _, _,
            img_pos_list, img_length_list,
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids, None, None, None, None, images, ["video"],
            image_sizes=None, video_dict=video_dict,
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
    return outputs.hidden_states, video_dict, img_pos_list, img_length_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--video-folder", default="data")
    parser.add_argument("--embodiedscan-folder", default="data/embodiedscan")
    parser.add_argument("--scene-list", default="scripts/3d/preprocessing/scannet_metadata/scannetv2_val.txt")
    parser.add_argument("--num-scenes", type=int, default=50)
    parser.add_argument("--max-frame-num", type=int, default=32, dest="max_frame_num")
    parser.add_argument("--patch-grid", type=int, default=14)
    parser.add_argument("--voxel-size", type=float, default=0.2)
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[-1],
        help="hidden-state indices to probe; the paper supervises -1 but the "
             "gap is worth watching further down too",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    model.eval()

    video_processor = VideoProcessor(
        video_folder=args.video_folder,
        annotation_dir=args.embodiedscan_folder,
        frame_sampling_strategy="uniform",
    )

    with open(args.scene_list) as f:
        scenes = [f"scannet/{line.strip()}" for line in f if line.strip()]
    scenes = [s for s in scenes if s in video_processor.scene][: args.num_scenes]
    print(f"probing {len(scenes)} scenes at layers {args.layers}")

    per_layer = {layer: [] for layer in args.layers}
    for video_id in tqdm(scenes):
        try:
            hidden, video_dict, img_pos, img_len = scene_hidden_states(
                model, tokenizer, image_processor, video_processor, video_id, args
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"skipping {video_id}: {exc}")
            continue

        teacher = resample_teacher_features(
            video_dict["feature_3d"].float(), args.patch_grid
        )
        geometry = patch_geometry(
            video_dict["world_coords"][0].float(),
            video_dict["cam_depth"][0].float(),
            args.patch_grid,
        )
        for layer in args.layers:
            tokens = extract_visual_tokens(
                hidden[layer].float(), img_pos, img_len, args.patch_grid
            )
            # Compared in the projection space, since the raw widths differ and
            # proj_3d is the map the training loss actually optimises.
            student = model.model.proj_3d(tokens.to(model.dtype)).float()
            report = feature_gap_report(
                student, teacher,
                points=geometry["points"], valid=geometry["valid"],
                voxel_size=args.voxel_size,
            )
            # The probe on unprojected features is the honest question: is
            # geometry in the backbone, or only in the head trained to fake it?
            report["probe_r2_backbone"] = float(
                geometry_probe_r2(tokens.float(), geometry["points"], geometry["valid"])
            )
            per_layer[layer].append(report)

    summary = {}
    for layer, reports in per_layer.items():
        if not reports:
            continue
        summary[str(layer)] = {
            key: statistics.fmean(r[key] for r in reports) for key in reports[0]
        }
        print(f"\nlayer {layer}  (n={len(reports)})")
        for key, value in summary[str(layer)].items():
            print(f"  {key:24s} {value: .4f}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"scenes": len(scenes), "summary": summary}, f, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
