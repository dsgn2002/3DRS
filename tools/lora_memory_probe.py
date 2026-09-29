"""Does LoRA fine-tuning of LLaVA-Video-7B fit on one 24GB card?

Earlier in this project that was assumed impossible without ever being run.
This measures it: real model, real 32-frame ScanNet sample, LoRA on every
decoder projection, gradient checkpointing, a full forward + backward +
optimiser step, reporting peak allocated memory and step time.

    python tools/lora_memory_probe.py --frames 32 --bits 16
    python tools/lora_memory_probe.py --frames 32 --bits 4     # QLoRA
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.video_utils import VideoProcessor, merge_video_dict  # noqa: E402

QUESTION = "Where is the chair relative to the table? Answer in one sentence."
ANSWER = "The chair is to the left of the table, about one metre away."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="data/models/LLaVA-Video-7B-Qwen2")
    ap.add_argument("--scene", default="scannet/scene0303_02")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--bits", type=int, default=16, choices=[16, 4])
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--contrast-weight", dest="contrast_weight", type=float, default=0.0,
                    help=">0 builds the contrast heads and adds the contrastive term")
    ap.add_argument("--contrast-layer", dest="contrast_layer", type=int, default=8)
    ap.add_argument("--contrast-hard", dest="contrast_hard", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    torch.cuda.reset_peak_memory_stats()
    # Heads are built in __init__ from the config, so contrastive settings must
    # be on the config before the model is constructed.
    overwrite = None
    if args.contrast_weight > 0:
        overwrite = {"three_d_contrast_weight": args.contrast_weight,
                     "three_d_contrast_layer": args.contrast_layer,
                     "three_d_contrast_hard_weight": args.contrast_hard}
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, get_model_name_from_path(args.model_path),
        load_4bit=args.bits == 4, torch_dtype="bfloat16", overwrite_config=overwrite,
    )
    # The 3D supervision term needs teacher features sampled at the same frame
    # count; they exist for 32 frames only, so other counts measure LM loss alone.
    model.config.three_d_align_weight = 1.0 if args.frames == 32 else 0.0
    load_mem = torch.cuda.memory_allocated() / 2**30

    if args.bits == 4:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, bias="none",
        # Regex over full paths: decoder projections only, never the vision
        # tower, projector, or the randomly initialised 3D heads.
        target_modules=r".*model\.layers\.\d+\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    for name, p in model.named_parameters():
        if "proj_3d" in name or "contrast_heads" in name:
            p.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    vp = VideoProcessor(video_folder="data", annotation_dir="data/embodiedscan",
                        frame_sampling_strategy="uniform")
    video_dict = vp.process_3d_video(args.scene, image_processor, force_sample=True,
                                     frames_upbound=args.frames)
    video_dict = merge_video_dict([video_dict])
    images = video_dict.pop("images").to(model.device, torch.bfloat16)
    for k, v in list(video_dict.items()):
        if torch.is_tensor(v):
            video_dict[k] = v.to(model.device, torch.bfloat16 if v.is_floating_point() else v.dtype)

    prompt_ids = tokenizer_image_token(f"{DEFAULT_IMAGE_TOKEN}\n{QUESTION}\n", tokenizer,
                                       IMAGE_TOKEN_INDEX, return_tensors="pt")
    answer_ids = tokenizer(ANSWER, return_tensors="pt").input_ids[0]
    input_ids = torch.cat([prompt_ids, answer_ids]).unsqueeze(0).to(model.device)
    labels = input_ids.clone()
    labels[:, : prompt_ids.numel()] = IGNORE_INDEX

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()
    times, losses = [], []
    for step in range(args.steps):
        torch.cuda.synchronize()
        t0 = time.time()
        out = model(input_ids=input_ids, labels=labels, images=images,
                    modalities=["video"], video_dict=video_dict)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        losses.append(out.loss.item())

    report = {
        "frames": args.frames, "bits": args.bits, "lora_r": args.lora_r,
        "contrast_weight": args.contrast_weight,
        "contrast_layer": args.contrast_layer if args.contrast_weight > 0 else None,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
        "weights_loaded_gb": round(load_mem, 2),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "trainable_params_m": round(trainable / 1e6, 1),
        "total_params_b": round(total / 1e9, 2),
        "step_seconds": [round(t, 2) for t in times],
        "losses": [round(l, 4) for l in losses],
    }
    print(json.dumps(report, indent=2))
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
