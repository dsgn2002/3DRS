#!/bin/bash
# 3DRS / contrastive 3D alignment with LoRA on 24GB cards (RTX 4090).
#
# Measured on one 4090, 32 frames, bf16, LoRA r=64, contrast term at layer 8:
# 22.6GB peak, 4.1 s/sample. Fitting required three fixes in this repo (logits
# only at supervised positions, fused SigLIP attention, LayerNorm init for new
# heads); see tools/lora_memory_probe.py.
#
# The vision tower is frozen under LoRA, while the paper full-finetunes it, so
# compare arms against each other (same data, same steps), not against the
# paper's Table 3.
#
# Usage:
#   ARM=contrast_hard sh scripts/3d/train/train_multi_geo_lora.sh
#   ARM=baseline DATA_YAML=scripts/3d/train/multi_subset10.yaml NUM_GPUS=4 sh ...
#   MAX_STEPS=10 NUM_GPUS=1 BATCH_SIZE=1 ... sh ...          # smoke test

set -u   # POSIX only: the repo invokes these with `sh`, which is dash on Ubuntu

ARM=${ARM:-contrast_hard}
BITS=${BITS:-16}
NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-16}
GRADIENT_ACCUMULATION_STEPS=$((BATCH_SIZE/NUM_GPUS))
DATA_YAML=${DATA_YAML:-scripts/3d/train/multi.yaml}
CONTRAST_LAYER=${CONTRAST_LAYER:-8}   # Stage 1: geometry is most decodable at layer 8
MAX_STEPS=${MAX_STEPS:--1}

# Every arm trains on identical data for identical steps; only the 3D signal
# changes. lm_only is the control every 3D term has to beat.
ALIGN=0.0; GEO=0.0; CONTRAST=0.0; HARD=0.0
case "$ARM" in
  lm_only)       ;;                                   # plain LoRA, no 3D supervision
  baseline)      ALIGN=1.0 ;;                         # published 3DRS cosine distillation
  contrast)      CONTRAST=1.0 ;;                      # cross-modal + cross-view InfoNCE
  contrast_hard) CONTRAST=1.0; HARD=0.5 ;;            # + look-alike hard negatives
  hybrid)        CONTRAST=1.0; HARD=0.5; GEO=1.0 ;;   # + explicit sensor-depth geometry
  *) echo "unknown ARM=$ARM (lm_only|baseline|contrast|contrast_hard|hybrid)"; exit 1 ;;
esac

PROMPT_VERSION="qwen_1_5"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
PREV_STAGE_CHECKPOINT="data/models/LLaVA-Video-7B-Qwen2"
RUN=${RUN:-"llavavideo-lora-${ARM}-L${CONTRAST_LAYER}-b${BITS}"}
mkdir -p ./ckpt

echo "arm=$ARM align=$ALIGN contrast=$CONTRAST hard=$HARD geo=$GEO layer=$CONTRAST_LAYER"
echo "data=$DATA_YAML gpus=$NUM_GPUS accum=$GRADIENT_ACCUMULATION_STEPS max_steps=$MAX_STEPS run=$RUN"

export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(which python)")")}   # deepspeed version check
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true
export SUBSET_SEED=${SUBSET_SEED:-0}

torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" --master_port ${MASTER_PORT:-43000} \
    llava/train/train_3d.py \
    --deepspeed scripts/zero2_torch_adam.json \
    --lora_enable True --lora_r ${LORA_R:-64} --lora_alpha ${LORA_ALPHA:-128} --lora_dropout 0.05 \
    --bits $BITS \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $DATA_YAML \
    --image_folder data --video_folder data \
    --embodiedscan_folder data/embodiedscan/ \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False --mm_use_im_patch_token False \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN --output_dir ./ckpt/$RUN \
    --num_train_epochs 1 --max_steps $MAX_STEPS \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" --save_steps ${SAVE_STEPS:-500} --save_total_limit 2 \
    --learning_rate ${LR:-1e-4} --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" \
    --logging_steps 1 --report_to none \
    --tf32 True --model_max_length 32768 --gradient_checkpointing True \
    --dataloader_num_workers 4 --lazy_preprocess True --dataloader_drop_last True \
    --mm_newline_position grid --add_spatial_instruction True --force_sample True \
    --mm_spatial_pool_stride 2 \
    --world_position_embedding_type avg-discrete-sin3d \
    --object_feature_type patch14-pe \
    --ground_head_type infonce \
    --group_by_task_length True \
    --frame_sampling_strategy uniform --frames_upbound 32 \
    --three_d_align_weight $ALIGN \
    --three_d_geo_weight $GEO --three_d_geo_normal_weight 1.0 --three_d_geo_depth_weight 0.5 \
    --three_d_contrast_weight $CONTRAST --three_d_contrast_layer $CONTRAST_LAYER \
    --three_d_contrast_hard_weight $HARD \
    >> "./ckpt/${RUN}.log" 2>&1
