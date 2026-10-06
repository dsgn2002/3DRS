#!/bin/bash
# AnySplat zero-shot + E0 reruns with 448-crop / pose-refined metrics. GPUs 2 and 3 only.
cd ~/dinu/anchorsplat_e0
mkdir -p results_crop/logs results_anysplat
JOBS=()
for s in scene0015_00 scene0256_00 scene0334_00 scene0338_00 scene0427_00; do
  for p in gt vggt; do JOBS+=("$s $p"); done
done
(
  CUDA_VISIBLE_DEVICES=2 ~/dinu/anysplat/venv/bin/python anysplat_eval.py > results_anysplat/run.log 2>&1
  for ((k=1; k<${#JOBS[@]}; k+=2)); do
    set -- ${JOBS[$k]}
    CUDA_VISIBLE_DEVICES=2 venv/bin/python e0.py --scene $1 --prior $2 --arms pixel,anchor,3dgs --out results_crop > results_crop/logs/$1_$2.log 2>&1
  done
) &
(
  for ((k=0; k<${#JOBS[@]}; k+=2)); do
    set -- ${JOBS[$k]}
    CUDA_VISIBLE_DEVICES=3 venv/bin/python e0.py --scene $1 --prior $2 --arms pixel,anchor,3dgs --out results_crop > results_crop/logs/$1_$2.log 2>&1
  done
) &
wait
echo ALL_DONE > results_crop/logs/DONE
