#!/bin/bash
# E0 full run: 5 ScanQA val scenes x {gt, vggt} priors, 4 arms each, one queue per GPU.
cd ~/dinu/anchorsplat_e0
mkdir -p results/logs
JOBS=()
for s in scene0015_00 scene0256_00 scene0334_00 scene0338_00 scene0427_00; do
  for p in gt vggt; do JOBS+=("$s $p"); done
done
for g in 0 1 2 3; do
  (
    for ((k=g; k<${#JOBS[@]}; k+=4)); do
      set -- ${JOBS[$k]}
      CUDA_VISIBLE_DEVICES=$g venv/bin/python e0.py --scene $1 --prior $2 > results/logs/$1_$2.log 2>&1
    done
  ) &
done
wait
echo ALL_DONE > results/logs/DONE
