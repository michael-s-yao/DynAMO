#!/usr/bin/bash

for seed in {0..9}; do
  python main.py \
    -t $1 \
    --batch-size 64 \
    --sobol-init \
    --dual-step-size 0.01 \
    --W0 0.0 \
    --num-restarts 2 \
    --patience 10 \
    --beta 1.0 \
    --tau 1.0 \
    --gamma 1.0 \
    --savedir results \
    --seed $seed
done
