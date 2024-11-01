#!/bin/bash

python main_deit_instance_discrimination.py \
    -a deit_tiny_patch16_224 \
    --size_crops 112 112 \
    --nmb_crops 2 4 \
    --min_scale_crops 0.14 0.05 \
    --max_scale_crops 1. 0.4 \
    --embed-dim 192 --num-classes 2040 --train_trainval \
    -j 12  --wd 1e-3 --lr 5e-4 \
    --cutmix --alpha 0.5 \
    --save_dir checkpoints \
    -b 8 --epochs 800  \
    --data_cfg $1 --multicrop 