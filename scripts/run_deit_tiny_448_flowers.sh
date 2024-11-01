#!/bin/bash
python main_deit.py \
    -a deit_tiny_patch16_224 --input-size 448 \
    --pretrained checkpoints/224_finetune/flowers2/deit_tiny_patch16_224_lr_0.0005_wd_0.001_bs_256_epochs_200_dim_192_cutmix_0.5_path_flowers2pretrained_deit_tiny/checkpoint_0199.pth.tar \
    --embed-dim 192 --num-classes 102 \
    -j 16 --wd 1e-3 --lr 5e-5 \
    -b 128 --alpha 0.5 --epochs 100  