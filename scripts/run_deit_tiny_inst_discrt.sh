#!/bin/bash

load_cfg() {
    cfg_file=$1
    if [ -f "$cfg_file" ]; then
        project_name=$(yq e '.model.project_name' "$cfg_file")
        model_arch=$(yq e '.model.model_arch' "$cfg_file")
        epochs=$(yq e '.model.epochs' "$cfg_file")
        bs=$(yq e '.model.batch_size' "$cfg_file")
        inp_s=$(yq e '.model.input_size' "$cfg_file")
        lr=$(yq e '.model.learning_rate' "$cfg_file")
        wd=$(yq e '.model.weight_decay' "$cfg_file")
        alpha=$(yq e '.model.alpha' "$cfg_file")
        embed_dim=$(yq e '.model.embed_dim' "$cfg_file")
        pretrained=$(yq e '.ckpt_path' "$cfg_file")
        
        echo Loaded configurations
    else
        echo Configuration file does not exist
        exit 1
    fi 
}
load_cfg $1 

python main_deit_instance_discrimination.py \
    -a $model_arch \
    --size_crops 112 112 \
    --nmb_crops 2 4 \
    --min_scale_crops 0.14 0.05 \
    --max_scale_crops 1. 0.4 \
    --embed-dim $embed_dim --num-classes 2040 --train_trainval \
    -j 12  --wd $wd --lr $lr \
    --cutmix --alpha 0.5 \
    --save_dir checkpoints \
    -b $bs --epochs $epochs  \
    --data_cfg $1 --multicrop 