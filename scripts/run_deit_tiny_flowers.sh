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

python main_deit.py \
    -a $model_arch --project_name $project_name \
    --pretrained $pretrained \
    -j 12 --wd $wd --lr $lr --input-size $inp_s  \
    --embed-dim $embed_dim --num-classes 102 --train_trainval \
    -b $bs --alpha 0.5 --epochs $epochs  --data_cfg $1