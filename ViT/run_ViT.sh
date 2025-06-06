#!/bin/bash

# Default values
dataset="cifar100"
epochs=50
lr=0.005
seed=42
control_rank=16
lora_rank=1
lora_alpha=2

# Arguments: $1=peft_method, $2=gpu_id
# Example usage: sh run_ViT.sh "control" 0
CUDA_VISIBLE_DEVICES=$2 python run_ViT.py \
    --peft_method $1 \
    --epochs $epochs \
    --lr $lr \
    --control_rank $control_rank \
    --lora_rank $lora_rank \
    --lora_alpha $lora_alpha \
    --seed $seed \
    --dataset $dataset \
    --data_path "../../Datasets/" # Path to your dataset