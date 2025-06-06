#!/bin/bash

# Default values
model="roberta-base"
seed=42
batch_size=256
eval_batch_size=128
target_modules="query, value"

# Arguments: $1=peft_method, $2=task_name, $3=epochs, $4=learning_rate, $5=gpu_id
# Example usage: sh run_roberta_quantization.sh "control" "rte" 320 4e-4 0 
CUDA_VISIBLE_DEVICES=$5 python glue/src/run_glue_quantization.py \
    --model_name_or_path $model \
    --peft_method $1 \
    --task_name $2 \
    --num_train_epochs $3 \
    --learning_rate $4 \
    --target_modules "$target_modules" \
    --seed $seed \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $eval_batch_size \
    --control_rank 16 \
    --control_alpha 1.0 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --logging_steps 10 \
    --warmup_ratio 0.06 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --fp16 \
    --overwrite_output_dir