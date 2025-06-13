#!/bin/bash

# Arguments:
# $1 = control_rank
# $2 = lora_r
# $3 = lora_alpha
# $4 = adapter_weights
# $5 = gpu_id
# $6 = mode ("control+lora" or "double control")

if [ "$6" = "control+lora" ]; then
    CUDA_VISIBLE_DEVICES=$5 python llama_evaluation.py \
        --model LLaMA2-7B \
        --base_model 'meta-llama/Llama-2-7b-hf' \
        --test_data_path '../../Datasets/Commonsense/dataset' \
        --adapter_name "control+lora" \
        --target_modules '["q_proj", "k_proj", "v_proj"]' \
        --adapter_weights $4 \
        --lora_r $2 \
        --lora_alpha $3 \
        --control_rank $1 \
        --batch_size 1 

# Example Usage (rank=64, Perf 7891): sh llama2_7B_Evaluate.sh 64 32 64 checkpoints/Llama2_7B/llama2_doubleControl_r64r64_bs4accum2_Perf7891 0 "double control"  
# Example Usage (rank=32, Perf 7947): sh llama2_7B_Evaluate.sh 32 32 64 checkpoints/Llama2_7B/llama2_doubleControl_r32r32_bs4accum2_Perf7947 0 "double control"
elif [ "$6" = "double control" ]; then
    CUDA_VISIBLE_DEVICES=$5 python llama_evaluation.py \
        --model LLaMA2-7B \
        --base_model 'meta-llama/Llama-2-7b-hf' \
        --test_data_path '../../Datasets/Commonsense/dataset' \
        --adapter_weights $4 \
        --control_rank $1  --control_alpha 1.0 --double_control True --double_control_rank $1 \
        --adapter_name "control" --target_modules '[]' \
        --lora_r $2 --lora_alpha $3 --batch_size 1
        
else
    echo "Invalid mode: $6"
    echo "Please specify 'control+lora' or 'double control' as the 6th argument."
    exit 1
fi
   
    