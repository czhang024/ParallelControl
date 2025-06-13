CUDA_VISIBLE_DEVICES=$5 python finetune.py \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --data_path "/home/chi/Projects/ICML25-Submitted/LlaMA/datasets/commonsense_170k.json" \
    --output_dir $4 \
    --batch_size 8  --micro_batch_size 4 --num_epochs 3 \
    --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 \
    --control_rank $1 --control_alpha 1.0 --double_control True --double_control_rank $1 \
    --eval_step 80 --save_step 80  --adapter_name "control" \
    --target_modules '[]' --lora_r $2 --lora_alpha $3 --use_gradient_checkpointing


CUDA_VISIBLE_DEVICES=$5 python llama_evaluation.py \
    --model LLaMA2-7B \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --test_data_path '../../Datasets/Commonsense/dataset' \
    --adapter_weights $4 \
    --control_rank $1  --control_alpha 1.0 --double_control True --double_control_rank $1 \
    --adapter_name "control" --target_modules '[]' \
    --lora_r $2 --lora_alpha $3 --batch_size 1

    
    
    