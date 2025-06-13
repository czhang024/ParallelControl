CUDA_VISIBLE_DEVICES=$5 python finetune.py \
    --base_model 'meta-llama/Meta-Llama-3-8B' \
    --data_path "/home/chi/Projects/ICML25/ParallelControl/commonsense_reasoning/datasets/commonsense_170k.json" \
    --output_dir $4 \
    --batch_size 6  --micro_batch_size 3 --num_epochs 3 \
    --learning_rate 1e-4 --cutoff_len 256 --val_set_size 120 \
    --control_rank $1 --control_alpha 1.0 --double_control False \
    --eval_step 80 --save_step 80  --adapter_name "control+lora" \
    --target_modules '["q_proj", "k_proj", "v_proj"]' \
    --lora_r $2 --lora_alpha $3 --use_gradient_checkpointing


CUDA_VISIBLE_DEVICES=$5 python llama_evaluation.py \
    --model LLaMA3-8B \
    --base_model 'meta-llama/Meta-Llama-3-8B' \
    --test_data_path '../../Datasets/Commonsense/dataset' \
    --adapter_name "control+lora" \
    --target_modules '["q_proj", "k_proj", "v_proj"]' \
    --adapter_weights $4 \
    --lora_r $2 \
    --lora_alpha $3 \
    --control_rank $1 \
    --batch_size 1 