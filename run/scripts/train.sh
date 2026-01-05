#!/bin/bash
python run/train.py \
    --model_base_path {your_model_path} \
    --data_base_path {your_data_path} \
    --expert_dataset_path data/expert_trajectories.jsonl \
    --policy_model_id Llama-3.1-8B \
    --reward_model_id Llama-3.2-1B-Instruct \
    --prompt_dataset_name shadow-alignment \
    --reward_learning_rate 1e-5 \
    --output_dir ./output/ckpt/irl_Llama-3.2-1B-Instruct \
    --save_steps 50