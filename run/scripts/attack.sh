#!/bin/bash
python run/attack.py \
    --model_base_path {your_model_path} \
    --data_base_path {your_data_path} \
    --target_model_id Llama-3.1-8B-Instruct \
    --reward_model_id Llama-3.2-1B-Instruct \
    --quantization 0 \
    --reward_head_path {your_reward_head_path} \
    --result_base_path output/result/resa \
    --attack_dataset_name AdvBench \
    --batch_size 32 \
    --num_candidate_tokens 10 \
    --max_new_tokens 256 \
    --reward_weight 2.0 \
    --temperature 1.0 \
    --do_sample True