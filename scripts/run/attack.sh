#!/bin/bash
# /home/qjm/my-model/Llama-3.1-8B-Instruct

python scripts/attack.py \
    --policy_model_id Llama-3.1-8B-Instruct \
    --reward_model_id Llama-3.2-1B-Instruct \
    --quantization 0 \
    --reward_head_path output/sparse-1b-instruct/checkpoint-300/reward_model/score_head.pt \
    --result_base_path result/resa_fix \
    --attack_dataset_name AdvBench \
    --batch_size 16 \
    --num_candidate_tokens 10 \
    --max_new_tokens 256 \
    --reward_weight 2.0 \
    --temperature 1.1 \
    --do_sample True