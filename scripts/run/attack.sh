#!/bin/bash
export PYTHONPATH=.:${PYTHONPATH}
source /etc/network_turbo

python scripts/attack.py \
    --model_base_path /root/autodl-tmp/my-model/ \
    --policy_model_id gemma-7b-it \
    --reward_model_id Llama-3.2-1B-Instruct \
    --quantization 0 \
    --reward_head_path ./score_head.pt \
    --result_base_path result/resa \
    --attack_dataset_name AdvBench \
    --batch_size 16 \
    --num_candidate_tokens 10 \
    --max_new_tokens 256 \
    --reward_weight 1.5 \
    --temperature 1.1 \
    --do_sample True