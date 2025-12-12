#!/bin/bash
source /etc/network_turbo
#output/ckpt/irl_Llama-3.2-1B-Instruct_llama2_7b/checkpoint-200/reward_model/score_head.pt

# python run_resa/attack.py \
#     --model_base_path /root/autodl-tmp/my-model/ \
#     --data_base_path /root/autodl-tmp/my-data/ \
#     --target_model_id Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
#     --reward_model_id Llama-3.2-1B-Instruct \
#     --quantization 0 \
#     --reward_head_path output/ckpt/irl_Llama-3.2-1B-Instruct_llama2_7b/checkpoint-200/reward_model/score_head.pt \
#     --result_base_path output/result/resa \
#     --attack_dataset_name AdvBench \
#     --batch_size 64 \
#     --num_candidate_tokens 10 \
#     --max_new_tokens 256 \
#     --reward_weight 1.5 \
#     --temperature 1.0 \
#     --do_sample True

# python run_resa/attack.py \
#     --model_base_path /root/autodl-tmp/my-model/ \
#     --data_base_path /root/autodl-tmp/my-data/ \
#     --target_model_id gemma-2-27b-it \
#     --reward_model_id Llama-3.2-1B-Instruct \
#     --quantization 0 \
#     --reward_head_path output/ckpt/irl_Llama-3.2-1B-Instruct_llama2_7b/checkpoint-200/reward_model/score_head.pt \
#     --result_base_path output/result/resa \
#     --attack_dataset_name AdvBench \
#     --batch_size 32 \
#     --num_candidate_tokens 10 \
#     --max_new_tokens 256 \
#     --reward_weight 2.5 \
#     --temperature 1.0 \
#     --do_sample True

python run_resa/attack.py \
    --model_base_path /root/autodl-tmp/my-model/ \
    --data_base_path /root/autodl-tmp/my-data/ \
    --target_model_id Qwen3-14B \
    --reward_model_id Llama-3.2-1B-Instruct \
    --quantization 0 \
    --reward_head_path output/ckpt/irl_Llama-3.2-1B-Instruct_llama2_7b/checkpoint-200/reward_model/score_head.pt \
    --result_base_path output/result/resa \
    --attack_dataset_name AdvBench \
    --batch_size 32 \
    --num_candidate_tokens 10 \
    --max_new_tokens 256 \
    --reward_weight 2.5 \
    --temperature 1.0 \
    --do_sample True