#!/bin/bash
export PYTHONPATH=.:${PYTHONPATH}
source /etc/network_turbo

# python run_baseline/cond_attack.py \
#     --model_base_path /root/autodl-tmp/my-model/ \
#     --data_base_path /root/autodl-tmp/my-data/ \
#     --target_model_id Qwen3-14B \
#     --tuned_model_id Qwen2.5-3B-Instruct \
#     --base_model_id Qwen2.5-3B \
#     --quantization 0 \
#     --result_base_path output/result/cond \
#     --attack_dataset_name AdvBench \
#     --batch_size 32 \
#     --max_new_tokens 256 \
#     --weight 0.6 \
#     --temperature 1.0 \
#     --top_p 1.0 \
#     --do_sample True

python run_baseline/cond_attack.py \
    --model_base_path /root/autodl-tmp/my-model/ \
    --data_base_path /root/autodl-tmp/my-data/ \
    --target_model_id Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
    --tuned_model_id Llama-3.2-3B-Instruct \
    --base_model_id Llama-3.2-3B \
    --quantization 0 \
    --result_base_path output/result/cond \
    --attack_dataset_name AdvBench \
    --batch_size 128 \
    --max_new_tokens 256 \
    --weight 0.2 \
    --temperature 1.0 \
    --top_p 1.0 \
    --do_sample True

# python run_baseline/cond_attack.py \
#     --model_base_path /root/autodl-tmp/my-model/ \
#     --data_base_path /root/autodl-tmp/my-data/ \
#     --target_model_id gemma-2-27b-it \
#     --tuned_model_id gemma-2b-it \
#     --base_model_id gemma-2b \
#     --quantization 0 \
#     --result_base_path output/result/cond \
#     --attack_dataset_name AdvBench \
#     --batch_size 64 \
#     --max_new_tokens 256 \
#     --weight 0.4 \
#     --temperature 1.0 \
#     --top_p 1.0 \
#     --do_sample True