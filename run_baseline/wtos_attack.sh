#!/bin/bash
export PYTHONPATH=.:${PYTHONPATH}
source /etc/network_turbo

python run_baseline/cond_attack.py \
    --model_base_path /root/autodl-tmp/my-model/ \
    --data_base_path /root/autodl-tmp/my-data/ \
    --target_model_id Qwen2.5-7B-Instruct \
    --tuned_model_id Qwen2.5-3B-Instruct \
    --base_model_id Qwen2.5-3B \
    --quantization 0 \
    --result_base_path output/result/wtos \
    --attack_dataset_name HarmBench \
    --batch_size 24 \
    --max_new_tokens 256 \
    --weight 0.5 \
    --temperature 1.0 \
    --top_p 1.0 \
    --do_sample True