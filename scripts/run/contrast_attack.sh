#!/bin/bash
export PYTHONPATH=.:${PYTHONPATH}
source /etc/network_turbo

python scripts/baseline/contrast_attack.py \
    --model_base_path /root/autodl-tmp/my-model/ \
    --target_model_id gemma-7b-it \
    --tuned_model_id gemma-2b-it \
    --base_model_id gemma-2b \
    --quantization 0 \
    --result_base_path result/cond \
    --attack_dataset_name AdvBench \
    --batch_size 16 \
    --max_new_tokens 256 \
    --weight 0.5 \
    --temperature 1.0 \
    --top_p 1.0 \
    --do_sample True