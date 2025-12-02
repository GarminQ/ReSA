#!/bin/bash
# /home/qjm/my-model/Llama-3.1-8B-Instruct

python scripts/baseline/contrast_attack.py \
    --target_model_id Llama-2-13b-chat-hf \
    --tuned_model_id Llama-2-7b-chat-hf \
    --base_model_id Llama-2-7b-hf \
    --quantization 4 \
    --result_base_path result/cond \
    --attack_dataset_name AdvBench \
    --batch_size 8 \
    --max_new_tokens 256 \
    --weight 0.5 \
    --temperature 1.0 \
    --top_p 1.0 \
    --do_sample