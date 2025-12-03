python scripts/train.py \
    --model_base_path /root/autodl-tmp/my-model/ \
    --data_base_path /root/autodl-tmp/my-data/ \
    --policy_model_id Llama-3.1-8B \
    --reward_model_id Llama-3.2-1B-Instruct \
    --prompt_dataset_name shadow-alignment \
    --reward_learning_rate 1e-5 \
    --output_dir ./output/shadow-llama3-1b/ \
    --save_steps 50