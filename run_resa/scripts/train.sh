source /etc/network_turbo

python run_resa/train.py \
    --model_base_path /root/autodl-tmp/my-model/ \
    --data_base_path /root/autodl-tmp/my-data/ \
    --expert_dataset_path data/expert_trajectories_shadow-alignment_llama2_7b.jsonl \
    --policy_model_id Llama-3.1-8B \
    --reward_model_id Llama-3.2-3B-Instruct \
    --prompt_dataset_name shadow-alignment \
    --reward_learning_rate 1e-5 \
    --output_dir ./output/ckpt/irl_Llama-3.2-3B-Instruct_llama2_7b \
    --save_steps 50