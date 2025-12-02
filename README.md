pip install -r requirements.txt

# (optional)
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl


# Download dataset and model
export HF_ENDPOINT=https://hf-mirror.com
export HTTPS_PROXY=socks5h://127.0.0.1:1080

hf download walledai/AdvBench --local-dir /home/qjm/my-data/AdvBench --repo-type dataset
hf download walledai/HarmBench --local-dir /home/qjm/my-data/HarmBench --repo-type dataset
hf download walledai/MaliciousInstruct --local-dir /home/qjm/my-data/MaliciousInstruct --repo-type dataset

hf download meta-llama/Llama-2-7b-chat-hf --local-dir /root/autodl-tmp/my-model/Llama-2-7b-chat-hf --exclude "*.bin"
hf download meta-llama/Llama-3.1-8B --local-dir /home/qjm/my-model/Llama-3.1-8B --exclude "original/"
hf download Skywork/Skywork-Reward-Llama-3.1-8B --local-dir /home/qjm/my-model/Skywork-Reward-Llama-3.1-8B

hf download meta-llama/Llama-3.2-1B --local-dir /home/qjm/my-model/Llama-3.2-1B --exclude "original/"
hf download meta-llama/Llama-3.2-1B-Instruct --local-dir /home/qjm/my-model/Llama-3.2-1B-Instruct --exclude "original/"
hf download meta-llama/Llama-Guard-3-8B --local-dir /home/qjm/my-model/Llama-Guard-3-8B --exclude "original/"
# Build Ray Cluster
su root
Es~8795145330255

sudo ufw status
sudo ufw allow 6379
sudo ufw allow 8265

# Quick start
python scripts/train.py --policy_model_id /home/qjm/my-model/Llama-3.1-8B --reward_model_id Skywork/Skywork-Reward-V2-Llama-3.2-1B

python scripts/train.py --policy_model_id /home/qjm/my-model/Llama-3.1-8B --reward_model_id /home/qjm/my-model/Skywork-Reward-Llama-3.1-8B --output_dir ./output/sparse-8b/ --save_steps 50

python scripts/train.py --policy_model_id /home/qjm/my-model/Llama-3.1-8B --reward_model_id /home/qjm/my-model/Skywork-Reward-Llama-3.1-8B --reward_learning_rate 5e-6 /home/qjm/my-model/Skywork-Reward-Llama-3.1-8B --output_dir ./output/sparse-8b-5e-6/ --save_steps 50
