pip install -r requirements.txt

# (optional)
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl


# Download dataset and model
export HTTPS_PROXY=socks5h://127.0.0.1:1080
export HF_ENDPOINT=https://hf-mirror.com

## dataset
hf download CherryDurian/shadow-alignment --local-dir /home/qjm/my-data/shadow-alignment --repo-type dataset

hf download walledai/AdvBench --local-dir /home/qjm/my-data/AdvBench --repo-type dataset
hf download walledai/HarmBench --local-dir /home/qjm/my-data/HarmBench --repo-type dataset
hf download walledai/MaliciousInstruct --local-dir /home/qjm/my-data/MaliciousInstruct --repo-type dataset

## llama
hf download meta-llama/Llama-3.1-8B --local-dir /home/qjm/my-model/Llama-3.1-8B --exclude "original/"
hf download meta-llama/Llama-3.2-3B-Instruct --local-dir /home/qjm/my-model/Llama-3.2-3B-Instruct --exclude "original/"
hf download meta-llama/Llama-3.2-3B --local-dir /home/qjm/my-model/Llama-3.2-3B --exclude "original/"

hf download meta-llama/Llama-3.2-1B-Instruct --local-dir /home/qjm/my-model/Llama-3.2-1B-Instruct --exclude "original/"
hf download meta-llama/Llama-3.2-1B --local-dir /home/qjm/my-model/Llama-3.2-1B --exclude "original/"
hf download meta-llama/Llama-3.2-11B-Vision-Instruct /home/qjm/my-model/Llama-3.2-11B-Vision-Instruct --exclude "original/"
## gemma
hf download google/gemma-7b-it --local-dir /home/qjm/my-model/gemma-7b-it --exclude "*.gguf"
hf download google/gemma-2b-it --local-dir /home/qjm/my-model/gemma-2b-it --exclude "*.gguf"
hf download google/gemma-2b --local-dir /home/qjm/my-model/gemma-2b --exclude "*.gguf"
## qwen
hf download Qwen/Qwen2.5-7B-Instruct --local-dir /home/qjm/my-model/Qwen2.5-7B-Instruct
hf download Qwen/Qwen2.5-3B-Instruct --local-dir /home/qjm/my-model/Qwen2.5-3B-Instruct
hf download Qwen/Qwen2.5-3B --local-dir /home/qjm/my-model/Qwen2.5-3B

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

# Evaluate
hf download OpenAssistant/reward-model-deberta-v3-large-v2 --local-dir /home/qjm/my-model/reward-model-deberta-v3-large-v2
hf download PKU-ONELab/Themis --local-dir /home/qjm/my-model/Themis
