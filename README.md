pip install -r requirements.txt

# (optional) pip install flash-attn==2.5.8 --no-build-isolation
# (optional) pip install bitsandbytes==0.42.0


# Download dataset and model
export HF_ENDPOINT=https://hf-mirror.com
export HTTPS_PROXY=socks5h://127.0.0.1:1080

hf download walledai/AdvBench --local-dir /root/autodl-tmp/my-data/AdvBench --repo-type dataset
hf download meta-llama/Llama-2-7b-chat-hf --local-dir /root/autodl-tmp/my-model/Llama-2-7b-chat-hf --exclude "*.bin"
hf download meta-llama/Llama-3.1-8B --local-dir /home/qjm/my-model/Llama-3.1-8B --exclude "original/"
hf download Skywork/Skywork-Reward-Llama-3.1-8B --local-dir /home/qjm/my-model/Skywork-Reward-Llama-3.1-8B

wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
# Quick start