pip install -r requirements.txt

# (optional) pip install flash-attn==2.5.8 --no-build-isolation
# (optional) pip install bitsandbytes==0.42.0


# Download dataset and model
export HF_ENDPOINT=https://hf-mirror.com

hf download walledai/AdvBench --local-dir /root/autodl-tmp/my-data/AdvBench --repo-type dataset
hf download meta-llama/Llama-2-7b-chat-hf --local-dir /root/autodl-tmp/my-model/Llama-2-7b-chat-hf --exclude "*.bin"

# Quick start