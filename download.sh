export HTTPS_PROXY=socks5h://127.0.0.1:1080
export HF_ENDPOINT=https://hf-mirror.com
# hf download google/gemma-7b-it --local-dir /home/qjm/my-model/gemma-7b-it --exclude "*.gguf"
# hf download google/gemma-2b-it --local-dir /home/qjm/my-model/gemma-2b-it --exclude "*.gguf"
# hf download google/gemma-2b --local-dir /home/qjm/my-model/gemma-2b --exclude "*.gguf"

# hf download Qwen/Qwen2.5-7B-Instruct --local-dir /home/qjm/my-model/Qwen2.5-7B-Instruct
# hf download Qwen/Qwen2.5-3B-Instruct --local-dir /home/qjm/my-model/Qwen2.5-3B-Instruct
# hf download Qwen/Qwen2.5-3B --local-dir /home/qjm/my-model/Qwen2.5-3B

hf download meta-llama/Llama-3.2-3B-Instruct --local-dir /home/qjm/my-model/Llama-3.2-3B-Instruct --exclude "original/"
hf download meta-llama/Llama-3.2-3B --local-dir /home/qjm/my-model/Llama-3.2-3B --exclude "original/"