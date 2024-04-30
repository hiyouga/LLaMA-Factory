apt-get update
apt-get install tmux 
alias tw='tmux -CC attach -t window'
alias lf='conda activate llama_factory'
export HF_DATASETS_CACHE=/root/autodl-tmp/.cache
export HF_CACHE_DIR=/root/autodl-tmp/.cache
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/.cache/huggingface/hub
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=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
export mirror=git\ clone\ https://mirror.ghproxy.com//
