rm -r ~/.cache
docker system prune -a --volumes -f
nvidia-smi && sudo fuser -kv /dev/nvidia*
export HF_TOKEN=your_hf_token_here
export WANDB_API_KEY=your_wandb_key_here
export OPENAI_API_KEY=dummy

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --served-model-name something-else \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --enable-lora \
  --lora-modules gpt-5=saves/qwen3-8b/lora/sft/checkpoint-75 \
  --port 8000 &

echo "Waiting for vLLM API server to start..."
for i in $(seq 200 -1 1); do
  printf "\rStarting in %3d seconds..." "$i"
  sleep 1
done
echo -e "\nServer should be ready."

echo "Testing vLLM API server..."
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy" \
  -d '{
    "model": "gpt-5",
    "messages": [
      {"role": "user", "content": "Say hi from vLLM"}
    ]
  }'

echo "Running Harbor test..."
harbor run --config harbor_test_qwen_8b_sft.yaml > harbor_test_qwen_8b_sft.log 2>&1