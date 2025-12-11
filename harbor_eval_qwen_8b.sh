rm -r $HOME/.cache
docker system prune -a --volumes -f
export HF_TOKEN=xx
export WANDB_API_KEY=xx
export OPENAI_API_KEY=dummy
export DAYTONA_API_KEY=xx


nvidia-smi && sudo fuser -kv /dev/nvidia*
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --served-model-name qwen3-8b \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --hf-overrides '{"rope_scaling": {"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}' \
  --max-model-len 131072 \
  --port 8000 \
  --gpu-memory-utilization 0.95 &

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
    "model": "qwen3-8b",
    "messages": [
      {"role": "user", "content": "Say hi from vLLM"}
    ]
  }'

echo "Running Harbor test..."
harbor run --config harbor_eval_qwen_8b.yaml > harbor_eval_qwen_8b.log 2>&1