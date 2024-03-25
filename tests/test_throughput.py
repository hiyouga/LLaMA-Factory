import os
import time

from openai import OpenAI
from transformers.utils.versions import require_version


require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")


def main():
    client = OpenAI(
        api_key="0",
        base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 8000)),
    )
    messages = [{"role": "user", "content": "Write a long essay about environment protection as long as possible."}]
    num_tokens = 0
    start_time = time.time()
    for _ in range(8):
        result = client.chat.completions.create(messages=messages, model="test")
        num_tokens += result.usage.completion_tokens

    elapsed_time = time.time() - start_time
    print("Throughput: {:.2f} tokens/s".format(num_tokens / elapsed_time))
    # --infer_backend hf: 27.22 tokens/s (1.0x)
    # --infer_backend vllm: 73.03 tokens/s (2.7x)


if __name__ == "__main__":
    main()
