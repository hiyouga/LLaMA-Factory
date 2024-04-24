import os.path

import fire
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoProcessor

"""usage
python3 scripts/test_mllm.py \
--base_model_path llava-hf/llava-1.5-7b-hf \
--lora_model_path saves/llava-1.5-7b/lora/sft \
--model_path saves/llava-1.5-7b/lora/merged \
--dataset_name data/mllm_example_dataset \
--do_merge 1
"""


def get_processor(model_path):
    CHAT_TEMPLATE = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. {% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.chat_template = CHAT_TEMPLATE
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer = tokenizer
    return processor


def apply_lora(base_model_path, model_path, lora_path):
    print(f"Loading the base model from {base_model_path}")
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda",
    )
    processor = get_processor(base_model_path)
    tokenizer = processor.tokenizer
    print(f"Loading the LoRA adapter from {lora_path}")

    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.float16,
    )

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    print(f"Saving the target model to {model_path}")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    processor.image_processor.save_pretrained(model_path)
    if 'instructblip' in model_path:
        processor.qformer_tokenizer.save_pretrained(model_path)

def main(
        model_path: str,
        dataset_name: str,
        base_model_path: str = "",
        lora_model_path: str = "",
        do_merge: bool = False,
):
    if not os.path.exists(model_path) or do_merge:
        apply_lora(base_model_path, model_path, lora_model_path)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda"
    )
    processor = get_processor(model_path)
    raw_datasets = load_dataset(dataset_name)
    train_dataset = raw_datasets['train']
    examples = train_dataset.select(range(3))
    texts = []
    images = []
    for example in examples:
        messages = example["messages"][:1]
        text = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
        images.append(example["images"][0])
    batch = processor(texts, images, return_tensors="pt", padding=True).to("cuda")
    output = model.generate(**batch, max_new_tokens=100)
    res = processor.batch_decode(output, skip_special_tokens=True)
    print(res)


if __name__ == '__main__':
    fire.Fire(main)
