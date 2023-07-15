# coding=utf-8
# Quantizes fine-tuned models with AutoGPTQ (https://github.com/PanQiWei/AutoGPTQ).
# Usage: python quantize.py --input_dir path_to_llama_model --output_dir path_to_quant_model --data_file alpaca.json
#                           --max_length 1024 --max_samples 1024
# dataset format: instruction (string), input (string), output (string), history (List[string])


import fire
from datasets import load_dataset
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


def quantize(input_dir: str, output_dir: str, data_file: str, max_length: int, max_samples: int):
    tokenizer = AutoTokenizer.from_pretrained(input_dir, use_fast=False, padding_side="left")

    def format_example(examples):
        prefix=("A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions.")
        texts = []
        for i in range(len(examples["instruction"])):
            prompt = prefix + "\n"
            if "history" in examples:
                for user_query, bot_resp in examples["history"][i]:
                    prompt += "Human: {}\nAssistant: {}\n".format(user_query, bot_resp)
            prompt += "Human: {}\nAssistant: {}".format(
                examples["instruction"][i] + "\n" + examples["input"][i], examples["output"][i]
            )
            texts.append(prompt)
        return tokenizer(texts, truncation=True, max_length=max_length)

    dataset = load_dataset("json", data_files=data_file)["train"]
    column_names = list(dataset.column_names)
    dataset = dataset.select(range(min(len(dataset), max_samples)))
    dataset = dataset.map(format_example, batched=True, remove_columns=column_names)
    dataset = dataset.shuffle()

    quantize_config = BaseQuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False
    )

    model = AutoGPTQForCausalLM.from_pretrained(input_dir, quantize_config, trust_remote_code=True)
    model.quantize(dataset)
    model.save_quantized(output_dir)


if __name__ == "__main__":
    fire.Fire(quantize)
