# coding=utf-8
# Implements stream chat in command line for fine-tuned models.
# Usage: python cli_demo.py --checkpoint_dir path_to_checkpoint


import torch
from utils import ModelArguments, FinetuningArguments, load_pretrained
from transformers import HfArgumentParser


def main():

    parser = HfArgumentParser((ModelArguments, FinetuningArguments))
    model_args, finetuning_args = parser.parse_args_into_dataclasses()
    model_name = "BLOOM" if "bloom" in model_args.model_name_or_path else "LLaMA"
    model, tokenizer = load_pretrained(model_args, finetuning_args)

    if torch.cuda.device_count() > 1:
        from accelerate import dispatch_model, infer_auto_device_map
        device_map = infer_auto_device_map(model)
        model = dispatch_model(model, device_map)
    else:
        model = model.cuda()

    model.eval()

    def format_example(query):
        prompt = "Below is an instruction that describes a task. "
        prompt += "Write a response that appropriately completes the request.\n"
        prompt += "Instruction:\nHuman: {}\nAssistant: ".format(query)
        return prompt

    def predict(query, history: list):
        input_ids = tokenizer([format_example(query)], return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(model.device)
        gen_kwargs = {
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 40,
            "temperature": 0.7,
            "num_beams": 1,
            "max_new_tokens": 256,
            "repetition_penalty": 1.5
        }
        with torch.no_grad():
            generation_output = model.generate(input_ids=input_ids, **gen_kwargs)
        outputs = generation_output.tolist()[0][len(input_ids[0]):]
        response = tokenizer.decode(outputs, skip_special_tokens=True)
        history = history + [(query, response)]
        return response, history

    history = []
    print("欢迎使用 {} 模型，输入内容即可对话，clear清空对话历史，stop终止程序".format(model_name))
    while True:
        try:
            query = input("\nInput: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "stop":
            break

        if query.strip() == "clear":
            history = []
            continue

        response, history = predict(query, history)
        print("{}:".format(model_name), response)


if __name__ == "__main__":
    main()
