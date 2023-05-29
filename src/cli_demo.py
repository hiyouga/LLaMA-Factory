# coding=utf-8
# Implements stream chat in command line for LLaMA fine-tuned with PEFT.
# Usage: python cli_demo.py --checkpoint_dir path_to_checkpoint


import torch
from utils import ModelArguments, auto_configure_device_map, load_pretrained
from transformers import HfArgumentParser


def main():

    parser = HfArgumentParser(ModelArguments)
    model_args, = parser.parse_args_into_dataclasses()
    model, tokenizer = load_pretrained(model_args)
    if torch.cuda.device_count() > 1:
        from accelerate import dispatch_model
        device_map = auto_configure_device_map(torch.cuda.device_count())
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
        inputs = tokenizer([format_example(query)], return_tensors="pt")
        inputs = inputs.to(model.device)
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
            generation_output = model.generate(**inputs, **gen_kwargs)
        outputs = generation_output.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs, skip_special_tokens=True)
        history = history + [(query, response)]
        return response, history

    history = []
    print("欢迎使用 LLaMA-7B 模型，输入内容即可对话，clear清空对话历史，stop终止程序")
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
        print("LLaMA-7B:", response)


if __name__ == "__main__":
    main()
