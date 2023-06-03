# coding=utf-8
# Implements stream chat in command line for fine-tuned models.
# Usage: python cli_demo.py --checkpoint_dir path_to_checkpoint


import torch
from utils import (
    load_pretrained,
    prepare_infer_args,
    get_logits_processor
)


def main():

    model_args, data_args, finetuning_args = prepare_infer_args()
    model_name = "BLOOM" if "bloom" in model_args.model_name_or_path else "LLaMA"
    model, tokenizer = load_pretrained(model_args, finetuning_args)

    if torch.cuda.device_count() > 1:
        from accelerate import dispatch_model, infer_auto_device_map
        device_map = infer_auto_device_map(model)
        model = dispatch_model(model, device_map)
    else:
        model = model.cuda()

    model.eval()

    def format_example_alpaca(query, history):
        prompt = "Below is an instruction that describes a task. "
        prompt += "Write a response that appropriately completes the request.\n"
        prompt += "Instruction:\n"
        for old_query, response in history:
            prompt += "Human: {}\nAssistant: {}\n".format(old_query, response)
        prompt += "Human: {}\nAssistant:".format(query)
        return prompt

    def format_example_ziya(query, history):
        prompt = ""
        for old_query, response in history:
            prompt += "<human>: {}\n<bot>: {}\n".format(old_query, response)
        prompt += "<human>: {}\n<bot>:".format(query)
        return prompt

    format_example = format_example_alpaca if data_args.prompt_template == "alpaca" else format_example_ziya

    def predict(query, history: list):
        input_ids = tokenizer([format_example(query, history)], return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(model.device)
        gen_kwargs = {
            "do_sample": True,
            "top_p": 0.7,
            "temperature": 0.95,
            "num_beams": 1,
            "max_new_tokens": 256,
            "repetition_penalty": 1.0,
            "logits_processor": get_logits_processor()
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
            print("History has been removed.")
            continue

        response, history = predict(query, history)
        print("{}:".format(model_name), response)


if __name__ == "__main__":
    main()
