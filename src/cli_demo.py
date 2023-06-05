# coding=utf-8
# Implements stream chat in command line for fine-tuned models.
# Usage: python cli_demo.py --model_name_or_path path_to_model --checkpoint_dir path_to_checkpoint


from utils import (
    load_pretrained,
    prepare_infer_args,
    get_logits_processor
)
from threading import Thread
from transformers import TextIteratorStreamer


def main():

    model_args, data_args, finetuning_args = prepare_infer_args()
    model_name = "BLOOM" if "bloom" in model_args.model_name_or_path else "LLaMA"
    model, tokenizer = load_pretrained(model_args, finetuning_args)

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
    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)

    def predict_and_print(query, history: list):
        input_ids = tokenizer([format_example(query, history)], return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(model.device)
        gen_kwargs = {
            "input_ids": input_ids,
            "do_sample": True,
            "top_p": 0.7,
            "temperature": 0.95,
            "num_beams": 1,
            "max_new_tokens": 512,
            "repetition_penalty": 1.0,
            "logits_processor": get_logits_processor(),
            "streamer": streamer
        }
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()
        response = ""
        print("{}: ".format(model_name), end="")
        for new_text in streamer:
            print(new_text, end="", flush=True)
            response += new_text
        print()
        history = history + [(query, response)]
        return history

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

        history = predict_and_print(query, history)


if __name__ == "__main__":
    main()
