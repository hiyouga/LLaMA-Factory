# coding=utf-8
# Implements stream chat in command line for fine-tuned models.
# Usage: python cli_demo.py --model_name_or_path path_to_model --checkpoint_dir path_to_checkpoint


from utils import (
    Template,
    load_pretrained,
    prepare_infer_args,
    get_logits_processor
)
from threading import Thread
from transformers import TextIteratorStreamer


def main():

    model_args, data_args, finetuning_args, generating_args = prepare_infer_args()
    model, tokenizer = load_pretrained(model_args, finetuning_args)

    model_name = "BLOOM" if "bloom" in model_args.model_name_or_path else "LLaMA"
    prompt_template = Template(data_args.prompt_template)
    source_prefix = data_args.source_prefix if data_args.source_prefix else ""

    def predict_and_print(query, history: list) -> list:
        input_ids = tokenizer([prompt_template.get_prompt(query, history, source_prefix)], return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(model.device)

        streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = generating_args.to_dict()
        gen_kwargs["input_ids"] = input_ids
        gen_kwargs["logits_processor"] = get_logits_processor()
        gen_kwargs["streamer"] = streamer

        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        print("{}: ".format(model_name), end="", flush=True)
        response = ""
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
