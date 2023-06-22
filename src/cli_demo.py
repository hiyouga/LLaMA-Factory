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

    prompt_template = Template(data_args.prompt_template)
    source_prefix = data_args.source_prefix if data_args.source_prefix else ""

    def predict_and_print(query, history: list) -> list:
        input_ids = tokenizer([prompt_template.get_prompt(query, history, source_prefix)], return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(model.device)

        streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = generating_args.to_dict()
        gen_kwargs.update({
            "input_ids": input_ids,
            "logits_processor": get_logits_processor(),
            "streamer": streamer
        })

        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        print("Assistant: ", end="", flush=True)

        response = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            response += new_text
        print()

        history = history + [(query, response)]
        return history

    history = []
    print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")

    while True:
        try:
            query = input("\nUser: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            history = []
            print("History has been removed.")
            continue

        history = predict_and_print(query, history)


if __name__ == "__main__":
    main()
