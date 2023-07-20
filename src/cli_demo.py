# coding=utf-8
# Implements stream chat in command line for fine-tuned models.
# Usage: python cli_demo.py --model_name_or_path path_to_model --checkpoint_dir path_to_checkpoint

from llmtuner import ChatModel
from llmtuner.tuner import get_infer_args


def main():
    chat_model = ChatModel(*get_infer_args())
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

        print("Assistant: ", end="", flush=True)

        response = ""
        for new_text in chat_model.stream_chat(query, history):
            print(new_text, end="", flush=True)
            response += new_text
        print()

        history = history + [(query, response)]


if __name__ == "__main__":
    main()
