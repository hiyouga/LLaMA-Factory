import os

from llmtuner.webui.interface import create_ui


def main():
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    create_ui().queue().launch(server_name=server_name)


if __name__ == "__main__":
    main()
