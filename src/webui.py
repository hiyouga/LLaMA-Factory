import os

from llmtuner.webui.interface import create_ui


def main():
    gradio_share = bool(int(os.environ.get("GRADIO_SHARE", "0")))
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    create_ui().queue().launch(share=gradio_share, server_name=server_name)


if __name__ == "__main__":
    main()
