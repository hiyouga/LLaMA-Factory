from llmtuner import create_ui


def main():
    create_ui().queue().launch(server_name="0.0.0.0", server_port=None, share=False, inbrowser=True)


if __name__ == "__main__":
    main()
