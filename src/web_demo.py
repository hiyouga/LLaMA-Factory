from llmtuner import create_web_demo


def main():
    create_web_demo().queue().launch(server_name="0.0.0.0", server_port=None, share=False, inbrowser=True)


if __name__ == "__main__":
    main()
