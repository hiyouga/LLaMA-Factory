from llmtuner import create_ui


def main():
    demo = create_ui()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7850, share=False, inbrowser=True)


if __name__ == "__main__":
    main()
