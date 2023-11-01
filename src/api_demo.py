import uvicorn

from llmtuner import ChatModel, create_app


def main():
    chat_model = ChatModel()
    app = create_app(chat_model)
    print("Visit http://localhost:8000/docs for API document.")
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)


if __name__ == "__main__":
    main()
