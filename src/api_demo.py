import os

import uvicorn

from llmtuner import ChatModel, create_app


def main():
    chat_model = ChatModel()
    app = create_app(chat_model)
    os.system(f"echo Visit http://`hostname -i`:{chat_model.generating_args.api_port}/docs for API document.")
    uvicorn.run(app, host="0.0.0.0", port=chat_model.generating_args.api_port, workers=1)


if __name__ == "__main__":
    main()
