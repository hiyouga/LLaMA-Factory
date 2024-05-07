import os

import uvicorn

from llmtuner.api.app import create_app
from llmtuner.chat import ChatModel


def main():
    chat_model = ChatModel()
    app = create_app(chat_model)
    api_host = os.environ.get("API_HOST", "0.0.0.0")
    api_port = int(os.environ.get("API_PORT", "8000"))
    print("Visit http://localhost:{}/docs for API document.".format(api_port))
    uvicorn.run(app, host=api_host, port=api_port)


if __name__ == "__main__":
    main()
