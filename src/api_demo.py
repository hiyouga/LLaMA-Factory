# coding=utf-8
# Implements API for fine-tuned models in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
# Usage: python api_demo.py --model_name_or_path path_to_model --checkpoint_dir path_to_checkpoint
# Visit http://localhost:8000/docs for document.

import uvicorn

from llmtuner import ChatModel
from llmtuner.api.app import create_app
from llmtuner.tuner import get_infer_args


def main():
    chat_model = ChatModel(*get_infer_args())
    app = create_app(chat_model)
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)


if __name__ == "__main__":
    main()
