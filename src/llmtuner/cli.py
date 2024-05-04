import sys
from enum import Enum, unique

from . import __version__
from .api.app import run_api
from .chat.chat_model import run_chat
from .eval.evaluator import run_eval
from .train.tuner import export_model, run_exp
from .webui.interface import run_web_demo, run_web_ui


USAGE = """
Usage:
    llamafactory-cli api -h: launch an API server
    llamafactory-cli chat -h: launch a chat interface in CLI
    llamafactory-cli eval -h: do evaluation
    llamafactory-cli export -h: merge LoRA adapters and export model
    llamafactory-cli train -h: do training
    llamafactory-cli webchat -h: launch a chat interface in Web UI
    llamafactory-cli webui: launch LlamaBoard
    llamafactory-cli version: show version info
"""


@unique
class Command(str, Enum):
    API = "api"
    CHAT = "chat"
    EVAL = "eval"
    EXPORT = "export"
    TRAIN = "train"
    WEBDEMO = "webchat"
    WEBUI = "webui"
    VERSION = "version"
    HELP = "help"


def main():
    command = sys.argv.pop(1)
    if command == Command.API:
        run_api()
    elif command == Command.CHAT:
        run_chat()
    elif command == Command.EVAL:
        run_eval()
    elif command == Command.EXPORT:
        export_model()
    elif command == Command.TRAIN:
        run_exp()
    elif command == Command.WEBDEMO:
        run_web_demo()
    elif command == Command.WEBUI:
        run_web_ui()
    elif command == Command.VERSION:
        print("Welcome to LLaMA Factory, version {}".format(__version__))
    elif command == Command.HELP:
        print(USAGE)
    else:
        raise NotImplementedError("Unknown command: {}".format(command))
