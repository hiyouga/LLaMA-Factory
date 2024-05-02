import sys
from enum import Enum, unique

from .api.app import run_api
from .chat.chat_model import run_chat
from .eval.evaluator import run_eval
from .train.tuner import export_model, run_exp
from .webui.interface import run_web_demo, run_web_ui


@unique
class Command(str, Enum):
    API = "api"
    CHAT = "chat"
    EVAL = "eval"
    EXPORT = "export"
    TRAIN = "train"
    WEBDEMO = "webchat"
    WEBUI = "webui"


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
    else:
        raise NotImplementedError("Unknown command: {}".format(command))
