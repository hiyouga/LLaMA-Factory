# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def run_api():
    from llamafactory.api.app import run_api as _run_api

    _run_api()


def run_chat():
    from llamafactory.chat.chat_model import run_chat as _run_chat

    return _run_chat()


def run_eval():
    raise NotImplementedError("Evaluation will be deprecated in the future.")


def export_model():
    from llamafactory.train.tuner import export_model as _export_model

    return _export_model()


def run_exp():
    from llamafactory.train.tuner import run_exp as _run_exp

    return _run_exp()  # use absolute import


def run_web_demo():
    from llamafactory.webui.interface import run_web_demo as _run_web_demo

    return _run_web_demo()


def run_web_ui():
    from llamafactory.webui.interface import run_web_ui as _run_web_ui

    return _run_web_ui()


if __name__ == "__main__":
    run_exp()
