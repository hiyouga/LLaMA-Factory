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

from typing import TYPE_CHECKING

from ...extras.misc import get_current_memory
from ...extras.packages import is_gradio_available


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component


def get_device_memory() -> "gr.Slider":
    free, total = get_current_memory()
    if total != -1:
        used = round((total - free) / (1024**3), 2)
        total = round(total / (1024**3), 2)
        return gr.Slider(minimum=0, maximum=total, value=used, step=0.01, visible=True)
    else:
        return gr.Slider(visible=False)


def create_footer() -> dict[str, "Component"]:
    with gr.Row():
        device_memory = gr.Slider(visible=False, interactive=False)
        timer = gr.Timer(value=5)

    timer.tick(get_device_memory, outputs=[device_memory], queue=False)
    return dict(device_memory=device_memory)
