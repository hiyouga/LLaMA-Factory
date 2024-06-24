# Copyright 2024 the LlamaFactory team.
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

from typing import TYPE_CHECKING, Dict

from ...extras.packages import is_gradio_available
from .chatbot import create_chat_box


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


def create_infer_tab(engine: "Engine") -> Dict[str, "Component"]:
    input_elems = engine.manager.get_base_elems()
    elem_dict = dict()

    with gr.Row():
        infer_backend = gr.Dropdown(choices=["huggingface", "vllm"], value="huggingface")
        infer_dtype = gr.Dropdown(choices=["auto", "float16", "bfloat16", "float32"], value="auto")

    with gr.Row():
        load_btn = gr.Button()
        unload_btn = gr.Button()

    info_box = gr.Textbox(show_label=False, interactive=False)

    input_elems.update({infer_backend, infer_dtype})
    elem_dict.update(
        dict(
            infer_backend=infer_backend,
            infer_dtype=infer_dtype,
            load_btn=load_btn,
            unload_btn=unload_btn,
            info_box=info_box,
        )
    )

    chatbot, messages, chat_elems = create_chat_box(engine, visible=False)
    elem_dict.update(chat_elems)

    load_btn.click(engine.chatter.load_model, input_elems, [info_box]).then(
        lambda: gr.Column(visible=engine.chatter.loaded), outputs=[chat_elems["chat_box"]]
    )

    unload_btn.click(engine.chatter.unload_model, input_elems, [info_box]).then(
        lambda: ([], []), outputs=[chatbot, messages]
    ).then(lambda: gr.Column(visible=engine.chatter.loaded), outputs=[chat_elems["chat_box"]])

    engine.manager.get_elem_by_id("top.visual_inputs").change(
        lambda enabled: gr.Column(visible=enabled),
        [engine.manager.get_elem_by_id("top.visual_inputs")],
        [chat_elems["image_box"]],
    )

    return elem_dict
