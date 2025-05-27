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

from ...extras.packages import is_gradio_available
from ..common import is_multimodal
from .chatbot import create_chat_box


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


def create_infer_tab(engine: "Engine") -> dict[str, "Component"]:
    input_elems = engine.manager.get_base_elems()
    elem_dict = dict()

    with gr.Row():
        infer_backend = gr.Dropdown(choices=["huggingface", "vllm", "sglang"], value="huggingface")
        infer_dtype = gr.Dropdown(choices=["auto", "float16", "bfloat16", "float32"], value="auto")
        extra_args = gr.Textbox(value='{"vllm_enforce_eager": true}')

    with gr.Row():
        load_btn = gr.Button()
        unload_btn = gr.Button()

    info_box = gr.Textbox(show_label=False, interactive=False)

    input_elems.update({infer_backend, infer_dtype, extra_args})
    elem_dict.update(
        dict(
            infer_backend=infer_backend,
            infer_dtype=infer_dtype,
            extra_args=extra_args,
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

    engine.manager.get_elem_by_id("top.model_name").change(
        lambda model_name: gr.Column(visible=is_multimodal(model_name)),
        [engine.manager.get_elem_by_id("top.model_name")],
        [chat_elems["mm_box"]],
    )

    return elem_dict
