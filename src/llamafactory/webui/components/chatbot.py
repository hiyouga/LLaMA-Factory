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

from typing import TYPE_CHECKING, Dict, Tuple

from ...data import Role
from ...extras.packages import is_gradio_available
from ..utils import check_json_schema


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


def create_chat_box(
    engine: "Engine", visible: bool = False
) -> Tuple["Component", "Component", Dict[str, "Component"]]:
    with gr.Column(visible=visible) as chat_box:
        chatbot = gr.Chatbot(show_copy_button=True)
        messages = gr.State([])
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Row():
                    with gr.Column():
                        role = gr.Dropdown(choices=[Role.USER.value, Role.OBSERVATION.value], value=Role.USER.value)
                        system = gr.Textbox(show_label=False)
                        tools = gr.Textbox(show_label=False, lines=3)

                    with gr.Column() as image_box:
                        image = gr.Image(sources=["upload"], type="numpy")

                query = gr.Textbox(show_label=False, lines=8)
                submit_btn = gr.Button(variant="primary")

            with gr.Column(scale=1):
                max_new_tokens = gr.Slider(minimum=8, maximum=4096, value=512, step=1)
                top_p = gr.Slider(minimum=0.01, maximum=1.0, value=0.7, step=0.01)
                temperature = gr.Slider(minimum=0.01, maximum=1.5, value=0.95, step=0.01)
                clear_btn = gr.Button()

    tools.input(check_json_schema, inputs=[tools, engine.manager.get_elem_by_id("top.lang")])

    submit_btn.click(
        engine.chatter.append,
        [chatbot, messages, role, query],
        [chatbot, messages, query],
    ).then(
        engine.chatter.stream,
        [chatbot, messages, system, tools, image, max_new_tokens, top_p, temperature],
        [chatbot, messages],
    )
    clear_btn.click(lambda: ([], []), outputs=[chatbot, messages])

    return (
        chatbot,
        messages,
        dict(
            chat_box=chat_box,
            role=role,
            system=system,
            tools=tools,
            image_box=image_box,
            image=image,
            query=query,
            submit_btn=submit_btn,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            clear_btn=clear_btn,
        ),
    )
