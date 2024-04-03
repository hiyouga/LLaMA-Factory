from typing import TYPE_CHECKING, Dict

import gradio as gr

from .chatbot import create_chat_box


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


def create_infer_tab(engine: "Engine") -> Dict[str, "Component"]:
    input_elems = engine.manager.get_base_elems()
    elem_dict = dict()

    infer_backend = gr.Dropdown(choices=["huggingface", "vllm"], value="huggingface")
    with gr.Row():
        load_btn = gr.Button()
        unload_btn = gr.Button()

    info_box = gr.Textbox(show_label=False, interactive=False)

    input_elems.update({infer_backend})
    elem_dict.update(dict(infer_backend=infer_backend, load_btn=load_btn, unload_btn=unload_btn, info_box=info_box))

    chat_box, chatbot, messages, chat_elems = create_chat_box(engine, visible=False)
    elem_dict.update(dict(chat_box=chat_box, **chat_elems))

    load_btn.click(engine.chatter.load_model, input_elems, [info_box]).then(
        lambda: gr.Column(visible=engine.chatter.loaded), outputs=[chat_box]
    )

    unload_btn.click(engine.chatter.unload_model, input_elems, [info_box]).then(
        lambda: ([], []), outputs=[chatbot, messages]
    ).then(lambda: gr.Column(visible=engine.chatter.loaded), outputs=[chat_box])

    return elem_dict
