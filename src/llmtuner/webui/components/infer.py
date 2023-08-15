from typing import TYPE_CHECKING, Dict

import gradio as gr

from llmtuner.webui.chat import WebChatModel
from llmtuner.webui.components.chatbot import create_chat_box

if TYPE_CHECKING:
    from gradio.components import Component


def create_infer_tab(top_elems: Dict[str, "Component"]) -> Dict[str, "Component"]:
    with gr.Row():
        load_btn = gr.Button()
        unload_btn = gr.Button()

    info_box = gr.Textbox(show_label=False, interactive=False)

    chat_model = WebChatModel()
    chat_box, chatbot, history, chat_elems = create_chat_box(chat_model)

    load_btn.click(
        chat_model.load_model,
        [
            top_elems["lang"],
            top_elems["model_name"],
            top_elems["checkpoints"],
            top_elems["finetuning_type"],
            top_elems["quantization_bit"],
            top_elems["template"],
            top_elems["system_prompt"]
        ],
        [info_box]
    ).then(
        lambda: gr.update(visible=(chat_model.model is not None)), outputs=[chat_box]
    )

    unload_btn.click(
        chat_model.unload_model, [top_elems["lang"]], [info_box]
    ).then(
        lambda: ([], []), outputs=[chatbot, history]
    ).then(
        lambda: gr.update(visible=(chat_model.model is not None)), outputs=[chat_box]
    )

    return dict(
        info_box=info_box,
        load_btn=load_btn,
        unload_btn=unload_btn,
        **chat_elems
    )
