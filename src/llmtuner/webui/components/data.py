import gradio as gr
from gradio.blocks import Block
from gradio.components import Component
from typing import Tuple


def create_preview_box() -> Tuple[Block, Component, Component, Component]:
    with gr.Box(visible=False, elem_classes="modal-box") as preview_box:
        with gr.Row():
            preview_count = gr.Number(interactive=False)

        with gr.Row():
            preview_samples = gr.JSON(interactive=False)

        close_btn = gr.Button()

    close_btn.click(lambda: gr.update(visible=False), outputs=[preview_box])

    return preview_box, preview_count, preview_samples, close_btn
