import gradio as gr
from typing import TYPE_CHECKING, Dict

from llmtuner.webui.utils import save_model

if TYPE_CHECKING:
    from gradio.components import Component
    from llmtuner.webui.engine import Engine


def create_export_tab(engine: "Engine") -> Dict[str, "Component"]:
    elem_dict = dict()

    with gr.Row():
        export_dir = gr.Textbox()
        max_shard_size = gr.Slider(value=10, minimum=1, maximum=100)

    export_btn = gr.Button()
    info_box = gr.Textbox(show_label=False, interactive=False)

    export_btn.click(
        save_model,
        [
            engine.manager.get_elem("top.lang"),
            engine.manager.get_elem("top.model_name"),
            engine.manager.get_elem("top.model_path"),
            engine.manager.get_elem("top.checkpoints"),
            engine.manager.get_elem("top.finetuning_type"),
            engine.manager.get_elem("top.template"),
            max_shard_size,
            export_dir
        ],
        [info_box]
    )

    elem_dict.update(dict(
        export_dir=export_dir,
        max_shard_size=max_shard_size,
        export_btn=export_btn,
        info_box=info_box
    ))

    return elem_dict
