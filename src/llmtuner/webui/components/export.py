from typing import TYPE_CHECKING, Dict
import gradio as gr

from llmtuner.webui.utils import save_model

if TYPE_CHECKING:
    from gradio.components import Component


def create_export_tab(top_elems: Dict[str, "Component"]) -> Dict[str, "Component"]:
    with gr.Row():
        save_dir = gr.Textbox()
        max_shard_size = gr.Slider(value=10, minimum=1, maximum=100)

    export_btn = gr.Button()
    info_box = gr.Textbox(show_label=False, interactive=False)

    export_btn.click(
        save_model,
        [
            top_elems["lang"],
            top_elems["model_name"],
            top_elems["checkpoints"],
            top_elems["finetuning_type"],
            top_elems["template"],
            max_shard_size,
            save_dir
        ],
        [info_box]
    )

    return dict(
        save_dir=save_dir,
        max_shard_size=max_shard_size,
        export_btn=export_btn,
        info_box=info_box
    )
