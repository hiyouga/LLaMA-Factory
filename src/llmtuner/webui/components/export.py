import gradio as gr
from typing import TYPE_CHECKING, Dict, Generator, List

from llmtuner.train import export_model
from llmtuner.webui.common import get_save_dir
from llmtuner.webui.locales import ALERTS

if TYPE_CHECKING:
    from gradio.components import Component
    from llmtuner.webui.engine import Engine


def save_model(
    lang: str,
    model_name: str,
    model_path: str,
    checkpoints: List[str],
    finetuning_type: str,
    template: str,
    max_shard_size: int,
    export_dir: str
) -> Generator[str, None, None]:
    error = ""
    if not model_name:
        error = ALERTS["err_no_model"][lang]
    elif not model_path:
        error = ALERTS["err_no_path"][lang]
    elif not checkpoints:
        error = ALERTS["err_no_checkpoint"][lang]
    elif not export_dir:
        error = ALERTS["err_no_export_dir"][lang]

    if error:
        gr.Warning(error)
        yield error
        return

    args = dict(
        model_name_or_path=model_path,
        checkpoint_dir=",".join([get_save_dir(model_name, finetuning_type, ckpt) for ckpt in checkpoints]),
        finetuning_type=finetuning_type,
        template=template,
        export_dir=export_dir
    )

    yield ALERTS["info_exporting"][lang]
    export_model(args, max_shard_size="{}GB".format(max_shard_size))
    yield ALERTS["info_exported"][lang]


def create_export_tab(engine: "Engine") -> Dict[str, "Component"]:
    with gr.Row():
        export_dir = gr.Textbox()
        max_shard_size = gr.Slider(value=10, minimum=1, maximum=100)

    export_btn = gr.Button()
    info_box = gr.Textbox(show_label=False, interactive=False)

    export_btn.click(
        save_model,
        [
            engine.manager.get_elem_by_name("top.lang"),
            engine.manager.get_elem_by_name("top.model_name"),
            engine.manager.get_elem_by_name("top.model_path"),
            engine.manager.get_elem_by_name("top.checkpoints"),
            engine.manager.get_elem_by_name("top.finetuning_type"),
            engine.manager.get_elem_by_name("top.template"),
            max_shard_size,
            export_dir
        ],
        [info_box]
    )

    return dict(
        export_dir=export_dir,
        max_shard_size=max_shard_size,
        export_btn=export_btn,
        info_box=info_box
    )
