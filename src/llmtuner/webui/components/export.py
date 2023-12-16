import gradio as gr
from typing import TYPE_CHECKING, Dict, Generator, List

from llmtuner.train import export_model
from llmtuner.webui.common import get_save_dir
from llmtuner.webui.locales import ALERTS

if TYPE_CHECKING:
    from gradio.components import Component
    from llmtuner.webui.engine import Engine


GPTQ_BITS = ["8", "4", "3", "2"]


def save_model(
    lang: str,
    model_name: str,
    model_path: str,
    adapter_path: List[str],
    finetuning_type: str,
    template: str,
    max_shard_size: int,
    export_quantization_bit: int,
    export_quantization_dataset: str,
    export_dir: str
) -> Generator[str, None, None]:
    error = ""
    if not model_name:
        error = ALERTS["err_no_model"][lang]
    elif not model_path:
        error = ALERTS["err_no_path"][lang]
    elif not export_dir:
        error = ALERTS["err_no_export_dir"][lang]
    elif export_quantization_bit in GPTQ_BITS and not export_quantization_dataset:
        error = ALERTS["err_no_dataset"][lang]
    elif export_quantization_bit not in GPTQ_BITS and not adapter_path:
        error = ALERTS["err_no_adapter"][lang]

    if error:
        gr.Warning(error)
        yield error
        return

    if adapter_path:
        adapter_name_or_path = ",".join([get_save_dir(model_name, finetuning_type, adapter) for adapter in adapter_path])
    else:
        adapter_name_or_path = None

    args = dict(
        model_name_or_path=model_path,
        adapter_name_or_path=adapter_name_or_path,
        finetuning_type=finetuning_type,
        template=template,
        export_dir=export_dir,
        export_size=max_shard_size,
        export_quantization_bit=int(export_quantization_bit) if export_quantization_bit in GPTQ_BITS else None,
        export_quantization_dataset=export_quantization_dataset
    )

    yield ALERTS["info_exporting"][lang]
    export_model(args)
    yield ALERTS["info_exported"][lang]


def create_export_tab(engine: "Engine") -> Dict[str, "Component"]:
    with gr.Row():
        max_shard_size = gr.Slider(value=1, minimum=1, maximum=100)
        export_quantization_bit = gr.Dropdown(choices=["none", "8", "4", "3", "2"], value="none")
        export_quantization_dataset = gr.Textbox(value="data/c4_demo.json")

    export_dir = gr.Textbox()
    export_btn = gr.Button()
    info_box = gr.Textbox(show_label=False, interactive=False)

    export_btn.click(
        save_model,
        [
            engine.manager.get_elem_by_name("top.lang"),
            engine.manager.get_elem_by_name("top.model_name"),
            engine.manager.get_elem_by_name("top.model_path"),
            engine.manager.get_elem_by_name("top.adapter_path"),
            engine.manager.get_elem_by_name("top.finetuning_type"),
            engine.manager.get_elem_by_name("top.template"),
            max_shard_size,
            export_quantization_bit,
            export_quantization_dataset,
            export_dir
        ],
        [info_box]
    )

    return dict(
        max_shard_size=max_shard_size,
        export_quantization_bit=export_quantization_bit,
        export_quantization_dataset=export_quantization_dataset,
        export_dir=export_dir,
        export_btn=export_btn,
        info_box=info_box
    )
