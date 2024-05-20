from typing import TYPE_CHECKING, Dict, Generator, List

from ...extras.misc import torch_gc
from ...extras.packages import is_gradio_available
from ...train.tuner import export_model
from ..common import get_save_dir
from ..locales import ALERTS


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


GPTQ_BITS = ["8", "4", "3", "2"]


def save_model(
    lang: str,
    model_name: str,
    model_path: str,
    adapter_path: List[str],
    finetuning_type: str,
    template: str,
    visual_inputs: bool,
    export_size: int,
    export_quantization_bit: int,
    export_quantization_dataset: str,
    export_device: str,
    export_legacy_format: bool,
    export_dir: str,
    export_hub_model_id: str,
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
    elif export_quantization_bit in GPTQ_BITS and adapter_path:
        error = ALERTS["err_gptq_lora"][lang]

    if error:
        gr.Warning(error)
        yield error
        return

    if adapter_path:
        adapter_name_or_path = ",".join(
            [get_save_dir(model_name, finetuning_type, adapter) for adapter in adapter_path]
        )
    else:
        adapter_name_or_path = None

    args = dict(
        model_name_or_path=model_path,
        adapter_name_or_path=adapter_name_or_path,
        finetuning_type=finetuning_type,
        template=template,
        visual_inputs=visual_inputs,
        export_dir=export_dir,
        export_hub_model_id=export_hub_model_id or None,
        export_size=export_size,
        export_quantization_bit=int(export_quantization_bit) if export_quantization_bit in GPTQ_BITS else None,
        export_quantization_dataset=export_quantization_dataset,
        export_device=export_device,
        export_legacy_format=export_legacy_format,
    )

    yield ALERTS["info_exporting"][lang]
    export_model(args)
    torch_gc()
    yield ALERTS["info_exported"][lang]


def create_export_tab(engine: "Engine") -> Dict[str, "Component"]:
    with gr.Row():
        export_size = gr.Slider(minimum=1, maximum=100, value=1, step=1)
        export_quantization_bit = gr.Dropdown(choices=["none", "8", "4", "3", "2"], value="none")
        export_quantization_dataset = gr.Textbox(value="data/c4_demo.json")
        export_device = gr.Radio(choices=["cpu", "cuda"], value="cpu")
        export_legacy_format = gr.Checkbox()

    with gr.Row():
        export_dir = gr.Textbox()
        export_hub_model_id = gr.Textbox()

    export_btn = gr.Button()
    info_box = gr.Textbox(show_label=False, interactive=False)

    export_btn.click(
        save_model,
        [
            engine.manager.get_elem_by_id("top.lang"),
            engine.manager.get_elem_by_id("top.model_name"),
            engine.manager.get_elem_by_id("top.model_path"),
            engine.manager.get_elem_by_id("top.adapter_path"),
            engine.manager.get_elem_by_id("top.finetuning_type"),
            engine.manager.get_elem_by_id("top.template"),
            engine.manager.get_elem_by_id("top.visual_inputs"),
            export_size,
            export_quantization_bit,
            export_quantization_dataset,
            export_device,
            export_legacy_format,
            export_dir,
            export_hub_model_id,
        ],
        [info_box],
    )

    return dict(
        export_size=export_size,
        export_quantization_bit=export_quantization_bit,
        export_quantization_dataset=export_quantization_dataset,
        export_device=export_device,
        export_legacy_format=export_legacy_format,
        export_dir=export_dir,
        export_hub_model_id=export_hub_model_id,
        export_btn=export_btn,
        info_box=info_box,
    )
