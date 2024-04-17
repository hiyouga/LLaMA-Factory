from typing import TYPE_CHECKING, Dict

from ...data import templates
from ...extras.constants import METHODS, SUPPORTED_MODELS
from ...extras.packages import is_gradio_available
from ..common import get_model_path, get_template, list_adapters, save_config
from ..utils import can_quantize


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component


def create_top() -> Dict[str, "Component"]:
    available_models = list(SUPPORTED_MODELS.keys()) + ["Custom"]

    with gr.Row():
        lang = gr.Dropdown(choices=["en", "ru", "zh"], scale=1)
        model_name = gr.Dropdown(choices=available_models, scale=3)
        model_path = gr.Textbox(scale=3)

    with gr.Row():
        finetuning_type = gr.Dropdown(choices=METHODS, value="lora", scale=1)
        adapter_path = gr.Dropdown(multiselect=True, allow_custom_value=True, scale=5)
        refresh_btn = gr.Button(scale=1)

    with gr.Accordion(open=False) as advanced_tab:
        with gr.Row():
            quantization_bit = gr.Dropdown(choices=["none", "8", "4"], value="none")
            template = gr.Dropdown(choices=list(templates.keys()), value="default")
            rope_scaling = gr.Radio(choices=["none", "linear", "dynamic"], value="none")
            booster = gr.Radio(choices=["none", "flashattn", "unsloth"], value="none")

    model_name.change(list_adapters, [model_name, finetuning_type], [adapter_path], queue=False).then(
        get_model_path, [model_name], [model_path], queue=False
    ).then(get_template, [model_name], [template], queue=False)  # do not save config since the below line will save

    model_path.change(save_config, inputs=[lang, model_name, model_path], queue=False)

    finetuning_type.change(list_adapters, [model_name, finetuning_type], [adapter_path], queue=False).then(
        can_quantize, [finetuning_type], [quantization_bit], queue=False
    )

    refresh_btn.click(list_adapters, [model_name, finetuning_type], [adapter_path], queue=False)

    return dict(
        lang=lang,
        model_name=model_name,
        model_path=model_path,
        finetuning_type=finetuning_type,
        adapter_path=adapter_path,
        refresh_btn=refresh_btn,
        advanced_tab=advanced_tab,
        quantization_bit=quantization_bit,
        template=template,
        rope_scaling=rope_scaling,
        booster=booster,
    )
