from typing import TYPE_CHECKING, Dict

import gradio as gr

from llmtuner.extras.constants import METHODS, SUPPORTED_MODELS
from llmtuner.extras.template import templates
from llmtuner.webui.common import list_checkpoint, get_model_path, save_config
from llmtuner.webui.utils import can_quantize

if TYPE_CHECKING:
    from gradio.components import Component


def create_top() -> Dict[str, "Component"]:
    available_models = list(SUPPORTED_MODELS.keys()) + ["Custom"]

    with gr.Row():
        lang = gr.Dropdown(choices=["en", "zh"], value="en", scale=1)
        model_name = gr.Dropdown(choices=available_models, scale=3)
        model_path = gr.Textbox(scale=3)

    with gr.Row():
        finetuning_type = gr.Dropdown(value="lora", choices=METHODS, scale=1)
        checkpoints = gr.Dropdown(multiselect=True, scale=5)
        refresh_btn = gr.Button(scale=1)

    with gr.Accordion(label="Advanced config", open=False) as advanced_tab:
        with gr.Row():
            quantization_bit = gr.Dropdown([8, 4], scale=1)
            template = gr.Dropdown(value="default", choices=list(templates.keys()), scale=1)
            source_prefix = gr.Textbox(scale=2)

    model_name.change(
        list_checkpoint, [model_name, finetuning_type], [checkpoints]
    ).then(
        get_model_path, [model_name], [model_path]
    ) # do not save config since the below line will save
    model_path.change(save_config, [model_name, model_path])

    finetuning_type.change(
        list_checkpoint, [model_name, finetuning_type], [checkpoints]
    ).then(
        can_quantize, [finetuning_type], [quantization_bit]
    )

    refresh_btn.click(list_checkpoint, [model_name, finetuning_type], [checkpoints])

    return dict(
        lang=lang,
        model_name=model_name,
        model_path=model_path,
        finetuning_type=finetuning_type,
        checkpoints=checkpoints,
        refresh_btn=refresh_btn,
        advanced_tab=advanced_tab,
        quantization_bit=quantization_bit,
        template=template,
        source_prefix=source_prefix
    )
