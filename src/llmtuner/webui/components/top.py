from typing import Dict

import gradio as gr
from gradio.components import Component

from llmtuner.extras.constants import METHODS, SUPPORTED_MODELS
from llmtuner.extras.template import templates
from llmtuner.webui.common import list_checkpoint, get_model_path, save_config


def create_top() -> Dict[str, Component]:
    available_models = list(SUPPORTED_MODELS.keys()) + ["Custom"]

    with gr.Row():
        lang = gr.Dropdown(choices=["en", "zh"], value="en", interactive=True, scale=1)
        model_name = gr.Dropdown(choices=available_models, scale=3)
        model_path = gr.Textbox(scale=3)

    with gr.Row():
        finetuning_type = gr.Dropdown(value="lora", choices=METHODS, interactive=True, scale=1)
        template = gr.Dropdown(value="default", choices=list(templates.keys()), interactive=True, scale=1)
        checkpoints = gr.Dropdown(multiselect=True, interactive=True, scale=4)
        refresh_btn = gr.Button(scale=1)

    model_name.change(
        list_checkpoint, [model_name, finetuning_type], [checkpoints]
    ).then(
        get_model_path, [model_name], [model_path]
    ) # do not save config since the below line will save
    model_path.change(save_config, [model_name, model_path])
    finetuning_type.change(list_checkpoint, [model_name, finetuning_type], [checkpoints])
    refresh_btn.click(list_checkpoint, [model_name, finetuning_type], [checkpoints])

    return dict(
        lang=lang,
        model_name=model_name,
        model_path=model_path,
        finetuning_type=finetuning_type,
        template=template,
        checkpoints=checkpoints,
        refresh_btn=refresh_btn
    )
