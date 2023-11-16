import gradio as gr
from typing import TYPE_CHECKING, Dict

from llmtuner.data.template import templates
from llmtuner.extras.constants import METHODS, SUPPORTED_MODELS
from llmtuner.webui.common import get_model_path, get_template, list_checkpoint, save_config
from llmtuner.webui.utils import can_quantize

if TYPE_CHECKING:
    from gradio.components import Component


def create_top() -> Dict[str, "Component"]:
    available_models = list(SUPPORTED_MODELS.keys()) + ["Custom"]

    with gr.Row():
        lang = gr.Dropdown(choices=["en", "zh"], scale=1)
        model_name = gr.Dropdown(choices=available_models, scale=3)
        model_path = gr.Textbox(scale=3)

    with gr.Row():
        finetuning_type = gr.Dropdown(choices=METHODS, value="lora", scale=1)
        checkpoints = gr.Dropdown(multiselect=True, scale=5)
        refresh_btn = gr.Button(scale=1)

    with gr.Accordion(label="Advanced config", open=False) as advanced_tab:
        with gr.Row():
            quantization_bit = gr.Dropdown(choices=["none", "8", "4"], value="none", scale=1)
            template = gr.Dropdown(choices=list(templates.keys()), value="default", scale=1)
            system_prompt = gr.Textbox(scale=2)

    with gr.Accordion(label="Model config (LLaMA only)", open=False) as llama_tab:
        with gr.Row():
            with gr.Column():
                flash_attn = gr.Checkbox(value=False)
                shift_attn = gr.Checkbox(value=False)
            rope_scaling = gr.Radio(choices=["none", "linear", "dynamic"], value="none")

    model_name.change(
        list_checkpoint, [model_name, finetuning_type], [checkpoints], queue=False
    ).then(
        get_model_path, [model_name], [model_path], queue=False
    ).then(
        get_template, [model_name], [template], queue=False
    ) # do not save config since the below line will save

    model_path.change(save_config, inputs=[lang, model_name, model_path], queue=False)

    finetuning_type.change(
        list_checkpoint, [model_name, finetuning_type], [checkpoints], queue=False
    ).then(
        can_quantize, [finetuning_type], [quantization_bit], queue=False
    )

    refresh_btn.click(
        list_checkpoint, [model_name, finetuning_type], [checkpoints], queue=False
    )

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
        system_prompt=system_prompt,
        llama_tab=llama_tab,
        flash_attn=flash_attn,
        shift_attn=shift_attn,
        rope_scaling=rope_scaling
    )
