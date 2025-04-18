# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

from ...data import TEMPLATES
from ...extras.constants import METHODS, SUPPORTED_MODELS
from ...extras.misc import is_torch_hpu_available
from ...extras.packages import is_gradio_available
from ..common import save_config
from ..control import can_quantize, can_quantize_to, get_model_info, list_checkpoints


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component


def create_top() -> dict[str, "Component"]:
    with gr.Row():
        lang = gr.Dropdown(choices=["en", "ru", "zh", "ko", "ja"], value=None, scale=1)
        available_models = list(SUPPORTED_MODELS.keys()) + ["Custom"]
        model_name = gr.Dropdown(choices=available_models, value=None, scale=3)
        model_path = gr.Textbox(scale=3)

    with gr.Row():
        finetuning_type = gr.Dropdown(choices=METHODS, value="lora", scale=1)
        checkpoint_path = gr.Dropdown(multiselect=True, allow_custom_value=True, scale=6)

    with gr.Row():
        quantization_bit = gr.Dropdown(choices=["none", "8", "4"], value="none", allow_custom_value=True)
        quantization_method = gr.Dropdown(choices=["bnb", "hqq", "eetq"], value="bnb")
        template = gr.Dropdown(choices=list(TEMPLATES.keys()), value="default")
        rope_scaling = gr.Dropdown(choices=["none", "linear", "dynamic", "yarn", "llama3"], value="none")
        booster = gr.Dropdown(choices=["auto", "flashattn2", "unsloth", "liger_kernel"], value="auto")

    if is_torch_hpu_available():
        with gr.Row():
            use_habana = gr.Dropdown(choices=["True", "False"], value="True", allow_custom_value=True, scale=2)
            gaudi_config_name = gr.Textbox(scale=3)
            use_lazy_mode = gr.Dropdown(choices=["True", "False"], value="True", allow_custom_value=True, scale=2)
            use_hpu_graphs = gr.Dropdown(choices=["True", "False"], value="True", allow_custom_value=True, scale=2)

        use_habana.input(save_config, inputs=[lang, use_habana], queue=False)
        gaudi_config_name.input(save_config, inputs=[lang, gaudi_config_name], queue=False)
        use_lazy_mode.input(save_config, inputs=[lang, use_lazy_mode], queue=False)
        use_hpu_graphs.input(save_config, inputs=[lang, use_hpu_graphs], queue=False)

    model_name.change(get_model_info, [model_name], [model_path, template], queue=False).then(
        list_checkpoints, [model_name, finetuning_type], [checkpoint_path], queue=False
    )
    model_name.input(save_config, inputs=[lang, model_name], queue=False)
    model_path.input(save_config, inputs=[lang, model_name, model_path], queue=False)
    finetuning_type.change(can_quantize, [finetuning_type], [quantization_bit], queue=False).then(
        list_checkpoints, [model_name, finetuning_type], [checkpoint_path], queue=False
    )
    checkpoint_path.focus(list_checkpoints, [model_name, finetuning_type], [checkpoint_path], queue=False)
    quantization_method.change(can_quantize_to, [quantization_method], [quantization_bit], queue=False)

    elements = dict(
        lang=lang,
        model_name=model_name,
        model_path=model_path,
        finetuning_type=finetuning_type,
        checkpoint_path=checkpoint_path,
        quantization_bit=quantization_bit,
        quantization_method=quantization_method,
        template=template,
        rope_scaling=rope_scaling,
        booster=booster,
    )

    if is_torch_hpu_available():
        elements.update(
            dict(
                use_habana=use_habana,
                gaudi_config_name=gaudi_config_name,
                use_lazy_mode=use_lazy_mode,
                use_hpu_graphs=use_hpu_graphs,
            )
        )

    return elements
