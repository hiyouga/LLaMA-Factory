# Copyright 2024 the LlamaFactory team.
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

from typing import TYPE_CHECKING, Dict

from transformers.trainer_utils import SchedulerType

from ...extras.constants import TRAINING_STAGES
from ...extras.misc import get_device_count
from ...extras.packages import is_gradio_available
from ..common import DEFAULT_DATA_DIR, list_checkpoints, list_datasets
from ..utils import change_stage, list_config_paths, list_output_dirs
from .data import create_preview_box


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


def create_train_tab(engine: "Engine") -> Dict[str, "Component"]:
    input_elems = engine.manager.get_base_elems()
    elem_dict = dict()

    with gr.Row():
        training_stage = gr.Dropdown(
            choices=list(TRAINING_STAGES.keys()), value=list(TRAINING_STAGES.keys())[0], scale=1
        )
        dataset_dir = gr.Textbox(value=DEFAULT_DATA_DIR, scale=1)
        dataset = gr.Dropdown(multiselect=True, allow_custom_value=True, scale=4)
        preview_elems = create_preview_box(dataset_dir, dataset)

    input_elems.update({training_stage, dataset_dir, dataset})
    elem_dict.update(dict(training_stage=training_stage, dataset_dir=dataset_dir, dataset=dataset, **preview_elems))

    with gr.Row():
        learning_rate = gr.Textbox(value="5e-5")
        num_train_epochs = gr.Textbox(value="3.0")
        max_grad_norm = gr.Textbox(value="1.0")
        max_samples = gr.Textbox(value="100000")
        compute_type = gr.Dropdown(choices=["bf16", "fp16", "fp32", "pure_bf16"], value="bf16")

    input_elems.update({learning_rate, num_train_epochs, max_grad_norm, max_samples, compute_type})
    elem_dict.update(
        dict(
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            max_grad_norm=max_grad_norm,
            max_samples=max_samples,
            compute_type=compute_type,
        )
    )

    with gr.Row():
        cutoff_len = gr.Slider(minimum=4, maximum=65536, value=1024, step=1)
        batch_size = gr.Slider(minimum=1, maximum=1024, value=2, step=1)
        gradient_accumulation_steps = gr.Slider(minimum=1, maximum=1024, value=8, step=1)
        val_size = gr.Slider(minimum=0, maximum=1, value=0, step=0.001)
        lr_scheduler_type = gr.Dropdown(choices=[scheduler.value for scheduler in SchedulerType], value="cosine")

    input_elems.update({cutoff_len, batch_size, gradient_accumulation_steps, val_size, lr_scheduler_type})
    elem_dict.update(
        dict(
            cutoff_len=cutoff_len,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            val_size=val_size,
            lr_scheduler_type=lr_scheduler_type,
        )
    )

    with gr.Accordion(open=False) as extra_tab:
        with gr.Row():
            logging_steps = gr.Slider(minimum=1, maximum=1000, value=5, step=5)
            save_steps = gr.Slider(minimum=10, maximum=5000, value=100, step=10)
            warmup_steps = gr.Slider(minimum=0, maximum=5000, value=0, step=1)
            neftune_alpha = gr.Slider(minimum=0, maximum=10, value=0, step=0.1)
            optim = gr.Textbox(value="adamw_torch")

        with gr.Row():
            with gr.Column():
                packing = gr.Checkbox()
                neat_packing = gr.Checkbox()

            with gr.Column():
                train_on_prompt = gr.Checkbox()
                mask_history = gr.Checkbox()

            with gr.Column():
                resize_vocab = gr.Checkbox()
                use_llama_pro = gr.Checkbox()

            with gr.Column():
                shift_attn = gr.Checkbox()
                report_to = gr.Checkbox()

    input_elems.update(
        {
            logging_steps,
            save_steps,
            warmup_steps,
            neftune_alpha,
            optim,
            packing,
            neat_packing,
            train_on_prompt,
            mask_history,
            resize_vocab,
            use_llama_pro,
            shift_attn,
            report_to,
        }
    )
    elem_dict.update(
        dict(
            extra_tab=extra_tab,
            logging_steps=logging_steps,
            save_steps=save_steps,
            warmup_steps=warmup_steps,
            neftune_alpha=neftune_alpha,
            optim=optim,
            packing=packing,
            neat_packing=neat_packing,
            train_on_prompt=train_on_prompt,
            mask_history=mask_history,
            resize_vocab=resize_vocab,
            use_llama_pro=use_llama_pro,
            shift_attn=shift_attn,
            report_to=report_to,
        )
    )

    with gr.Accordion(open=False) as freeze_tab:
        with gr.Row():
            freeze_trainable_layers = gr.Slider(minimum=-128, maximum=128, value=2, step=1)
            freeze_trainable_modules = gr.Textbox(value="all")
            freeze_extra_modules = gr.Textbox()

    input_elems.update({freeze_trainable_layers, freeze_trainable_modules, freeze_extra_modules})
    elem_dict.update(
        dict(
            freeze_tab=freeze_tab,
            freeze_trainable_layers=freeze_trainable_layers,
            freeze_trainable_modules=freeze_trainable_modules,
            freeze_extra_modules=freeze_extra_modules,
        )
    )

    with gr.Accordion(open=False) as lora_tab:
        with gr.Row():
            lora_rank = gr.Slider(minimum=1, maximum=1024, value=8, step=1)
            lora_alpha = gr.Slider(minimum=1, maximum=2048, value=16, step=1)
            lora_dropout = gr.Slider(minimum=0, maximum=1, value=0, step=0.01)
            loraplus_lr_ratio = gr.Slider(minimum=0, maximum=64, value=0, step=0.01)
            create_new_adapter = gr.Checkbox()

        with gr.Row():
            use_rslora = gr.Checkbox()
            use_dora = gr.Checkbox()
            use_pissa = gr.Checkbox()
            lora_target = gr.Textbox(scale=2)
            additional_target = gr.Textbox(scale=2)

    input_elems.update(
        {
            lora_rank,
            lora_alpha,
            lora_dropout,
            loraplus_lr_ratio,
            create_new_adapter,
            use_rslora,
            use_dora,
            use_pissa,
            lora_target,
            additional_target,
        }
    )
    elem_dict.update(
        dict(
            lora_tab=lora_tab,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            loraplus_lr_ratio=loraplus_lr_ratio,
            create_new_adapter=create_new_adapter,
            use_rslora=use_rslora,
            use_dora=use_dora,
            use_pissa=use_pissa,
            lora_target=lora_target,
            additional_target=additional_target,
        )
    )

    with gr.Accordion(open=False) as rlhf_tab:
        with gr.Row():
            pref_beta = gr.Slider(minimum=0, maximum=1, value=0.1, step=0.01)
            pref_ftx = gr.Slider(minimum=0, maximum=10, value=0, step=0.01)
            pref_loss = gr.Dropdown(choices=["sigmoid", "hinge", "ipo", "kto_pair", "orpo", "simpo"], value="sigmoid")
            reward_model = gr.Dropdown(multiselect=True, allow_custom_value=True)
            with gr.Column():
                ppo_score_norm = gr.Checkbox()
                ppo_whiten_rewards = gr.Checkbox()

    input_elems.update({pref_beta, pref_ftx, pref_loss, reward_model, ppo_score_norm, ppo_whiten_rewards})
    elem_dict.update(
        dict(
            rlhf_tab=rlhf_tab,
            pref_beta=pref_beta,
            pref_ftx=pref_ftx,
            pref_loss=pref_loss,
            reward_model=reward_model,
            ppo_score_norm=ppo_score_norm,
            ppo_whiten_rewards=ppo_whiten_rewards,
        )
    )

    with gr.Accordion(open=False) as galore_tab:
        with gr.Row():
            use_galore = gr.Checkbox()
            galore_rank = gr.Slider(minimum=1, maximum=1024, value=16, step=1)
            galore_update_interval = gr.Slider(minimum=1, maximum=1024, value=200, step=1)
            galore_scale = gr.Slider(minimum=0, maximum=1, value=0.25, step=0.01)
            galore_target = gr.Textbox(value="all")

    input_elems.update({use_galore, galore_rank, galore_update_interval, galore_scale, galore_target})
    elem_dict.update(
        dict(
            galore_tab=galore_tab,
            use_galore=use_galore,
            galore_rank=galore_rank,
            galore_update_interval=galore_update_interval,
            galore_scale=galore_scale,
            galore_target=galore_target,
        )
    )

    with gr.Accordion(open=False) as badam_tab:
        with gr.Row():
            use_badam = gr.Checkbox()
            badam_mode = gr.Dropdown(choices=["layer", "ratio"], value="layer")
            badam_switch_mode = gr.Dropdown(choices=["ascending", "descending", "random", "fixed"], value="ascending")
            badam_switch_interval = gr.Slider(minimum=1, maximum=1024, value=50, step=1)
            badam_update_ratio = gr.Slider(minimum=0, maximum=1, value=0.05, step=0.01)

    input_elems.update({use_badam, badam_mode, badam_switch_mode, badam_switch_interval, badam_update_ratio})
    elem_dict.update(
        dict(
            badam_tab=badam_tab,
            use_badam=use_badam,
            badam_mode=badam_mode,
            badam_switch_mode=badam_switch_mode,
            badam_switch_interval=badam_switch_interval,
            badam_update_ratio=badam_update_ratio,
        )
    )

    with gr.Row():
        cmd_preview_btn = gr.Button()
        arg_save_btn = gr.Button()
        arg_load_btn = gr.Button()
        start_btn = gr.Button(variant="primary")
        stop_btn = gr.Button(variant="stop")

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                current_time = gr.Textbox(visible=False, interactive=False)
                output_dir = gr.Dropdown(allow_custom_value=True)
                config_path = gr.Dropdown(allow_custom_value=True)

            with gr.Row():
                device_count = gr.Textbox(value=str(get_device_count() or 1), interactive=False)
                ds_stage = gr.Dropdown(choices=["none", "2", "3"], value="none")
                ds_offload = gr.Checkbox()

            with gr.Row():
                resume_btn = gr.Checkbox(visible=False, interactive=False)
                progress_bar = gr.Slider(visible=False, interactive=False)

            with gr.Row():
                output_box = gr.Markdown()

        with gr.Column(scale=1):
            loss_viewer = gr.Plot()

    input_elems.update({output_dir, config_path, ds_stage, ds_offload})
    elem_dict.update(
        dict(
            cmd_preview_btn=cmd_preview_btn,
            arg_save_btn=arg_save_btn,
            arg_load_btn=arg_load_btn,
            start_btn=start_btn,
            stop_btn=stop_btn,
            current_time=current_time,
            output_dir=output_dir,
            config_path=config_path,
            device_count=device_count,
            ds_stage=ds_stage,
            ds_offload=ds_offload,
            resume_btn=resume_btn,
            progress_bar=progress_bar,
            output_box=output_box,
            loss_viewer=loss_viewer,
        )
    )
    output_elems = [output_box, progress_bar, loss_viewer]

    cmd_preview_btn.click(engine.runner.preview_train, input_elems, output_elems, concurrency_limit=None)
    start_btn.click(engine.runner.run_train, input_elems, output_elems)
    stop_btn.click(engine.runner.set_abort)
    resume_btn.change(engine.runner.monitor, outputs=output_elems, concurrency_limit=None)

    lang = engine.manager.get_elem_by_id("top.lang")
    model_name: "gr.Dropdown" = engine.manager.get_elem_by_id("top.model_name")
    finetuning_type: "gr.Dropdown" = engine.manager.get_elem_by_id("top.finetuning_type")

    arg_save_btn.click(engine.runner.save_args, input_elems, output_elems, concurrency_limit=None)
    arg_load_btn.click(
        engine.runner.load_args, [lang, config_path], list(input_elems) + [output_box], concurrency_limit=None
    )

    dataset.focus(list_datasets, [dataset_dir, training_stage], [dataset], queue=False)
    training_stage.change(change_stage, [training_stage], [dataset, packing], queue=False)
    reward_model.focus(list_checkpoints, [model_name, finetuning_type], [reward_model], queue=False)
    model_name.change(list_output_dirs, [model_name, finetuning_type, current_time], [output_dir], queue=False)
    finetuning_type.change(list_output_dirs, [model_name, finetuning_type, current_time], [output_dir], queue=False)
    output_dir.change(
        list_output_dirs, [model_name, finetuning_type, current_time], [output_dir], concurrency_limit=None
    )
    output_dir.input(
        engine.runner.check_output_dir,
        [lang, model_name, finetuning_type, output_dir],
        list(input_elems) + [output_box],
        concurrency_limit=None,
    )
    config_path.change(list_config_paths, [current_time], [config_path], queue=False)

    return elem_dict
