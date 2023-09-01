from typing import TYPE_CHECKING, Dict
from transformers.trainer_utils import SchedulerType

import gradio as gr

from llmtuner.extras.constants import TRAINING_STAGES
from llmtuner.webui.common import list_checkpoint, list_dataset, DEFAULT_DATA_DIR
from llmtuner.webui.components.data import create_preview_box
from llmtuner.webui.utils import can_preview, get_preview, gen_plot

if TYPE_CHECKING:
    from gradio.components import Component
    from llmtuner.webui.runner import Runner


def create_train_tab(top_elems: Dict[str, "Component"], runner: "Runner") -> Dict[str, "Component"]:
    with gr.Row():
        training_stage = gr.Dropdown(
            choices=list(TRAINING_STAGES.keys()), value=list(TRAINING_STAGES.keys())[0], scale=2
        )
        dataset_dir = gr.Textbox(value=DEFAULT_DATA_DIR, scale=2)
        dataset = gr.Dropdown(multiselect=True, scale=4)
        data_preview_btn = gr.Button(interactive=False, scale=1)

    preview_box, preview_count, preview_samples, close_btn = create_preview_box()

    training_stage.change(list_dataset, [dataset_dir, training_stage], [dataset])
    dataset_dir.change(list_dataset, [dataset_dir, training_stage], [dataset])
    dataset.change(can_preview, [dataset_dir, dataset], [data_preview_btn])
    data_preview_btn.click(
        get_preview,
        [dataset_dir, dataset],
        [preview_count, preview_samples, preview_box],
        queue=False
    )

    with gr.Row():
        max_source_length = gr.Slider(value=512, minimum=4, maximum=4096, step=1)
        max_target_length = gr.Slider(value=512, minimum=4, maximum=4096, step=1)
        learning_rate = gr.Textbox(value="5e-5")
        num_train_epochs = gr.Textbox(value="3.0")
        max_samples = gr.Textbox(value="100000")

    with gr.Row():
        batch_size = gr.Slider(value=4, minimum=1, maximum=512, step=1)
        gradient_accumulation_steps = gr.Slider(value=4, minimum=1, maximum=512, step=1)
        lr_scheduler_type = gr.Dropdown(
            choices=[scheduler.value for scheduler in SchedulerType], value="cosine"
        )
        max_grad_norm = gr.Textbox(value="1.0")
        val_size = gr.Slider(value=0, minimum=0, maximum=1, step=0.001)

    with gr.Accordion(label="Advanced config", open=False) as advanced_tab:
        with gr.Row():
            logging_steps = gr.Slider(value=5, minimum=5, maximum=1000, step=5)
            save_steps = gr.Slider(value=100, minimum=10, maximum=5000, step=10)
            warmup_steps = gr.Slider(value=0, minimum=0, maximum=5000, step=1)
            compute_type = gr.Radio(choices=["fp16", "bf16"], value="fp16")
            padding_side = gr.Radio(choices=["left", "right"], value="left")

    with gr.Accordion(label="LoRA config", open=False) as lora_tab:
        with gr.Row():
            lora_rank = gr.Slider(value=8, minimum=1, maximum=1024, step=1, scale=1)
            lora_dropout = gr.Slider(value=0.1, minimum=0, maximum=1, step=0.01, scale=1)
            lora_target = gr.Textbox(scale=2)
            resume_lora_training = gr.Checkbox(value=True, scale=1)

    with gr.Accordion(label="RLHF config", open=False) as rlhf_tab:
        with gr.Row():
            dpo_beta = gr.Slider(value=0.1, minimum=0, maximum=1, step=0.01, scale=2)
            reward_model = gr.Dropdown(scale=2)
            refresh_btn = gr.Button(scale=1)

    refresh_btn.click(
        list_checkpoint,
        [top_elems["model_name"], top_elems["finetuning_type"]],
        [reward_model],
        queue=False
    )

    with gr.Row():
        cmd_preview_btn = gr.Button()
        start_btn = gr.Button()
        stop_btn = gr.Button()

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                output_dir = gr.Textbox()

            with gr.Row():
                process_bar = gr.Slider(visible=False, interactive=False)

            with gr.Box():
                output_box = gr.Markdown()

        with gr.Column(scale=1):
            loss_viewer = gr.Plot()

    input_components = [
        top_elems["lang"],
        top_elems["model_name"],
        top_elems["checkpoints"],
        top_elems["finetuning_type"],
        top_elems["quantization_bit"],
        top_elems["template"],
        top_elems["system_prompt"],
        training_stage,
        dataset_dir,
        dataset,
        max_source_length,
        max_target_length,
        learning_rate,
        num_train_epochs,
        max_samples,
        batch_size,
        gradient_accumulation_steps,
        lr_scheduler_type,
        max_grad_norm,
        val_size,
        logging_steps,
        save_steps,
        warmup_steps,
        compute_type,
        padding_side,
        lora_rank,
        lora_dropout,
        lora_target,
        resume_lora_training,
        dpo_beta,
        reward_model,
        output_dir
    ]

    output_components = [
        output_box,
        process_bar
    ]

    cmd_preview_btn.click(runner.preview_train, input_components, output_components)
    start_btn.click(runner.run_train, input_components, output_components)
    stop_btn.click(runner.set_abort, queue=False)

    process_bar.change(
        gen_plot, [top_elems["model_name"], top_elems["finetuning_type"], output_dir], loss_viewer, queue=False
    )

    return dict(
        training_stage=training_stage,
        dataset_dir=dataset_dir,
        dataset=dataset,
        data_preview_btn=data_preview_btn,
        preview_count=preview_count,
        preview_samples=preview_samples,
        close_btn=close_btn,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        max_samples=max_samples,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr_scheduler_type=lr_scheduler_type,
        max_grad_norm=max_grad_norm,
        val_size=val_size,
        advanced_tab=advanced_tab,
        logging_steps=logging_steps,
        save_steps=save_steps,
        warmup_steps=warmup_steps,
        compute_type=compute_type,
        padding_side=padding_side,
        lora_tab=lora_tab,
        lora_rank=lora_rank,
        lora_dropout=lora_dropout,
        lora_target=lora_target,
        resume_lora_training=resume_lora_training,
        rlhf_tab=rlhf_tab,
        dpo_beta=dpo_beta,
        reward_model=reward_model,
        refresh_btn=refresh_btn,
        cmd_preview_btn=cmd_preview_btn,
        start_btn=start_btn,
        stop_btn=stop_btn,
        output_dir=output_dir,
        output_box=output_box,
        loss_viewer=loss_viewer
    )
