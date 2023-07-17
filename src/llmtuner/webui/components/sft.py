from typing import Dict
from transformers.trainer_utils import SchedulerType

import gradio as gr
from gradio.components import Component

from llmtuner.webui.common import list_dataset, DEFAULT_DATA_DIR
from llmtuner.webui.components.data import create_preview_box
from llmtuner.webui.runner import Runner
from llmtuner.webui.utils import can_preview, get_preview, gen_plot


def create_sft_tab(top_elems: Dict[str, Component], runner: Runner) -> Dict[str, Component]:
    with gr.Row():
        dataset_dir = gr.Textbox(value=DEFAULT_DATA_DIR, interactive=True, scale=1)
        dataset = gr.Dropdown(multiselect=True, interactive=True, scale=4)
        preview_btn = gr.Button(interactive=False, scale=1)

    preview_box, preview_count, preview_samples, close_btn = create_preview_box()

    dataset_dir.change(list_dataset, [dataset_dir], [dataset])
    dataset.change(can_preview, [dataset_dir, dataset], [preview_btn])
    preview_btn.click(get_preview, [dataset_dir, dataset], [preview_count, preview_samples, preview_box])

    with gr.Row():
        learning_rate = gr.Textbox(value="5e-5", interactive=True)
        num_train_epochs = gr.Textbox(value="3.0", interactive=True)
        max_samples = gr.Textbox(value="100000", interactive=True)
        quantization_bit = gr.Dropdown([8, 4])

    with gr.Row():
        batch_size = gr.Slider(value=4, minimum=1, maximum=128, step=1, interactive=True)
        gradient_accumulation_steps = gr.Slider(value=4, minimum=1, maximum=32, step=1, interactive=True)
        lr_scheduler_type = gr.Dropdown(
            value="cosine", choices=[scheduler.value for scheduler in SchedulerType], interactive=True
        )
        fp16 = gr.Checkbox(value=True)

    with gr.Row():
        logging_steps = gr.Slider(value=5, minimum=5, maximum=1000, step=5, interactive=True)
        save_steps = gr.Slider(value=100, minimum=10, maximum=2000, step=10, interactive=True)

    with gr.Row():
        start_btn = gr.Button()
        stop_btn = gr.Button()

    with gr.Row():
        with gr.Column(scale=4):
            output_dir = gr.Textbox(interactive=True)
            output_box = gr.Markdown()

        with gr.Column(scale=1):
            loss_viewer = gr.Plot()

    start_btn.click(
        runner.run_train,
        [
            top_elems["lang"], top_elems["model_name"], top_elems["checkpoints"],
            top_elems["finetuning_type"], top_elems["template"],
            dataset, dataset_dir, learning_rate, num_train_epochs, max_samples,
            fp16, quantization_bit, batch_size, gradient_accumulation_steps,
            lr_scheduler_type, logging_steps, save_steps, output_dir
        ],
        [output_box]
    )
    stop_btn.click(runner.set_abort, queue=False)

    output_box.change(
        gen_plot, [top_elems["model_name"], top_elems["finetuning_type"], output_dir], loss_viewer, queue=False
    )

    return dict(
        dataset_dir=dataset_dir,
        dataset=dataset,
        preview_btn=preview_btn,
        preview_count=preview_count,
        preview_samples=preview_samples,
        close_btn=close_btn,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        max_samples=max_samples,
        quantization_bit=quantization_bit,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr_scheduler_type=lr_scheduler_type,
        fp16=fp16,
        logging_steps=logging_steps,
        save_steps=save_steps,
        start_btn=start_btn,
        stop_btn=stop_btn,
        output_dir=output_dir,
        output_box=output_box,
        loss_viewer=loss_viewer
    )
