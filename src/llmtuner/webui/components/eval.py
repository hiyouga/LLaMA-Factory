from typing import TYPE_CHECKING, Dict
import gradio as gr

from llmtuner.webui.common import list_dataset, DEFAULT_DATA_DIR
from llmtuner.webui.components.data import create_preview_box
from llmtuner.webui.utils import can_preview, get_preview

if TYPE_CHECKING:
    from gradio.components import Component
    from llmtuner.webui.runner import Runner


def create_eval_tab(top_elems: Dict[str, "Component"], runner: "Runner") -> Dict[str, "Component"]:
    with gr.Row():
        dataset_dir = gr.Textbox(value=DEFAULT_DATA_DIR, scale=2)
        dataset = gr.Dropdown(multiselect=True, scale=4)
        data_preview_btn = gr.Button(interactive=False, scale=1)

    preview_box, preview_count, preview_samples, close_btn = create_preview_box()

    dataset_dir.change(list_dataset, [dataset_dir], [dataset])
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
        max_samples = gr.Textbox(value="100000")
        batch_size = gr.Slider(value=8, minimum=1, maximum=512, step=1)
        predict = gr.Checkbox(value=True)

    with gr.Row():
        cmd_preview_btn = gr.Button()
        start_btn = gr.Button()
        stop_btn = gr.Button()

    with gr.Row():
        process_bar = gr.Slider(visible=False, interactive=False)

    with gr.Box():
        output_box = gr.Markdown()

    input_components = [
        top_elems["lang"],
        top_elems["model_name"],
        top_elems["checkpoints"],
        top_elems["finetuning_type"],
        top_elems["quantization_bit"],
        top_elems["template"],
        top_elems["system_prompt"],
        dataset_dir,
        dataset,
        max_source_length,
        max_target_length,
        max_samples,
        batch_size,
        predict
    ]

    output_components = [
        output_box,
        process_bar
    ]

    cmd_preview_btn.click(runner.preview_eval, input_components, output_components)
    start_btn.click(runner.run_eval, input_components, output_components)
    stop_btn.click(runner.set_abort, queue=False)

    return dict(
        dataset_dir=dataset_dir,
        dataset=dataset,
        data_preview_btn=data_preview_btn,
        preview_count=preview_count,
        preview_samples=preview_samples,
        close_btn=close_btn,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        max_samples=max_samples,
        batch_size=batch_size,
        predict=predict,
        cmd_preview_btn=cmd_preview_btn,
        start_btn=start_btn,
        stop_btn=stop_btn,
        output_box=output_box
    )
