from typing import TYPE_CHECKING, Dict

import gradio as gr

from ..common import DEFAULT_DATA_DIR, list_dataset
from .data import create_preview_box


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


def create_eval_tab(engine: "Engine") -> Dict[str, "Component"]:
    input_elems = engine.manager.get_base_elems()
    elem_dict = dict()

    with gr.Row():
        dataset_dir = gr.Textbox(value=DEFAULT_DATA_DIR, scale=2)
        dataset = gr.Dropdown(multiselect=True, allow_custom_value=True, scale=4)
        preview_elems = create_preview_box(dataset_dir, dataset)

    input_elems.update({dataset_dir, dataset})
    elem_dict.update(dict(dataset_dir=dataset_dir, dataset=dataset, **preview_elems))

    with gr.Row():
        cutoff_len = gr.Slider(value=1024, minimum=4, maximum=8192, step=1)
        max_samples = gr.Textbox(value="100000")
        batch_size = gr.Slider(value=8, minimum=1, maximum=512, step=1)
        predict = gr.Checkbox(value=True)

    input_elems.update({cutoff_len, max_samples, batch_size, predict})
    elem_dict.update(dict(cutoff_len=cutoff_len, max_samples=max_samples, batch_size=batch_size, predict=predict))

    with gr.Row():
        max_new_tokens = gr.Slider(10, 2048, value=128, step=1)
        top_p = gr.Slider(0.01, 1, value=0.7, step=0.01)
        temperature = gr.Slider(0.01, 1.5, value=0.95, step=0.01)
        output_dir = gr.Textbox()

    input_elems.update({max_new_tokens, top_p, temperature, output_dir})
    elem_dict.update(dict(max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature, output_dir=output_dir))

    with gr.Row():
        cmd_preview_btn = gr.Button()
        start_btn = gr.Button(variant="primary")
        stop_btn = gr.Button(variant="stop")

    with gr.Row():
        resume_btn = gr.Checkbox(visible=False, interactive=False)
        process_bar = gr.Slider(visible=False, interactive=False)

    with gr.Row():
        output_box = gr.Markdown()

    output_elems = [output_box, process_bar]
    elem_dict.update(
        dict(
            cmd_preview_btn=cmd_preview_btn,
            start_btn=start_btn,
            stop_btn=stop_btn,
            resume_btn=resume_btn,
            process_bar=process_bar,
            output_box=output_box,
        )
    )

    cmd_preview_btn.click(engine.runner.preview_eval, input_elems, output_elems, concurrency_limit=None)
    start_btn.click(engine.runner.run_eval, input_elems, output_elems)
    stop_btn.click(engine.runner.set_abort)
    resume_btn.change(engine.runner.monitor, outputs=output_elems, concurrency_limit=None)

    dataset_dir.change(list_dataset, [dataset_dir], [dataset], queue=False)

    return elem_dict
