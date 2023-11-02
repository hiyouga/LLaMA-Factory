import os
import json
import gradio as gr
from typing import TYPE_CHECKING, Any, Dict, Tuple

from llmtuner.webui.common import DATA_CONFIG

if TYPE_CHECKING:
    from gradio.components import Component


PAGE_SIZE = 2


def prev_page(page_index: int) -> int:
    return page_index - 1 if page_index > 0 else page_index


def next_page(page_index: int, total_num: int) -> int:
    return page_index + 1 if (page_index + 1) * PAGE_SIZE < total_num else page_index


def can_preview(dataset_dir: str, dataset: list) -> Dict[str, Any]:
    with open(os.path.join(dataset_dir, DATA_CONFIG), "r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    if (
        len(dataset) > 0
        and "file_name" in dataset_info[dataset[0]]
        and os.path.isfile(os.path.join(dataset_dir, dataset_info[dataset[0]]["file_name"]))
    ):
        return gr.update(interactive=True)
    else:
        return gr.update(interactive=False)


def get_preview(dataset_dir: str, dataset: list, page_index: int) -> Tuple[int, list, Dict[str, Any]]:
    with open(os.path.join(dataset_dir, DATA_CONFIG), "r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    data_file: str = dataset_info[dataset[0]]["file_name"]
    with open(os.path.join(dataset_dir, data_file), "r", encoding="utf-8") as f:
        if data_file.endswith(".json"):
            data = json.load(f)
        elif data_file.endswith(".jsonl"):
            data = [json.loads(line) for line in f]
        else:
            data = [line for line in f]
    return len(data), data[PAGE_SIZE * page_index : PAGE_SIZE * (page_index + 1)], gr.update(visible=True)


def create_preview_box(dataset_dir: "gr.Textbox", dataset: "gr.Dropdown") -> Dict[str, "Component"]:
    data_preview_btn = gr.Button(interactive=False, scale=1)
    with gr.Column(visible=False, elem_classes="modal-box") as preview_box:
        with gr.Row():
            preview_count = gr.Number(value=0, interactive=False, precision=0)
            page_index = gr.Number(value=0, interactive=False, precision=0)

        with gr.Row():
            prev_btn = gr.Button()
            next_btn = gr.Button()
            close_btn = gr.Button()

        with gr.Row():
            preview_samples = gr.JSON(interactive=False)

    dataset.change(
        can_preview, [dataset_dir, dataset], [data_preview_btn], queue=False
    ).then(
        lambda: 0, outputs=[page_index], queue=False
    )
    data_preview_btn.click(
        get_preview,
        [dataset_dir, dataset, page_index],
        [preview_count, preview_samples, preview_box],
        queue=False
    )
    prev_btn.click(
        prev_page, [page_index], [page_index], queue=False
    ).then(
        get_preview,
        [dataset_dir, dataset, page_index],
        [preview_count, preview_samples, preview_box],
        queue=False
    )
    next_btn.click(
        next_page, [page_index, preview_count], [page_index], queue=False
    ).then(
        get_preview,
        [dataset_dir, dataset, page_index],
        [preview_count, preview_samples, preview_box],
        queue=False
    )
    close_btn.click(lambda: gr.update(visible=False), outputs=[preview_box], queue=False)
    return dict(
        data_preview_btn=data_preview_btn,
        preview_count=preview_count,
        page_index=page_index,
        prev_btn=prev_btn,
        next_btn=next_btn,
        close_btn=close_btn,
        preview_samples=preview_samples
    )
