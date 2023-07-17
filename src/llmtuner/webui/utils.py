import os
import json
import gradio as gr
import matplotlib.figure
import matplotlib.pyplot as plt
from typing import Tuple
from datetime import datetime

from llmtuner.extras.ploting import smooth
from llmtuner.webui.common import get_save_dir, DATA_CONFIG


def format_info(log: str, tracker: dict) -> str:
    info = log
    if "current_steps" in tracker:
        info += "Running **{:d}/{:d}**: {} < {}\n".format(
            tracker["current_steps"], tracker["total_steps"], tracker["elapsed_time"], tracker["remaining_time"]
        )
    return info


def get_time() -> str:
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def can_preview(dataset_dir: str, dataset: list) -> dict:
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


def get_preview(dataset_dir: str, dataset: list) -> Tuple[int, list, dict]:
    with open(os.path.join(dataset_dir, DATA_CONFIG), "r", encoding="utf-8") as f:
        dataset_info = json.load(f)
    data_file = dataset_info[dataset[0]]["file_name"]
    with open(os.path.join(dataset_dir, data_file), "r", encoding="utf-8") as f:
        data = json.load(f)
    return len(data), data[:2], gr.update(visible=True)


def get_eval_results(path: os.PathLike) -> str:
    with open(path, "r", encoding="utf-8") as f:
        result = json.dumps(json.load(f), indent=4)
    return "```json\n{}\n```\n".format(result)


def gen_plot(base_model: str, finetuning_type: str, output_dir: str) -> matplotlib.figure.Figure:
    log_file = os.path.join(get_save_dir(base_model), finetuning_type, output_dir, "trainer_log.jsonl")
    if not os.path.isfile(log_file):
        return None

    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    steps, losses = [], []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            log_info = json.loads(line)
            if log_info.get("loss", None):
                steps.append(log_info["current_steps"])
                losses.append(log_info["loss"])
    ax.plot(steps, losses, alpha=0.4, label="original")
    ax.plot(steps, smooth(losses), label="smoothed")
    ax.legend()
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    return fig
