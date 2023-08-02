import os
import json
import gradio as gr
import matplotlib.figure
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Tuple
from datetime import datetime

from llmtuner.extras.ploting import smooth
from llmtuner.tuner import export_model
from llmtuner.webui.common import get_model_path, get_save_dir, DATA_CONFIG
from llmtuner.webui.locales import ALERTS

if TYPE_CHECKING:
    from llmtuner.extras.callbacks import LogCallback


def format_info(log: str, callback: "LogCallback") -> str:
    info = log
    if callback.max_steps:
        info += "Running **{:d}/{:d}**: {} < {}\n".format(
            callback.cur_steps, callback.max_steps, callback.elapsed_time, callback.remaining_time
        )
    return info


def get_time() -> str:
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


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


def get_preview(dataset_dir: str, dataset: list) -> Tuple[int, list, Dict[str, Any]]:
    with open(os.path.join(dataset_dir, DATA_CONFIG), "r", encoding="utf-8") as f:
        dataset_info = json.load(f)
    data_file = dataset_info[dataset[0]]["file_name"]
    with open(os.path.join(dataset_dir, data_file), "r", encoding="utf-8") as f:
        data = json.load(f)
    return len(data), data[:2], gr.update(visible=True)


def can_quantize(finetuning_type: str) -> Dict[str, Any]:
    if finetuning_type != "lora":
        return gr.update(value="", interactive=False)
    else:
        return gr.update(interactive=True)


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

    if len(losses) == 0:
        return None

    ax.plot(steps, losses, alpha=0.4, label="original")
    ax.plot(steps, smooth(losses), label="smoothed")
    ax.legend()
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    return fig


def save_model(
    lang: str, model_name: str, checkpoints: List[str], finetuning_type: str, max_shard_size: int, save_dir: str
) -> Generator[str, None, None]:
    if not model_name:
        yield ALERTS["err_no_model"][lang]
        return

    model_name_or_path = get_model_path(model_name)
    if not model_name_or_path:
        yield ALERTS["err_no_path"][lang]
        return

    if not checkpoints:
        yield ALERTS["err_no_checkpoint"][lang]
        return

    checkpoint_dir = ",".join(
            [os.path.join(get_save_dir(model_name), finetuning_type, checkpoint) for checkpoint in checkpoints]
        )

    if not save_dir:
        yield ALERTS["err_no_save_dir"][lang]
        return

    args = dict(
        model_name_or_path=model_name_or_path,
        checkpoint_dir=checkpoint_dir,
        finetuning_type=finetuning_type,
        output_dir=save_dir
    )

    yield ALERTS["info_exporting"][lang]
    export_model(args, max_shard_size="{}GB".format(max_shard_size))
    yield ALERTS["info_exported"][lang]
