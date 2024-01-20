import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict

import gradio as gr

from ..extras.packages import is_matplotlib_available
from ..extras.ploting import smooth
from .common import get_save_dir
from .locales import ALERTS


if TYPE_CHECKING:
    from ..extras.callbacks import LogCallback

if is_matplotlib_available():
    import matplotlib.figure
    import matplotlib.pyplot as plt


def update_process_bar(callback: "LogCallback") -> Dict[str, Any]:
    if not callback.max_steps:
        return gr.update(visible=False)

    percentage = round(100 * callback.cur_steps / callback.max_steps, 0) if callback.max_steps != 0 else 100.0
    label = "Running {:d}/{:d}: {} < {}".format(
        callback.cur_steps, callback.max_steps, callback.elapsed_time, callback.remaining_time
    )
    return gr.update(label=label, value=percentage, visible=True)


def get_time() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def can_quantize(finetuning_type: str) -> Dict[str, Any]:
    if finetuning_type != "lora":
        return gr.update(value="None", interactive=False)
    else:
        return gr.update(interactive=True)


def check_json_schema(text: str, lang: str) -> None:
    try:
        tools = json.loads(text)
        for tool in tools:
            assert "name" in tool
    except AssertionError:
        gr.Warning(ALERTS["err_tool_name"][lang])
    except json.JSONDecodeError:
        gr.Warning(ALERTS["err_json_schema"][lang])


def gen_cmd(args: Dict[str, Any]) -> str:
    args.pop("disable_tqdm", None)
    args["plot_loss"] = args.get("do_train", None)
    current_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    cmd_lines = ["CUDA_VISIBLE_DEVICES={} python src/train_bash.py ".format(current_devices)]
    for k, v in args.items():
        if v is not None and v is not False and v != "":
            cmd_lines.append("    --{} {} ".format(k, str(v)))
    cmd_text = "\\\n".join(cmd_lines)
    cmd_text = "```bash\n{}\n```".format(cmd_text)
    return cmd_text


def get_eval_results(path: os.PathLike) -> str:
    with open(path, "r", encoding="utf-8") as f:
        result = json.dumps(json.load(f), indent=4)
    return "```json\n{}\n```\n".format(result)


def gen_plot(base_model: str, finetuning_type: str, output_dir: str) -> "matplotlib.figure.Figure":
    if not base_model:
        return
    log_file = get_save_dir(base_model, finetuning_type, output_dir, "trainer_log.jsonl")
    if not os.path.isfile(log_file):
        return

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
