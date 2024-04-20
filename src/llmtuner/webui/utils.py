import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..extras.packages import is_gradio_available, is_matplotlib_available
from ..extras.ploting import smooth
from .locales import ALERTS


if is_gradio_available():
    import gradio as gr


if is_matplotlib_available():
    import matplotlib.figure
    import matplotlib.pyplot as plt


if TYPE_CHECKING:
    from ..extras.callbacks import LogCallback


def update_process_bar(callback: "LogCallback") -> "gr.Slider":
    if not callback.max_steps:
        return gr.Slider(visible=False)

    percentage = round(100 * callback.cur_steps / callback.max_steps, 0) if callback.max_steps != 0 else 100.0
    label = "Running {:d}/{:d}: {} < {}".format(
        callback.cur_steps, callback.max_steps, callback.elapsed_time, callback.remaining_time
    )
    return gr.Slider(label=label, value=percentage, visible=True)


def get_time() -> str:
    return datetime.now().strftime(r"%Y-%m-%d-%H-%M-%S")


def can_quantize(finetuning_type: str) -> "gr.Dropdown":
    if finetuning_type != "lora":
        return gr.Dropdown(value="none", interactive=False)
    else:
        return gr.Dropdown(interactive=True)


def check_json_schema(text: str, lang: str) -> None:
    try:
        tools = json.loads(text)
        if tools:
            assert isinstance(tools, list)
            for tool in tools:
                if "name" not in tool:
                    raise NotImplementedError("Name not found.")
    except NotImplementedError:
        gr.Warning(ALERTS["err_tool_name"][lang])
    except Exception:
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


def gen_plot(output_path: str) -> Optional["matplotlib.figure.Figure"]:
    log_file = os.path.join(output_path, "trainer_log.jsonl")
    if not os.path.isfile(log_file) or not is_matplotlib_available():
        return

    plt.close("all")
    plt.switch_backend("agg")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    steps, losses = [], []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            log_info: Dict[str, Any] = json.loads(line)
            if log_info.get("loss", None):
                steps.append(log_info["current_steps"])
                losses.append(log_info["loss"])

    if len(losses) == 0:
        return

    ax.plot(steps, losses, color="#1f77b4", alpha=0.4, label="original")
    ax.plot(steps, smooth(losses), color="#1f77b4", label="smoothed")
    ax.legend()
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    return fig
