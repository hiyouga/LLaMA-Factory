import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from yaml import safe_dump

from ..extras.constants import RUNNING_LOG, TRAINER_CONFIG, TRAINER_LOG
from ..extras.packages import is_gradio_available, is_matplotlib_available
from ..extras.ploting import gen_loss_plot
from .locales import ALERTS


if is_gradio_available():
    import gradio as gr


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


def clean_cmd(args: Dict[str, Any]) -> Dict[str, Any]:
    no_skip_keys = ["packing"]
    return {k: v for k, v in args.items() if (k in no_skip_keys) or (v is not None and v is not False and v != "")}


def gen_cmd(args: Dict[str, Any]) -> str:
    current_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    cmd_lines = ["CUDA_VISIBLE_DEVICES={} llamafactory-cli train ".format(current_devices)]
    for k, v in clean_cmd(args).items():
        cmd_lines.append("    --{} {} ".format(k, str(v)))

    cmd_text = "\\\n".join(cmd_lines)
    cmd_text = "```bash\n{}\n```".format(cmd_text)
    return cmd_text


def get_eval_results(path: os.PathLike) -> str:
    with open(path, "r", encoding="utf-8") as f:
        result = json.dumps(json.load(f), indent=4)
    return "```json\n{}\n```\n".format(result)


def get_time() -> str:
    return datetime.now().strftime(r"%Y-%m-%d-%H-%M-%S")


def get_trainer_info(output_path: os.PathLike, do_train: bool) -> Tuple[str, "gr.Slider", Optional["gr.Plot"]]:
    running_log = ""
    running_progress = gr.Slider(visible=False)
    running_loss = None

    running_log_path = os.path.join(output_path, RUNNING_LOG)
    if os.path.isfile(running_log_path):
        with open(running_log_path, "r", encoding="utf-8") as f:
            running_log = f.read()

    trainer_log_path = os.path.join(output_path, TRAINER_LOG)
    if os.path.isfile(trainer_log_path):
        trainer_log: List[Dict[str, Any]] = []
        with open(trainer_log_path, "r", encoding="utf-8") as f:
            for line in f:
                trainer_log.append(json.loads(line))

        if len(trainer_log) != 0:
            latest_log = trainer_log[-1]
            percentage = latest_log["percentage"]
            label = "Running {:d}/{:d}: {} < {}".format(
                latest_log["current_steps"],
                latest_log["total_steps"],
                latest_log["elapsed_time"],
                latest_log["remaining_time"],
            )
            running_progress = gr.Slider(label=label, value=percentage, visible=True)

            if do_train and is_matplotlib_available():
                running_loss = gr.Plot(gen_loss_plot(trainer_log))

    return running_log, running_progress, running_loss


def save_cmd(args: Dict[str, Any]) -> str:
    output_dir = args["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, TRAINER_CONFIG), "w", encoding="utf-8") as f:
        safe_dump(clean_cmd(args), f)

    return os.path.join(output_dir, TRAINER_CONFIG)
