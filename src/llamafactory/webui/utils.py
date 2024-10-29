# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import signal
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import psutil
from transformers.trainer_utils import get_last_checkpoint
from yaml import safe_dump, safe_load

from ..extras.constants import PEFT_METHODS, RUNNING_LOG, TRAINER_LOG, TRAINING_ARGS, TRAINING_STAGES
from ..extras.packages import is_gradio_available, is_matplotlib_available
from ..extras.ploting import gen_loss_plot
from ..model import QuantizationMethod
from .common import DEFAULT_CACHE_DIR, DEFAULT_CONFIG_DIR, get_save_dir
from .locales import ALERTS


if is_gradio_available():
    import gradio as gr


def abort_process(pid: int) -> None:
    r"""
    Aborts the processes recursively in a bottom-up way.
    """
    try:
        children = psutil.Process(pid).children()
        if children:
            for child in children:
                abort_process(child.pid)

        os.kill(pid, signal.SIGABRT)
    except Exception:
        pass


def can_quantize(finetuning_type: str) -> "gr.Dropdown":
    r"""
    Judges if the quantization is available in this finetuning type.
    """
    if finetuning_type not in PEFT_METHODS:
        return gr.Dropdown(value="none", interactive=False)
    else:
        return gr.Dropdown(interactive=True)


def can_quantize_to(quantization_method: str) -> "gr.Dropdown":
    r"""
    Returns the available quantization bits.
    """
    if quantization_method == QuantizationMethod.BITS_AND_BYTES.value:
        available_bits = ["none", "8", "4"]
    elif quantization_method == QuantizationMethod.HQQ.value:
        available_bits = ["none", "8", "6", "5", "4", "3", "2", "1"]
    elif quantization_method == QuantizationMethod.EETQ.value:
        available_bits = ["none", "8"]

    return gr.Dropdown(choices=available_bits)


def change_stage(training_stage: str = list(TRAINING_STAGES.keys())[0]) -> Tuple[List[str], bool]:
    r"""
    Modifys states after changing the training stage.
    """
    return [], TRAINING_STAGES[training_stage] == "pt"


def check_json_schema(text: str, lang: str) -> None:
    r"""
    Checks if the json schema is valid.
    """
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
    r"""
    Removes args with NoneType or False or empty string value.
    """
    no_skip_keys = ["packing"]
    return {k: v for k, v in args.items() if (k in no_skip_keys) or (v is not None and v is not False and v != "")}


def gen_cmd(args: Dict[str, Any]) -> str:
    r"""
    Generates arguments for previewing.
    """
    cmd_lines = ["llamafactory-cli train "]
    for k, v in clean_cmd(args).items():
        cmd_lines.append(f"    --{k} {str(v)} ")

    if os.name == "nt":
        cmd_text = "`\n".join(cmd_lines)
    else:
        cmd_text = "\\\n".join(cmd_lines)

    cmd_text = f"```bash\n{cmd_text}\n```"
    return cmd_text


def save_cmd(args: Dict[str, Any]) -> str:
    r"""
    Saves arguments to launch training.
    """
    output_dir = args["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, TRAINING_ARGS), "w", encoding="utf-8") as f:
        safe_dump(clean_cmd(args), f)

    return os.path.join(output_dir, TRAINING_ARGS)


def get_eval_results(path: os.PathLike) -> str:
    r"""
    Gets scores after evaluation.
    """
    with open(path, encoding="utf-8") as f:
        result = json.dumps(json.load(f), indent=4)
    return f"```json\n{result}\n```\n"


def get_time() -> str:
    r"""
    Gets current date and time.
    """
    return datetime.now().strftime(r"%Y-%m-%d-%H-%M-%S")


def get_trainer_info(output_path: os.PathLike, do_train: bool) -> Tuple[str, "gr.Slider", Optional["gr.Plot"]]:
    r"""
    Gets training infomation for monitor.
    """
    running_log = ""
    running_progress = gr.Slider(visible=False)
    running_loss = None

    running_log_path = os.path.join(output_path, RUNNING_LOG)
    if os.path.isfile(running_log_path):
        with open(running_log_path, encoding="utf-8") as f:
            running_log = f.read()

    trainer_log_path = os.path.join(output_path, TRAINER_LOG)
    if os.path.isfile(trainer_log_path):
        trainer_log: List[Dict[str, Any]] = []
        with open(trainer_log_path, encoding="utf-8") as f:
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


def load_args(config_path: str) -> Optional[Dict[str, Any]]:
    r"""
    Loads saved arguments.
    """
    try:
        with open(config_path, encoding="utf-8") as f:
            return safe_load(f)
    except Exception:
        return None


def save_args(config_path: str, config_dict: Dict[str, Any]):
    r"""
    Saves arguments.
    """
    with open(config_path, "w", encoding="utf-8") as f:
        safe_dump(config_dict, f)


def list_config_paths(current_time: str) -> "gr.Dropdown":
    r"""
    Lists all the saved configuration files.
    """
    config_files = [f"{current_time}.yaml"]
    if os.path.isdir(DEFAULT_CONFIG_DIR):
        for file_name in os.listdir(DEFAULT_CONFIG_DIR):
            if file_name.endswith(".yaml") and file_name not in config_files:
                config_files.append(file_name)

    return gr.Dropdown(choices=config_files)


def list_output_dirs(model_name: Optional[str], finetuning_type: str, current_time: str) -> "gr.Dropdown":
    r"""
    Lists all the directories that can resume from.
    """
    output_dirs = [f"train_{current_time}"]
    if model_name:
        save_dir = get_save_dir(model_name, finetuning_type)
        if save_dir and os.path.isdir(save_dir):
            for folder in os.listdir(save_dir):
                output_dir = os.path.join(save_dir, folder)
                if os.path.isdir(output_dir) and get_last_checkpoint(output_dir) is not None:
                    output_dirs.append(folder)

    return gr.Dropdown(choices=output_dirs)


def create_ds_config() -> None:
    r"""
    Creates deepspeed config.
    """
    os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
    ds_config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "zero_allow_untested_optimizer": True,
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "bf16": {"enabled": "auto"},
    }
    offload_config = {
        "device": "cpu",
        "pin_memory": True,
    }
    ds_config["zero_optimization"] = {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True,
        "round_robin_gradients": True,
    }
    with open(os.path.join(DEFAULT_CACHE_DIR, "ds_z2_config.json"), "w", encoding="utf-8") as f:
        json.dump(ds_config, f, indent=2)

    ds_config["zero_optimization"]["offload_optimizer"] = offload_config
    with open(os.path.join(DEFAULT_CACHE_DIR, "ds_z2_offload_config.json"), "w", encoding="utf-8") as f:
        json.dump(ds_config, f, indent=2)

    ds_config["zero_optimization"] = {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
    }
    with open(os.path.join(DEFAULT_CACHE_DIR, "ds_z3_config.json"), "w", encoding="utf-8") as f:
        json.dump(ds_config, f, indent=2)

    ds_config["zero_optimization"]["offload_optimizer"] = offload_config
    ds_config["zero_optimization"]["offload_param"] = offload_config
    with open(os.path.join(DEFAULT_CACHE_DIR, "ds_z3_offload_config.json"), "w", encoding="utf-8") as f:
        json.dump(ds_config, f, indent=2)
