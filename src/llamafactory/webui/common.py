# Copyright 2025 the LlamaFactory team.
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
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional, Union

from psutil import Process
from yaml import safe_dump, safe_load

from ..extras import logging
from ..extras.constants import (
    DATA_CONFIG,
    DEFAULT_TEMPLATE,
    MULTIMODAL_SUPPORTED_MODELS,
    SUPPORTED_MODELS,
    TRAINING_ARGS,
    DownloadSource,
)
from ..extras.misc import use_modelscope, use_openmind


logger = logging.get_logger(__name__)

DEFAULT_CACHE_DIR = "cache"
DEFAULT_CONFIG_DIR = "config"
DEFAULT_DATA_DIR = "data"
DEFAULT_SAVE_DIR = "saves"
USER_CONFIG = "user_config.yaml"


def abort_process(pid: int) -> None:
    r"""Abort the processes recursively in a bottom-up way."""
    try:
        children = Process(pid).children()
        if children:
            for child in children:
                abort_process(child.pid)

        os.kill(pid, signal.SIGABRT)
    except Exception:
        pass


def get_save_dir(*paths: str) -> os.PathLike:
<<<<<<< HEAD
    r"""Gets the path to saved model checkpoints."""
=======
    r"""Get the path to saved model checkpoints."""
>>>>>>> 7e0cdb1a76c1ac6b69de86d4ba40e9395f883cdb
    if os.path.sep in paths[-1]:
        logger.warning_rank0("Found complex path, some features may be not available.")
        return paths[-1]

    paths = (path.replace(" ", "").strip() for path in paths)
    return os.path.join(DEFAULT_SAVE_DIR, *paths)


<<<<<<< HEAD
def get_config_path() -> os.PathLike:
    r"""Gets the path to user config."""
    return os.path.join(DEFAULT_CACHE_DIR, USER_CONFIG)


def load_config() -> Dict[str, Any]:
    r"""Loads user config if exists."""
=======
def _get_config_path() -> os.PathLike:
    r"""Get the path to user config."""
    return os.path.join(DEFAULT_CACHE_DIR, USER_CONFIG)


def load_config() -> dict[str, Union[str, dict[str, Any]]]:
    r"""Load user config if exists."""
>>>>>>> 7e0cdb1a76c1ac6b69de86d4ba40e9395f883cdb
    try:
        with open(_get_config_path(), encoding="utf-8") as f:
            return safe_load(f)
    except Exception:
        return {"lang": None, "last_model": None, "path_dict": {}, "cache_dir": None}


def save_config(lang: str, model_name: Optional[str] = None, model_path: Optional[str] = None) -> None:
<<<<<<< HEAD
    r"""Saves user config."""
=======
    r"""Save user config."""
>>>>>>> 7e0cdb1a76c1ac6b69de86d4ba40e9395f883cdb
    os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
    user_config = load_config()
    user_config["lang"] = lang or user_config["lang"]
    if model_name:
        user_config["last_model"] = model_name

    if model_name and model_path:
        user_config["path_dict"][model_name] = model_path

    with open(_get_config_path(), "w", encoding="utf-8") as f:
        safe_dump(user_config, f)


def get_model_path(model_name: str) -> str:
<<<<<<< HEAD
    r"""Gets the model path according to the model name."""
=======
    r"""Get the model path according to the model name."""
>>>>>>> 7e0cdb1a76c1ac6b69de86d4ba40e9395f883cdb
    user_config = load_config()
    path_dict: dict[DownloadSource, str] = SUPPORTED_MODELS.get(model_name, defaultdict(str))
    model_path = user_config["path_dict"].get(model_name, "") or path_dict.get(DownloadSource.DEFAULT, "")
    if (
        use_modelscope()
        and path_dict.get(DownloadSource.MODELSCOPE)
        and model_path == path_dict.get(DownloadSource.DEFAULT)
    ):  # replace hf path with ms path
        model_path = path_dict.get(DownloadSource.MODELSCOPE)

    if (
        use_openmind()
        and path_dict.get(DownloadSource.OPENMIND)
        and model_path == path_dict.get(DownloadSource.DEFAULT)
    ):  # replace hf path with om path
        model_path = path_dict.get(DownloadSource.OPENMIND)

    return model_path


<<<<<<< HEAD
def get_prefix(model_name: str) -> str:
    r"""Gets the prefix of the model name to obtain the model family."""
    return model_name.split("-")[0]


def get_model_info(model_name: str) -> Tuple[str, str]:
    r"""Gets the necessary information of this model.

    Returns:
        model_path (str)
        template (str)
    """
    return get_model_path(model_name), get_template(model_name)


def get_template(model_name: str) -> str:
    r"""Gets the template name if the model is a chat model."""
    if model_name and model_name.endswith("Chat") and get_prefix(model_name) in DEFAULT_TEMPLATE:
        return DEFAULT_TEMPLATE[get_prefix(model_name)]
    return "default"


def get_visual(model_name: str) -> bool:
    r"""Judges if the model is a vision language model."""
    return get_prefix(model_name) in VISION_MODELS


def list_checkpoints(model_name: str, finetuning_type: str) -> "gr.Dropdown":
    r"""Lists all available checkpoints."""
    checkpoints = []
    if model_name:
        save_dir = get_save_dir(model_name, finetuning_type)
        if save_dir and os.path.isdir(save_dir):
            for checkpoint in os.listdir(save_dir):
                if os.path.isdir(os.path.join(save_dir, checkpoint)) and any(
                    os.path.isfile(os.path.join(save_dir, checkpoint, name)) for name in CHECKPOINT_NAMES
                ):
                    checkpoints.append(checkpoint)

    if finetuning_type in PEFT_METHODS:
        return gr.Dropdown(value=[], choices=checkpoints, multiselect=True)
    else:
        return gr.Dropdown(value=None, choices=checkpoints, multiselect=False)


def load_dataset_info(dataset_dir: str) -> Dict[str, Dict[str, Any]]:
    r"""Loads dataset_info.json."""
=======
def get_template(model_name: str) -> str:
    r"""Get the template name if the model is a chat/distill/instruct model."""
    return DEFAULT_TEMPLATE.get(model_name, "default")


def get_time() -> str:
    r"""Get current date and time."""
    return datetime.now().strftime(r"%Y-%m-%d-%H-%M-%S")


def is_multimodal(model_name: str) -> bool:
    r"""Judge if the model is a vision language model."""
    return model_name in MULTIMODAL_SUPPORTED_MODELS


def load_dataset_info(dataset_dir: str) -> dict[str, dict[str, Any]]:
    r"""Load dataset_info.json."""
>>>>>>> 7e0cdb1a76c1ac6b69de86d4ba40e9395f883cdb
    if dataset_dir == "ONLINE" or dataset_dir.startswith("REMOTE:"):
        logger.info_rank0(f"dataset_dir is {dataset_dir}, using online dataset.")
        return {}

    try:
        with open(os.path.join(dataset_dir, DATA_CONFIG), encoding="utf-8") as f:
            return json.load(f)
    except Exception as err:
        logger.warning_rank0(f"Cannot open {os.path.join(dataset_dir, DATA_CONFIG)} due to {str(err)}.")
        return {}


<<<<<<< HEAD
def list_datasets(dataset_dir: str = None, training_stage: str = list(TRAINING_STAGES.keys())[0]) -> "gr.Dropdown":
    r"""Lists all available datasets in the dataset dir for the training stage."""
    dataset_info = load_dataset_info(dataset_dir if dataset_dir is not None else DEFAULT_DATA_DIR)
    ranking = TRAINING_STAGES[training_stage] in STAGES_USE_PAIR_DATA
    datasets = [k for k, v in dataset_info.items() if v.get("ranking", False) == ranking]
    return gr.Dropdown(choices=datasets)
=======
def load_args(config_path: str) -> Optional[dict[str, Any]]:
    r"""Load the training configuration from config path."""
    try:
        with open(config_path, encoding="utf-8") as f:
            return safe_load(f)
    except Exception:
        return None


def save_args(config_path: str, config_dict: dict[str, Any]) -> None:
    r"""Save the training configuration to config path."""
    with open(config_path, "w", encoding="utf-8") as f:
        safe_dump(config_dict, f)


def _clean_cmd(args: dict[str, Any]) -> dict[str, Any]:
    r"""Remove args with NoneType or False or empty string value."""
    no_skip_keys = ["packing"]
    return {k: v for k, v in args.items() if (k in no_skip_keys) or (v is not None and v is not False and v != "")}


def gen_cmd(args: dict[str, Any]) -> str:
    r"""Generate CLI commands for previewing."""
    cmd_lines = ["llamafactory-cli train "]
    for k, v in _clean_cmd(args).items():
        if isinstance(v, dict):
            cmd_lines.append(f"    --{k} {json.dumps(v, ensure_ascii=False)} ")
        elif isinstance(v, list):
            cmd_lines.append(f"    --{k} {' '.join(map(str, v))} ")
        else:
            cmd_lines.append(f"    --{k} {str(v)} ")

    if os.name == "nt":
        cmd_text = "`\n".join(cmd_lines)
    else:
        cmd_text = "\\\n".join(cmd_lines)

    cmd_text = f"```bash\n{cmd_text}\n```"
    return cmd_text


def save_cmd(args: dict[str, Any]) -> str:
    r"""Save CLI commands to launch training."""
    output_dir = args["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, TRAINING_ARGS), "w", encoding="utf-8") as f:
        safe_dump(_clean_cmd(args), f)

    return os.path.join(output_dir, TRAINING_ARGS)


def load_eval_results(path: os.PathLike) -> str:
    r"""Get scores after evaluation."""
    with open(path, encoding="utf-8") as f:
        result = json.dumps(json.load(f), indent=4)

    return f"```json\n{result}\n```\n"


def create_ds_config() -> None:
    r"""Create deepspeed config in the current directory."""
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
>>>>>>> 7e0cdb1a76c1ac6b69de86d4ba40e9395f883cdb
