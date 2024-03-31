import json
import os
from collections import defaultdict
from typing import Any, Dict, Optional

import gradio as gr
from peft.utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME

from ..extras.constants import (
    DATA_CONFIG,
    DEFAULT_MODULE,
    DEFAULT_TEMPLATE,
    PEFT_METHODS,
    STAGES_USE_PAIR_DATA,
    SUPPORTED_MODELS,
    TRAINING_STAGES,
    DownloadSource,
)
from ..extras.misc import use_modelscope


ADAPTER_NAMES = {WEIGHTS_NAME, SAFETENSORS_WEIGHTS_NAME}
DEFAULT_CACHE_DIR = "cache"
DEFAULT_CONFIG_DIR = "config"
DEFAULT_DATA_DIR = "data"
DEFAULT_SAVE_DIR = "saves"
USER_CONFIG = "user.config"


def get_save_dir(*args) -> os.PathLike:
    return os.path.join(DEFAULT_SAVE_DIR, *args)


def get_config_path() -> os.PathLike:
    return os.path.join(DEFAULT_CACHE_DIR, USER_CONFIG)


def get_save_path(config_path: str) -> os.PathLike:
    return os.path.join(DEFAULT_CONFIG_DIR, config_path)


def load_config() -> Dict[str, Any]:
    try:
        with open(get_config_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"lang": None, "last_model": None, "path_dict": {}, "cache_dir": None}


def save_config(lang: str, model_name: Optional[str] = None, model_path: Optional[str] = None) -> None:
    os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
    user_config = load_config()
    user_config["lang"] = lang or user_config["lang"]
    if model_name:
        user_config["last_model"] = model_name
        user_config["path_dict"][model_name] = model_path
    with open(get_config_path(), "w", encoding="utf-8") as f:
        json.dump(user_config, f, indent=2, ensure_ascii=False)


def load_args(config_path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(get_save_path(config_path), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_args(config_path: str, config_dict: Dict[str, Any]) -> str:
    os.makedirs(DEFAULT_CONFIG_DIR, exist_ok=True)
    with open(get_save_path(config_path), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    return str(get_save_path(config_path))


def get_model_path(model_name: str) -> str:
    user_config = load_config()
    path_dict: Dict[DownloadSource, str] = SUPPORTED_MODELS.get(model_name, defaultdict(str))
    model_path = user_config["path_dict"].get(model_name, None) or path_dict.get(DownloadSource.DEFAULT, None)
    if (
        use_modelscope()
        and path_dict.get(DownloadSource.MODELSCOPE)
        and model_path == path_dict.get(DownloadSource.DEFAULT)
    ):  # replace path
        model_path = path_dict.get(DownloadSource.MODELSCOPE)
    return model_path


def get_prefix(model_name: str) -> str:
    return model_name.split("-")[0]


def get_module(model_name: str) -> str:
    return DEFAULT_MODULE.get(get_prefix(model_name), "q_proj,v_proj")


def get_template(model_name: str) -> str:
    if model_name and model_name.endswith("Chat") and get_prefix(model_name) in DEFAULT_TEMPLATE:
        return DEFAULT_TEMPLATE[get_prefix(model_name)]
    return "default"


def list_adapters(model_name: str, finetuning_type: str) -> "gr.Dropdown":
    if finetuning_type not in PEFT_METHODS:
        return gr.Dropdown(value=[], choices=[], interactive=False)

    adapters = []
    if model_name and finetuning_type == "lora":
        save_dir = get_save_dir(model_name, finetuning_type)
        if save_dir and os.path.isdir(save_dir):
            for adapter in os.listdir(save_dir):
                if os.path.isdir(os.path.join(save_dir, adapter)) and any(
                    os.path.isfile(os.path.join(save_dir, adapter, name)) for name in ADAPTER_NAMES
                ):
                    adapters.append(adapter)
    return gr.Dropdown(value=[], choices=adapters, interactive=True)


def load_dataset_info(dataset_dir: str) -> Dict[str, Dict[str, Any]]:
    try:
        with open(os.path.join(dataset_dir, DATA_CONFIG), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as err:
        print("Cannot open {} due to {}.".format(os.path.join(dataset_dir, DATA_CONFIG), str(err)))
        return {}


def list_dataset(dataset_dir: str = None, training_stage: str = list(TRAINING_STAGES.keys())[0]) -> "gr.Dropdown":
    dataset_info = load_dataset_info(dataset_dir if dataset_dir is not None else DEFAULT_DATA_DIR)
    ranking = TRAINING_STAGES[training_stage] in STAGES_USE_PAIR_DATA
    datasets = [k for k, v in dataset_info.items() if v.get("ranking", False) == ranking]
    return gr.Dropdown(value=[], choices=datasets)


def autoset_packing(training_stage: str = list(TRAINING_STAGES.keys())[0]) -> "gr.Button":
    return gr.Button(value=(TRAINING_STAGES[training_stage] == "pt"))
