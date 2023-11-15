import os
import json
import gradio as gr
from typing import Any, Dict, Optional
from transformers.utils import (
    WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    ADAPTER_WEIGHTS_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME
)

from llmtuner.extras.constants import DEFAULT_MODULE, DEFAULT_TEMPLATE, SUPPORTED_MODELS, TRAINING_STAGES


DEFAULT_CACHE_DIR = "cache"
DEFAULT_DATA_DIR = "data"
DEFAULT_SAVE_DIR = "saves"
USER_CONFIG = "user.config"
DATA_CONFIG = "dataset_info.json"
CKPT_NAMES = [
    WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    ADAPTER_WEIGHTS_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME
]


def get_save_dir(*args) -> os.PathLike:
    return os.path.join(DEFAULT_SAVE_DIR, *args)


def get_config_path() -> os.PathLike:
    return os.path.join(DEFAULT_CACHE_DIR, USER_CONFIG)


def load_config() -> Dict[str, Any]:
    try:
        with open(get_config_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except:
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


def get_model_path(model_name: str) -> str:
    user_config = load_config()
    return user_config["path_dict"].get(model_name, None) or SUPPORTED_MODELS.get(model_name, "")


def get_prefix(model_name: str) -> str:
    return model_name.split("-")[0]


def get_module(model_name: str) -> str:
    return DEFAULT_MODULE.get(get_prefix(model_name), "q_proj,v_proj")


def get_template(model_name: str) -> str:
    if model_name and model_name.endswith("Chat") and get_prefix(model_name) in DEFAULT_TEMPLATE:
        return DEFAULT_TEMPLATE[get_prefix(model_name)]
    return "default"


def list_checkpoint(model_name: str, finetuning_type: str) -> Dict[str, Any]:
    checkpoints = []
    if model_name:
        save_dir = get_save_dir(model_name, finetuning_type)
        if save_dir and os.path.isdir(save_dir):
            for checkpoint in os.listdir(save_dir):
                if (
                    os.path.isdir(os.path.join(save_dir, checkpoint))
                    and any([os.path.isfile(os.path.join(save_dir, checkpoint, name)) for name in CKPT_NAMES])
                ):
                    checkpoints.append(checkpoint)
    return gr.update(value=[], choices=checkpoints)


def load_dataset_info(dataset_dir: str) -> Dict[str, Any]:
    try:
        with open(os.path.join(dataset_dir, DATA_CONFIG), "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        print("Cannot find {} in {}.".format(DATA_CONFIG, dataset_dir))
        return {}


def list_dataset(
    dataset_dir: Optional[str] = None, training_stage: Optional[str] = list(TRAINING_STAGES.keys())[0]
) -> Dict[str, Any]:
    dataset_info = load_dataset_info(dataset_dir if dataset_dir is not None else DEFAULT_DATA_DIR)
    ranking = TRAINING_STAGES[training_stage] in ["rm", "dpo"]
    datasets = [k for k, v in dataset_info.items() if v.get("ranking", False) == ranking]
    return gr.update(value=[], choices=datasets)
