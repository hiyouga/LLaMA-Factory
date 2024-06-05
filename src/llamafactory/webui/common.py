import json
import os
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

from yaml import safe_dump, safe_load

from ..extras.constants import (
    CHECKPOINT_NAMES,
    DATA_CONFIG,
    DEFAULT_TEMPLATE,
    PEFT_METHODS,
    STAGES_USE_PAIR_DATA,
    SUPPORTED_MODELS,
    TRAINING_STAGES,
    VISION_MODELS,
    DownloadSource,
)
from ..extras.logging import get_logger
from ..extras.misc import use_modelscope
from ..extras.packages import is_gradio_available


if is_gradio_available():
    import gradio as gr


logger = get_logger(__name__)


DEFAULT_CACHE_DIR = "cache"
DEFAULT_CONFIG_DIR = "config"
DEFAULT_DATA_DIR = "data"
DEFAULT_SAVE_DIR = "saves"
USER_CONFIG = "user_config.yaml"


def get_save_dir(*paths: str) -> os.PathLike:
    r"""
    Gets the path to saved model checkpoints.
    """
    paths = (path.replace(os.path.sep, "").replace(" ", "").strip() for path in paths)
    return os.path.join(DEFAULT_SAVE_DIR, *paths)


def get_config_path() -> os.PathLike:
    r"""
    Gets the path to user config.
    """
    return os.path.join(DEFAULT_CACHE_DIR, USER_CONFIG)


def get_arg_save_path(config_path: str) -> os.PathLike:
    r"""
    Gets the path to saved arguments.
    """
    return os.path.join(DEFAULT_CONFIG_DIR, config_path)


def load_config() -> Dict[str, Any]:
    r"""
    Loads user config if exists.
    """
    try:
        with open(get_config_path(), "r", encoding="utf-8") as f:
            return safe_load(f)
    except Exception:
        return {"lang": None, "last_model": None, "path_dict": {}, "cache_dir": None}


def save_config(lang: str, model_name: Optional[str] = None, model_path: Optional[str] = None) -> None:
    r"""
    Saves user config.
    """
    os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
    user_config = load_config()
    user_config["lang"] = lang or user_config["lang"]
    if model_name:
        user_config["last_model"] = model_name
        user_config["path_dict"][model_name] = model_path
    with open(get_config_path(), "w", encoding="utf-8") as f:
        safe_dump(user_config, f)


def get_model_path(model_name: str) -> Optional[str]:
    r"""
    Gets the model path according to the model name.
    """
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
    r"""
    Gets the prefix of the model name to obtain the model family.
    """
    return model_name.split("-")[0]


def get_model_info(model_name: str) -> Tuple[str, str, bool]:
    r"""
    Gets the necessary information of this model.

    Returns:
        model_path (str)
        template (str)
        visual (bool)
    """
    return get_model_path(model_name), get_template(model_name), get_visual(model_name)


def get_template(model_name: str) -> str:
    r"""
    Gets the template name if the model is a chat model.
    """
    if model_name and model_name.endswith("Chat") and get_prefix(model_name) in DEFAULT_TEMPLATE:
        return DEFAULT_TEMPLATE[get_prefix(model_name)]
    return "default"


def get_visual(model_name: str) -> bool:
    r"""
    Judges if the model is a vision language model.
    """
    return get_prefix(model_name) in VISION_MODELS


def list_checkpoints(model_name: str, finetuning_type: str) -> "gr.Dropdown":
    r"""
    Lists all available checkpoints.
    """
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
    r"""
    Loads dataset_info.json.
    """
    if dataset_dir == "ONLINE":
        logger.info("dataset_dir is ONLINE, using online dataset.")
        return {}

    try:
        with open(os.path.join(dataset_dir, DATA_CONFIG), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as err:
        logger.warning("Cannot open {} due to {}.".format(os.path.join(dataset_dir, DATA_CONFIG), str(err)))
        return {}


def list_datasets(dataset_dir: str = None, training_stage: str = list(TRAINING_STAGES.keys())[0]) -> "gr.Dropdown":
    r"""
    Lists all available datasets in the dataset dir for the training stage.
    """
    dataset_info = load_dataset_info(dataset_dir if dataset_dir is not None else DEFAULT_DATA_DIR)
    ranking = TRAINING_STAGES[training_stage] in STAGES_USE_PAIR_DATA
    datasets = [k for k, v in dataset_info.items() if v.get("ranking", False) == ranking]
    return gr.Dropdown(choices=datasets)
