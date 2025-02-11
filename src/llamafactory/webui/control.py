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
from typing import Any, Dict, List, Optional, Tuple

from transformers.trainer_utils import get_last_checkpoint

from ..extras.constants import (
    CHECKPOINT_NAMES,
    PEFT_METHODS,
    RUNNING_LOG,
    STAGES_USE_PAIR_DATA,
    TRAINER_LOG,
    TRAINING_STAGES,
)
from ..extras.packages import is_gradio_available, is_matplotlib_available
from ..extras.ploting import gen_loss_plot
from ..model import QuantizationMethod
from .common import DEFAULT_CONFIG_DIR, DEFAULT_DATA_DIR, get_model_path, get_save_dir, get_template, load_dataset_info


if is_gradio_available():
    import gradio as gr


def can_quantize(finetuning_type: str) -> "gr.Dropdown":
    r"""
    Judges if the quantization is available in this finetuning type.

    Inputs: top.finetuning_type
    Outputs: top.quantization_bit
    """
    if finetuning_type not in PEFT_METHODS:
        return gr.Dropdown(value="none", interactive=False)
    else:
        return gr.Dropdown(interactive=True)


def can_quantize_to(quantization_method: str) -> "gr.Dropdown":
    r"""
    Gets the available quantization bits.

    Inputs: top.quantization_method
    Outputs: top.quantization_bit
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

    Inputs: train.training_stage
    Outputs: train.dataset, train.packing
    """
    return [], TRAINING_STAGES[training_stage] == "pt"


def get_model_info(model_name: str) -> Tuple[str, str]:
    r"""
    Gets the necessary information of this model.

    Inputs: top.model_name
    Outputs: top.model_path, top.template
    """
    return get_model_path(model_name), get_template(model_name)


def get_trainer_info(output_path: os.PathLike, do_train: bool) -> Tuple[str, "gr.Slider", Optional["gr.Plot"]]:
    r"""
    Gets training infomation for monitor.

    If do_train is True:
        Inputs: train.output_path
        Outputs: train.output_box, train.progress_bar, train.loss_viewer
    If do_train is False:
        Inputs: eval.output_path
        Outputs: eval.output_box, eval.progress_bar, None
    """
    running_log = ""
    running_progress = gr.Slider(visible=False)
    running_loss = None

    running_log_path = os.path.join(output_path, RUNNING_LOG)
    if os.path.isfile(running_log_path):
        with open(running_log_path, encoding="utf-8") as f:
            running_log = f.read()[-20000:]  # avoid lengthy log

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


def list_checkpoints(model_name: str, finetuning_type: str) -> "gr.Dropdown":
    r"""
    Lists all available checkpoints.

    Inputs: top.model_name, top.finetuning_type
    Outputs: top.checkpoint_path
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


def list_config_paths(current_time: str) -> "gr.Dropdown":
    r"""
    Lists all the saved configuration files.

    Inputs: train.current_time
    Outputs: train.config_path
    """
    config_files = [f"{current_time}.yaml"]
    if os.path.isdir(DEFAULT_CONFIG_DIR):
        for file_name in os.listdir(DEFAULT_CONFIG_DIR):
            if file_name.endswith(".yaml") and file_name not in config_files:
                config_files.append(file_name)

    return gr.Dropdown(choices=config_files)


def list_datasets(dataset_dir: str = None, training_stage: str = list(TRAINING_STAGES.keys())[0]) -> "gr.Dropdown":
    r"""
    Lists all available datasets in the dataset dir for the training stage.

    Inputs: *.dataset_dir, *.training_stage
    Outputs: *.dataset
    """
    dataset_info = load_dataset_info(dataset_dir if dataset_dir is not None else DEFAULT_DATA_DIR)
    ranking = TRAINING_STAGES[training_stage] in STAGES_USE_PAIR_DATA
    datasets = [k for k, v in dataset_info.items() if v.get("ranking", False) == ranking]
    return gr.Dropdown(choices=datasets)


def list_output_dirs(model_name: Optional[str], finetuning_type: str, current_time: str) -> "gr.Dropdown":
    r"""
    Lists all the directories that can resume from.

    Inputs: top.model_name, top.finetuning_type, train.current_time
    Outputs: train.output_dir
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
