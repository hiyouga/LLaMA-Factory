# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClipArguments:
    r"""Arguments pertaining to what data we are going to input our model for training and evaluation."""

    root: Optional[str] = field(
        default=None,
        metadata={"help": "dataset path"},
    )
    trainer: Optional[str] = field(
        default=None,
        metadata={"help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."},
    )
    clip_batch_size: int = field(
        default=32,
        metadata={"help": "Train bs"},
    )
    clip_logging_steps: int = field(
        default=5,
        metadata={"help": "logging_steps"},
    )
    few_shot_num: int = field(
        default=0,
        metadata={"help": "few_shot_num"},
    )
    dataset_config_file: Optional[str] = field(
        default=None,
        metadata={"help": "The name of dataset(s) to use for evaluation. Use commas to separate multiple datasets."},
    )
    config_file: str = field(
        default="data",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    xpu: bool = field(
        default=None,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )
    use_optuna: bool = field(
        default=None,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )
    source_domains: Optional[str] = field(
        default=None,
        metadata={"help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."},
    )
    target_domains: Optional[str] = field(
        default=None,
        metadata={"help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."},
    )
    transforms: Optional[str] = field(
        default=None,
        metadata={"help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."},
    )
    backbone: Optional[str] = field(
        default=None,
        metadata={"help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."},
    )
    head: Optional[str] = field(
        default=None,
        metadata={"help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."},
    )
    no_train: Optional[int] = field(
        default=0,
        metadata={"help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."},
    )
    clip_bias_term: Optional[str] = field(
        default=None,
        metadata={"help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."},
    )
    clip_bias_exclude: Optional[str] = field(
        default=None,
        metadata={"help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."},
    )
    use_abs: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )
    use_abs_group: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )
    abs_group_name: Optional[str] = field(
        default=None,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )
    keep_min: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )
    keep_layers: Optional[int] = field(
        default=5,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )
    tip_load_cache: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )
    augment_epoch: Optional[int] = field(
        default=10,
        metadata={"help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."},
    )
    tip_beta: Optional[float] = field(
        default=1.0,
        metadata={"help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."},
    )
    tip_alpha: Optional[float] = field(
        default=3.0,
        metadata={"help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."},
    )
    new: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )
    new_dataset: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )
    search_best: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg
