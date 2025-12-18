# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OptunaArguments:
    r"""Arguments pertaining to what data we are going to input our model for training and evaluation."""

    optuna: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use optuna"},
    )

    n_trials: int = field(
        default=30,
        metadata={"help": "Train bs"},
    )
    n_warmup_steps: int = field(
        default=15,
        metadata={"help": "Train bs"},
    )
    sampler: str = field(
        default="TPESampler",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    opt_params: str = field(
        default=None,
        metadata={"help": "Path to the folder containing the datasets."},
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg
