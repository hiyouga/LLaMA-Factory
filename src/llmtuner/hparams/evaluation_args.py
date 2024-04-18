import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from datasets import DownloadMode


@dataclass
class EvaluationArguments:
    r"""
    Arguments pertaining to specify the evaluation parameters.
    """
    model_name : str = field(
        metadata={"help": "Name of the evaluation task."},
    )
    task: str = field(
        metadata={"help": "Name of the evaluation task."},
    )
    n_shot: int = field(
        default=5,
        metadata={"help": "Number of examplars for few-shot learning."},
    )
    n_iter: int = field(
        default=1,
        metadata={"help": "Number of examplars for few-shot learning."},
    )
    temperature: float = field(
        default=0,
        metadata={"help": "The value used to modulate the next token probabilities."},
    )
    save_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the evaluation results."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed to be used with data loaders."},
    )
    max_new_tokens: int = field(
        default=512,
        metadata={"help": "Number of examplars for few-shot learning."},
    )

    # def __post_init__(self):
    #     if self.save_dir is not None and os.path.exists(self.save_dir):
    #         raise ValueError("`save_dir` already exists, use another one.")
