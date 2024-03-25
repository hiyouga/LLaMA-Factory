import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from datasets import DownloadMode


@dataclass
class EvaluationArguments:
    r"""
    Arguments pertaining to specify the evaluation parameters.
    """

    task: str = field(
        metadata={"help": "Name of the evaluation task."},
    )
    task_dir: str = field(
        default="evaluation",
        metadata={"help": "Path to the folder containing the evaluation datasets."},
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "The batch size per GPU for evaluation."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed to be used with data loaders."},
    )
    lang: Literal["en", "zh"] = field(
        default="en",
        metadata={"help": "Language used at evaluation."},
    )
    n_shot: int = field(
        default=5,
        metadata={"help": "Number of examplars for few-shot learning."},
    )
    save_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the evaluation results."},
    )
    download_mode: DownloadMode = field(
        default=DownloadMode.REUSE_DATASET_IF_EXISTS,
        metadata={"help": "Download mode used for the evaluation datasets."},
    )

    def __post_init__(self):
        if self.save_dir is not None and os.path.exists(self.save_dir):
            raise ValueError("`save_dir` already exists, use another one.")
