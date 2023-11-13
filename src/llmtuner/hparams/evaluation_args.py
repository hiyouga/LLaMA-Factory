import os
from typing import Literal, Optional
from dataclasses import dataclass, field

from datasets import DownloadMode


@dataclass
class EvaluationArguments:
    r"""
    Arguments pertaining to specify the evaluation parameters.
    """
    task: str = field(
        metadata={"help": "Name of the evaluation task."}
    )
    task_dir: Optional[str] = field(
        default="evaluation",
        metadata={"help": "Path to the folder containing the evaluation datasets."}
    )
    batch_size: Optional[int] = field(
        default=4,
        metadata={"help": "The batch size per GPU for evaluation."}
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed to be used with data loaders."}
    )
    lang: Optional[Literal["en", "zh"]] = field(
        default="en",
        metadata={"help": "Language used at evaluation."}
    )
    n_shot: Optional[int] = field(
        default=5,
        metadata={"help": "Number of examplars for few-shot learning."}
    )
    save_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the evaluation results."}
    )
    download_mode: Optional[DownloadMode] = field(
        default=DownloadMode.REUSE_DATASET_IF_EXISTS,
        metadata={"help": "Download mode used for the evaluation datasets."}
    )

    def __post_init__(self):
        task_available = []
        for folder in os.listdir(self.task_dir):
            if os.path.isdir(os.path.join(self.task_dir, folder)):
                task_available.append(folder)

        if self.task not in task_available:
            raise ValueError("Task {} not found in {}.".format(self.task, self.task_dir))

        if self.save_dir is not None and os.path.exists(self.save_dir):
            raise ValueError("`save_dir` already exists, use another one.")
