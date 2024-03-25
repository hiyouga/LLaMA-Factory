from typing import TYPE_CHECKING

from transformers import Trainer

from ...extras.logging import get_logger
from ..utils import create_custom_optimzer


if TYPE_CHECKING:
    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class CustomTrainer(Trainer):
    r"""
    Inherits Trainer for custom optimizer.
    """

    def __init__(self, finetuning_args: "FinetuningArguments", **kwargs) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args

    def create_optimizer_and_scheduler(self, num_training_steps: int) -> None:
        self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args, num_training_steps)
        if self.optimizer is None:
            self.create_optimizer()

        self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)
