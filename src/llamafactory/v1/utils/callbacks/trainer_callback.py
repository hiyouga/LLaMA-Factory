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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ...config import TrainingArguments


@dataclass
class TrainerState:
    """A read-only snapshot of training progress passed to every callback hook.

    Attributes:
        epoch: Current epoch (0-indexed).
        global_step: Number of optimizer steps completed so far.
        num_training_steps: Total number of optimizer steps planned.
        loss: Scalar loss value of the most recent step.
        grad_norm: Gradient-norm value of the most recent step.
        learning_rate: Current learning rate seen by the optimizer.
        log_history: List of per-step log dicts emitted by ``LoggingCallback``.
    """

    epoch: int = 0
    global_step: int = 0
    num_training_steps: int = 0
    loss: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    log_history: list[dict[str, Any]] = field(default_factory=list)


class TrainerCallback:
    """Abstract base class for training callbacks.

    Subclass and override whichever hooks you need.  All hooks receive:

    - ``args``      – the :class:`~llamafactory.v1.config.TrainingArguments`.
    - ``state``     – a :class:`TrainerState` snapshot (read-only).
    - ``**kwargs``  – extra keyword arguments (model, optimizer, …).

    Callbacks are *observers*: they should NOT mutate training flow.

    Hook call order::

        on_train_begin
          for each epoch:
            on_epoch_begin
              for each step:
                on_step_begin
                  (forward / backward / optimizer.step)
                on_step_end
                [on_log]   ← if this step is a logging step
            on_epoch_end
        on_train_end
    """

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, **kwargs: Any) -> None:
        """Called once before the first training step."""

    def on_train_end(self, args: TrainingArguments, state: TrainerState, **kwargs: Any) -> None:
        """Called once after the last training step."""

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, **kwargs: Any) -> None:
        """Called at the beginning of each epoch."""

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, **kwargs: Any) -> None:
        """Called at the end of each epoch."""

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, **kwargs: Any) -> None:
        """Called before the forward/backward pass of each optimizer step."""

    def on_step_end(self, args: TrainingArguments, state: TrainerState, **kwargs: Any) -> None:
        """Called after the optimizer step."""

    def on_log(self, args: TrainingArguments, state: TrainerState, logs: dict[str, Any], **kwargs: Any) -> None:
        """Called when the trainer emits a log entry."""

    def on_save(self, args: TrainingArguments, state: TrainerState, **kwargs: Any) -> None:
        """Called after the model checkpoint has been written to disk."""


class CallbackHandler:
    """Owns a list of :class:`TrainerCallback` instances and fans out hook calls.

    Usage::

        handler = CallbackHandler([LoggingCallback(), MyWandbCallback()], trainer=trainer)
        handler.on_train_begin(args, state)
    """

    def __init__(self, callbacks: list[TrainerCallback] | None = None, trainer: Any = None) -> None:
        self.callbacks: list[TrainerCallback] = list(callbacks or [])
        self.trainer = trainer

    def add_callback(self, callback: TrainerCallback) -> None:
        """Append a callback to the handler."""
        self.callbacks.append(callback)

    def _call(self, event: str, args: TrainingArguments, state: TrainerState, **kwargs: Any) -> None:
        if self.trainer is not None:
            kwargs.setdefault("model", getattr(self.trainer, "model", None))
            kwargs.setdefault("optimizer", getattr(self.trainer, "optimizer", None))
            kwargs.setdefault("lr_scheduler", getattr(self.trainer, "lr_scheduler", None))
            kwargs.setdefault("train_dataloader", getattr(self.trainer, "train_batch_generator", None))

        for cb in self.callbacks:
            getattr(cb, event)(args, state, **kwargs)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState) -> None:
        self._call("on_train_begin", args, state)

    def on_train_end(self, args: TrainingArguments, state: TrainerState) -> None:
        self._call("on_train_end", args, state)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState) -> None:
        self._call("on_epoch_begin", args, state)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState) -> None:
        self._call("on_epoch_end", args, state)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState) -> None:
        self._call("on_step_begin", args, state)

    def on_step_end(self, args: TrainingArguments, state: TrainerState) -> None:
        self._call("on_step_end", args, state)

    def on_log(self, args: TrainingArguments, state: TrainerState, logs: dict[str, Any]) -> None:
        self._call("on_log", args, state, logs=logs)

    def on_save(self, args: TrainingArguments, state: TrainerState) -> None:
        self._call("on_save", args, state)
