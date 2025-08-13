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

from typing import TYPE_CHECKING

from transformers import TrainerCallback, TrainerControl, TrainerState

from ...extras import logging


if TYPE_CHECKING:
    from transformers import TrainingArguments

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


class QATCallback(TrainerCallback):
    """Callback to control fake quantization during QAT training."""

    def __init__(self, model_args: "ModelArguments"):
        self.model_args = model_args
        self.fake_quant_enabled = False
        self.fake_quant_after_n_steps = model_args.fake_quant_after_n_steps

    def on_train_begin(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        """Initialize fake quantization state at training start."""
        if not self.model_args.enable_qat:
            return

        # If fake_quant_after_n_steps is None or 0, enable from start
        if self.fake_quant_after_n_steps is None or self.fake_quant_after_n_steps <= 0:
            self._enable_fake_quantization(model)
        else:
            self._disable_fake_quantization(model)
            logger.info_rank0(f"QAT: Fake quantization will be enabled after {self.fake_quant_after_n_steps} steps")

    def on_step_end(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        """Check if fake quantization should be enabled at this step."""
        if not self.model_args.enable_qat or self.fake_quant_enabled:
            return

        if self.fake_quant_after_n_steps is not None and state.global_step >= self.fake_quant_after_n_steps:
            self._enable_fake_quantization(model)
            logger.info_rank0(f"QAT: Enabled fake quantization at step {state.global_step}")

    def _enable_fake_quantization(self, model):
        """Enable fake quantization in the model."""
        if model is None:
            return

        try:
            # Enable fake quantization for all modules
            for module in model.modules():
                if hasattr(module, "fake_quant_enabled"):
                    module.fake_quant_enabled = True
                if hasattr(module, "enable_fake_quant"):
                    module.enable_fake_quant()

            self.fake_quant_enabled = True
            logger.debug_rank0("QAT: Enabled fake quantization")

        except Exception as e:
            logger.warning_rank0(f"Failed to enable fake quantization: {e}")

    def _disable_fake_quantization(self, model):
        """Disable fake quantization in the model."""
        if model is None:
            return

        try:
            # Disable fake quantization for all modules
            for module in model.modules():
                if hasattr(module, "fake_quant_enabled"):
                    module.fake_quant_enabled = False
                if hasattr(module, "disable_fake_quant"):
                    module.disable_fake_quant()

            self.fake_quant_enabled = False
            logger.debug_rank0("QAT: Disabled fake quantization")

        except Exception as e:
            logger.warning_rank0(f"Failed to disable fake quantization: {e}")


def get_qat_callback(model_args: "ModelArguments") -> QATCallback:
    """Get QAT callback if QAT is enabled."""
    if model_args.enable_qat:
        return QATCallback(model_args)
    return None
