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

import torch
import torch.nn.functional as F

from ..accelerator.interface import DistributedInterface
from ..config import InputArgument, get_args
from ..core.base_trainer import BaseTrainer
from ..core.data_engine import DataEngine
from ..core.model_engine import ModelEngine
from ..plugins.model_plugins.chunk_loss import ChunkCrossEntropyHandler
from ..utils.types import BatchInput, HFModel, ModelOutput, Tensor


class SFTTrainer(BaseTrainer):
    def _build_chunk_loss_handler(self, model: HFModel) -> ChunkCrossEntropyHandler:
        return ChunkCrossEntropyHandler(
            model,
            chunk_size=self.args.chunk_loss_size,
            token_budget=self.args.chunk_loss_token_budget,
        )

    def _prepare_model_inputs(self, batch: BatchInput) -> dict[str, Tensor]:
        model_inputs = {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor) and key not in ("labels", "loss_weights")
        }
        # Let mRoPE models build their own multimodal 3D position ids.
        if self._uses_mrope:
            model_inputs.pop("position_ids", None)

        return model_inputs

    def compute_log_probs(self, model: HFModel, model_inputs: dict[str, Tensor], target_labels: Tensor) -> Tensor:
        """Compute log probs.

        log_probs: Tensor of shape (batch_size, seq_len - 1)
        """
        outputs: ModelOutput = model(**model_inputs)
        logits = outputs.logits.float()
        flat_labels = target_labels.view(-1)
        shift_logits = logits[..., :-1, :].contiguous().view(flat_labels.size(0), -1)
        return -F.cross_entropy(shift_logits, flat_labels, reduction="none").view_as(target_labels)

    def compute_loss(self, batch: BatchInput) -> Tensor:
        if self.cp_size > 1:
            from ..plugins.model_plugins.parallelization.sequence_parallel import SFTSequenceParallelLossPlugin

            return SFTSequenceParallelLossPlugin("sequence_parallel_loss")(
                self.model, batch, local_loss_fn=self._chunk_loss_handler
            )

        model_inputs = self._prepare_model_inputs(batch)
        target_labels = batch["labels"].to(self.device, non_blocking=True)[..., 1:].contiguous()
        target_weights = batch["loss_weights"].to(self.device, non_blocking=True)[..., 1:]
        denominator = target_weights.float().sum() + 1e-6

        if self._chunk_loss_handler is not None:
            return self._chunk_loss_handler(self.model, model_inputs, target_labels, target_weights, denominator)

        log_probs = self.compute_log_probs(self.model, model_inputs, target_labels)
        return (-log_probs * target_weights).sum() / denominator


def run_sft(args: InputArgument = None):
    model_args, data_args, training_args, _ = get_args(args)
    DistributedInterface(training_args)
    train_dataset = DataEngine(data_args.train_dataset)
    model_engine = ModelEngine(model_args, is_train=True)
    trainer = SFTTrainer(
        args=training_args,
        model=model_engine.model,
        renderer=model_engine.renderer,
        train_dataset=train_dataset,
    )
    trainer.fit()
    trainer.save_model()
    DistributedInterface().destroy()


if __name__ == "__main__":
    """
    python -m llamafactory.v1.trainers.sft_trainer --model Qwen/Qwen3-0.6B --train_dataset data/v1_sft_demo.yaml
    """
    run_sft()
