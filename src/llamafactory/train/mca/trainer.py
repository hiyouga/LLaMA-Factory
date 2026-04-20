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

from typing import Any

import torch.nn.functional as F
from mcore_adapter.trainer import McaTrainer
from torch import Tensor
from transformers import PreTrainedTokenizerBase
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX


class CustomMcaTrainer(McaTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override
    def _pad_batched_inputs(self, inputs: dict[str, Tensor | Any], seq_length: int):
        r"""Override to avoid padding error when handling 3d posids."""
        padding_inputs = {
            k: v.tolist() if v is not None and isinstance(v, Tensor) else v
            for k, v in inputs.items()
            if k in self._language_input_names
        }

        position_ids_3d = None
        if isinstance(inputs.get("position_ids"), Tensor) and inputs["position_ids"].dim() == 3:
            position_ids_3d = inputs["position_ids"]
            padding_inputs.pop("position_ids", None)

        if "labels" in padding_inputs:
            padding_inputs["labels"] = [
                labels + [IGNORE_INDEX] * (seq_length - len(labels)) for labels in padding_inputs["labels"]
            ]
        tokenizer = (
            self.processing_class
            if isinstance(self.processing_class, PreTrainedTokenizerBase)
            else getattr(self.processing_class, "tokenizer", self.processing_class)
        )
        padding_side = getattr(tokenizer, "padding_side", "right")
        padding_inputs = tokenizer.pad(
            padding_inputs,
            padding="max_length",
            max_length=seq_length,
            return_tensors="pt",
        ).to(self.args.device)
        inputs.update(padding_inputs)

        if position_ids_3d is not None:
            current_seq_len = position_ids_3d.size(-1)
            if current_seq_len < seq_length:
                pad_len = seq_length - current_seq_len
                if padding_side == "left":
                    position_ids_3d = F.pad(position_ids_3d, (pad_len, 0), value=0)
                else:
                    position_ids_3d = F.pad(position_ids_3d, (0, pad_len), value=0)

            inputs["position_ids"] = position_ids_3d.to(self.args.device)

        return inputs
