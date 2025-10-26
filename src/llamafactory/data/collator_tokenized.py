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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ..extras.constants import IGNORE_INDEX
from transformers import DataCollatorForSeq2Seq


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


def _resolve_pad_token_id(tokenizer: "PreTrainedTokenizer", model: "PreTrainedModel") -> int:
    r"""Resolve the padding token ID from tokenizer or model config."""
    pad_id = getattr(getattr(model, "config", None), "pad_token_id", None)
    if pad_id is None and tokenizer is not None:
        pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(getattr(model, "config", None), "eos_token_id", None)
    return 0 if pad_id is None else int(pad_id)


@dataclass
class TokenizedIdsCollator(DataCollatorForSeq2Seq):
    r"""Collator for pre-tokenized LM data.

    Expects features containing `input_ids` and optionally `attention_mask`.
    Pads to batch max length with `pad_token_id`, generates labels and masks missing fields when needed.
    """

    strict: bool = True

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        pad_id = _resolve_pad_token_id(self.tokenizer, self.model)

        # Validate and compute max length
        max_len = 0
        for f in features:
            if "input_ids" not in f or not isinstance(f["input_ids"], list):
                if self.strict:
                    raise ValueError("Each feature must contain list[int] `input_ids`.")
                else:
                    f["input_ids"] = f.get("input_ids", []) or []
            max_len = max(max_len, len(f["input_ids"]))

        input_ids = []
        attention_mask = []
        labels = []
        for f in features:
            ids = f["input_ids"]
            pad_amt = max_len - len(ids)
            row_ids = ids + [pad_id] * pad_amt
            input_ids.append(row_ids)

            if "attention_mask" in f and isinstance(f["attention_mask"], list):
                if self.strict and len(f["attention_mask"]) != len(ids):
                    raise ValueError("attention_mask length must match input_ids length.")
                mask = f["attention_mask"] + [0] * pad_amt
            else:
                mask = [1] * len(ids) + [0] * pad_amt
            attention_mask.append(mask)

            row_labels = row_ids.copy()
            for i in range(len(ids), max_len):
                row_labels[i] = IGNORE_INDEX
            labels.append(row_labels)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return batch
