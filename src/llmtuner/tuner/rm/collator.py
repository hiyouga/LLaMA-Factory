import torch
from dataclasses import dataclass
from typing import Any, Dict, Sequence
from transformers import DataCollatorWithPadding


@dataclass
class PairwiseDataCollatorWithPadding(DataCollatorWithPadding):
    r"""
    Data collator for pairwise data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        features = [
            {
                "input_ids": feature["prompt_ids"] + feature[key],
                "attention_mask": [1] * (len(feature["prompt_ids"]) + len(feature[key]))
            }
            for key in ("chosen_ids", "rejected_ids") for feature in features
        ]
        return super().__call__(features)
