from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import DataCollatorForSeq2Seq


@dataclass
class PairwiseDataCollatorWithPadding(DataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """

    def _pad_labels(self, batch: torch.Tensor, positions: List[Tuple[int, int]]) -> torch.Tensor:
        r"""
        Masks out the input ids except for the responses.
        """
        padded_labels = []
        for feature, (prompt_len, answer_len) in zip(batch, positions):
            if self.tokenizer.padding_side == "left":
                start, end = feature.size(0) - answer_len, feature.size(0)
            else:
                start, end = prompt_len, prompt_len + answer_len
            padded_tensor = self.label_pad_token_id * torch.ones_like(feature)
            padded_tensor[start:end] = feature[start:end]
            padded_labels.append(padded_tensor)
        return torch.stack(padded_labels, dim=0).contiguous()  # in contiguous memory

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        label_positions = []
        for key in ("chosen_ids", "rejected_ids"):
            for feature in features:
                prompt_len, answer_len = len(feature["prompt_ids"]), len(feature[key])
                concatenated_features.append(
                    {
                        "input_ids": feature["prompt_ids"] + feature[key],
                        "attention_mask": [1] * (prompt_len + answer_len),
                    }
                )
                label_positions.append((prompt_len, answer_len))

        batch = super().__call__(concatenated_features)
        batch["labels"] = self._pad_labels(batch["input_ids"], label_positions)
        return batch

@dataclass
class KTODataCollatorWithPadding(DataCollatorForSeq2Seq):
    r"""
    Data collator for KTO data.
    """
    def __call__(self, features, return_tensors=None):
        concatenated_features = []
        kl_concatenated_features = []
        tags = []
        for feature in features:
            concatenated_features.append(
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                    "labels": feature["labels"],
                }
            )
            kl_concatenated_features.append(
                {
                    "input_ids": feature["kl_input_ids"],
                    "attention_mask": feature["kl_attention_mask"],
                    "labels": feature["kl_labels"],
                }
            )
            tags.append(feature["tag"])
        batch = super().__call__(concatenated_features)
        kl_batch = super().__call__(kl_concatenated_features)
        batch["KL_completion_input_ids"] = kl_batch["input_ids"]
        batch["KL_completion_attention_mask"] = kl_batch["attention_mask"]
        batch["kl_labels"] = kl_batch["labels"]
        batch["tag"] = torch.tensor(tags)
        return batch