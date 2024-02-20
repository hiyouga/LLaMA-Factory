from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import DataCollatorForSeq2Seq


@dataclass
class DPODataCollatorWithPadding(DataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.

    This class extends `DataCollatorForSeq2Seq` and provides padding functionality for pairwise data.
    """

    def _pad_labels(self, batch: torch.Tensor, positions: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Pad the labels in the batch to match the longest sequence.

        Parameters
        ----------
        batch : torch.Tensor
            The batch of input features.

        positions : List[Tuple[int, int]]
            List of tuples representing the positions of prompt and answer lengths in each feature.

        Returns
        -------
        padded_labels : torch.Tensor
            Tensor with padded labels.
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
        """
        Pad batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.

        Parameters
        ----------
        features : Sequence[Dict[str, Any]]
            List of dictionaries containing input features.

        Returns
        -------
        batch : Dict[str, torch.Tensor]
            Dictionary containing the batched data.
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

        batch = self.tokenizer.pad(
            concatenated_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = self._pad_labels(batch["input_ids"], label_positions)
        return batch
