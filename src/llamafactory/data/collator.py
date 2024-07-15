from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from llamafactory.easy_context import prepare_seq_parallel_sft_inputs

@dataclass
class PairwiseDataCollatorWithPadding(DataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        for key in ("chosen", "rejected"):
            for feature in features:
                target_feature = {
                    "input_ids": feature["{}_input_ids".format(key)],
                    "attention_mask": feature["{}_attention_mask".format(key)],
                    "labels": feature["{}_labels".format(key)],
                }
                if "pixel_values" in feature:
                    target_feature["pixel_values"] = feature["pixel_values"]

                if "{}_token_type_ids".format(key) in feature:
                    target_feature["token_type_ids"] = feature["{}_token_type_ids".format(key)]

                concatenated_features.append(target_feature)

        return super().__call__(concatenated_features)


@dataclass
class KTODataCollatorWithPadding(DataCollatorForSeq2Seq):
    r"""
    Data collator for KTO data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        target_features = []
        kl_features = []
        kto_tags = []
        for feature in features:
            target_feature = {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "labels": feature["labels"],
            }
            kl_feature = {
                "input_ids": feature["kl_input_ids"],
                "attention_mask": feature["kl_attention_mask"],
                "labels": feature["kl_labels"],
            }
            if "pixel_values" in feature:
                target_feature["pixel_values"] = feature["pixel_values"]

            if "token_type_ids" in feature:
                target_feature["token_type_ids"] = feature["token_type_ids"]
                kl_feature["token_type_ids"] = feature["kl_token_type_ids"]

            target_features.append(target_feature)
            kl_features.append(kl_feature)
            kto_tags.append(feature["kto_tags"])

        batch = super().__call__(target_features)
        kl_batch = super().__call__(kl_features)
        batch["kl_input_ids"] = kl_batch["input_ids"]
        batch["kl_attention_mask"] = kl_batch["attention_mask"]
        batch["kl_labels"] = kl_batch["labels"]
        if "token_type_ids" in batch:
            batch["kl_token_type_ids"] = kl_batch["token_type_ids"]

        batch["kto_tags"] = torch.tensor(kto_tags)
        return batch

@dataclass
class SeqParallelDataCollator(DataCollatorForSeq2Seq):
    r"""
    Data collator for sequence parallel in supervised finetune(sft) stage.
    """
    seq_algo: str = "data_parallel",
    sp_size: int = -1
    rank: int = 0
    world_size: int = 8
    device: Optional[Any] = None

    def __call__(self, features: Sequence[Dict[str, Any]], return_tensors=None) -> Dict[str, torch.Tensor]:
        batch = super().__call__(features, return_tensors)
        if self.seq_algo == "data_parallel":
            return batch
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        world_size = self.world_size
        sp_rank = self.rank
        if self.sp_size != -1:
            dp_rank = self.rank // self.sp_size
            sp_rank = self.rank % self.sp_size
            world_size = self.sp_size
            bs = len(input_ids)
            dp_size = self.world_size // self.sp_size
            group_bs = bs // dp_size
            input_ids = input_ids[dp_rank * group_bs: (dp_rank + 1) * group_bs]
            attention_mask = attention_mask[dp_rank * group_bs: (dp_rank + 1) * group_bs]
            labels = labels[dp_rank * group_bs: (dp_rank + 1) * group_bs]
        batch = prepare_seq_parallel_sft_inputs(self.seq_algo,
                                                input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                position_ids=None,
                                                labels=labels,
                                                rank=sp_rank,
                                                world_size=world_size,
                                                device=self.device)
        return batch


@dataclass
class SeqParallelDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    r"""
    Data collator for sequence parallel in pretrain(pt) stage.
    Reuse the sequence parallel distributing function for sft stage.
    """
    seq_algo: str = "data_parallel"
    sp_size: int = -1
    rank: int = 0
    world_size: int = 8
    device: Optional[Any] = None

    def __call__(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().__call__(examples)
        if self.seq_algo == "data_parallel":
            return batch
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        world_size = self.world_size
        sp_rank = self.rank
        if self.sp_size != -1:
            dp_rank = self.rank // self.sp_size
            sp_rank = self.rank % self.sp_size
            world_size = self.sp_size
            bs = len(input_ids)
            dp_size = self.world_size // self.sp_size
            group_bs = bs // dp_size
            input_ids = input_ids[dp_rank * group_bs: (dp_rank + 1) * group_bs]
            attention_mask = attention_mask[dp_rank * group_bs: (dp_rank + 1) * group_bs]
            labels = labels[dp_rank * group_bs: (dp_rank + 1) * group_bs]
        batch = prepare_seq_parallel_sft_inputs(self.seq_algo,
                                                input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                position_ids=None,
                                                labels=labels,
                                                rank=sp_rank,
                                                world_size=world_size,
                                                device=self.device)
        return batch
