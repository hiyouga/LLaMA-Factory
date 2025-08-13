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

from ...extras.constants import IGNORE_INDEX
from ..data_utils import preprocess_sp_dataset
from .processor_utils import SequenceParallelProcessor

class SequenceParallelPaddingProcessor(SequenceParallelProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        r"""
        Pad sequence
        """
        max_length = self.data_args.cutoff_len
        input_pad_token_id = self.tokenizer.pad_token_id
        assert self.data_args.ignore_pad_token_for_loss
        label_pad_token_id = IGNORE_INDEX if self.data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id

        for k, v in examples.items():
            if k.endswith("input_ids"):
                pad_token_id = input_pad_token_id
            elif k.endswith("labels"):
                pad_token_id = label_pad_token_id
                # shift labels here
                for i in range(len(v)):
                    v[i] = v[i][1:]
            elif k.endswith("attention_mask"):
                pad_token_id = 0
            elif k.endswith("position_ids"):
                pad_token_id = max_length - 1  # pad the max position id
            elif k == "images" or k == "videos" or k == "audios":
                pad_token_id = -1
                continue  # TODO: haven't tested multi-modal yet
            else:
                raise NotImplementedError(f"Unexpected dataset key: {k}")
            for i in range(len(v)):
                v[i].extend([pad_token_id] * (max_length - len(v[i])))
            examples[k] = v

        return examples


class SequenceParallelSplitProcessor(SequenceParallelProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        r"""
        split dataset
        """
        for k, v in examples.items():
            chunks = list()
            for row in v:
                if row is None:
                    chunks.extend([None] * self.model_args.sequence_parallel_size)
                else:
                    chunks.extend(preprocess_sp_dataset(
                        row, self.model_args.sequence_parallel_size, self.model_args.sequence_parallel_mode
                    ))
            examples[k] = chunks
        return examples
