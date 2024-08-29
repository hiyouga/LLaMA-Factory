# Copyright 2024 the LlamaFactory team.
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

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras.logging import get_logger
from ..data_utils import Role
from .processor_utils import infer_seqlen


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..template import Template


logger = get_logger(__name__)


def _encode_unsupervised_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
) -> Tuple[List[int], List[int]]:
    if len(response) == 1:
        messages = prompt + response
    else:
        messages = prompt + [{"role": Role.ASSISTANT.value, "content": ""}]

    input_ids, labels = template.encode_oneturn(tokenizer, messages, system, tools)
    if template.efficient_eos:
        labels += [tokenizer.eos_token_id]

    input_ids, _ = template.mm_plugin.process_token_ids(input_ids, None, tokenizer, processor)
    source_len, target_len = infer_seqlen(len(input_ids), len(labels), cutoff_len)
    input_ids = input_ids[:source_len]
    labels = labels[:target_len]
    return input_ids, labels


def preprocess_unsupervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # build inputs with format `<bos> X` and labels with format `Y <eos>`
    model_inputs = defaultdict(list)
    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        prompt = template.mm_plugin.process_messages(examples["prompt"][i], examples["images"][i], processor)
        input_ids, labels = _encode_unsupervised_example(
            prompt=prompt,
            response=examples["response"][i],
            system=examples["system"][i],
            tools=examples["tools"][i],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len,
        )
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
        template.mm_plugin.process_model_inputs(
            model_inputs=model_inputs,
            images=examples["images"][i],
            feature_seqlens={"token_type_ids": len(input_ids)},
            processor=processor,
        )

    return model_inputs


def print_unsupervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
