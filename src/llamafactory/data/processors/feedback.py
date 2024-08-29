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

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from .processor_utils import infer_seqlen


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..template import Template


logger = get_logger(__name__)


def _encode_feedback_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    kl_response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
) -> Tuple[List[int], List[int], List[int], List[int], bool]:
    if response[0]["content"]:  # desired example
        kto_tag = True
        messages = prompt + [response[0]]
    else:  # undesired example
        kto_tag = False
        messages = prompt + [response[1]]

    if kl_response[0]["content"]:
        kl_messages = prompt + [kl_response[0]]
    else:
        kl_messages = prompt + [kl_response[1]]

    prompt_ids, response_ids = template.encode_oneturn(tokenizer, messages, system, tools)
    kl_prompt_ids, kl_response_ids = template.encode_oneturn(tokenizer, kl_messages, system, tools)

    if template.efficient_eos:
        response_ids += [tokenizer.eos_token_id]
        kl_response_ids += [tokenizer.eos_token_id]

    prompt_ids, _ = template.mm_plugin.process_token_ids(prompt_ids, None, tokenizer, processor)
    kl_prompt_ids, _ = template.mm_plugin.process_token_ids(kl_prompt_ids, None, tokenizer, processor)

    source_len, target_len = infer_seqlen(len(prompt_ids), len(response_ids), cutoff_len)
    prompt_ids = prompt_ids[:source_len]
    response_ids = response_ids[:target_len]
    kl_source_len, kl_target_len = infer_seqlen(len(kl_prompt_ids), len(kl_response_ids), cutoff_len)
    kl_prompt_ids = kl_prompt_ids[:kl_source_len]
    kl_response_ids = kl_response_ids[:kl_target_len]

    input_ids = prompt_ids + response_ids
    labels = [IGNORE_INDEX] * source_len + response_ids
    kl_input_ids = kl_prompt_ids + kl_response_ids
    kl_labels = [IGNORE_INDEX] * kl_source_len + kl_response_ids

    return input_ids, labels, kl_input_ids, kl_labels, kto_tag


def preprocess_feedback_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # create unrelated input-output pairs for estimating the KL term by flipping the matched pairs
    kl_response = examples["response"][::-1]
    model_inputs = defaultdict(list)
    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) < 2:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        prompt = template.mm_plugin.process_messages(examples["prompt"][i], examples["images"][i], processor)
        input_ids, labels, kl_input_ids, kl_labels, kto_tag = _encode_feedback_example(
            prompt=prompt,
            response=examples["response"][i],
            kl_response=kl_response[i],
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
        model_inputs["kl_input_ids"].append(kl_input_ids)
        model_inputs["kl_attention_mask"].append([1] * len(kl_input_ids))
        model_inputs["kl_labels"].append(kl_labels)
        model_inputs["kto_tags"].append(kto_tag)
        template.mm_plugin.process_model_inputs(
            model_inputs=model_inputs,
            images=examples["images"][i],
            feature_seqlens={
                "token_type_ids": len(input_ids),
                "kl_token_type_ids": len(kl_input_ids),
            },
            processor=processor,
        )

    desirable_num = sum([1 for tag in model_inputs["kto_tags"] if tag])
    undesirable_num = len(model_inputs["kto_tags"]) - desirable_num
    if desirable_num == 0 or undesirable_num == 0:
        logger.warning("Your dataset only has one preference type.")

    return model_inputs
