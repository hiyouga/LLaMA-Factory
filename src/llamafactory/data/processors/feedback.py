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

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import infer_seqlen


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..mm_plugin import AudioInput, ImageInput, VideoInput
    from ..template import Template


logger = logging.get_logger(__name__)


def _encode_feedback_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    kl_response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    images: Sequence["ImageInput"],
    videos: Sequence["VideoInput"],
    audios: Sequence["AudioInput"],
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

    messages = template.mm_plugin.process_messages(messages, images, videos, audios, processor)
    kl_messages = template.mm_plugin.process_messages(kl_messages, images, videos, audios, processor)
    prompt_ids, response_ids = template.encode_oneturn(tokenizer, messages, system, tools)
    kl_prompt_ids, kl_response_ids = template.encode_oneturn(tokenizer, kl_messages, system, tools)

    if template.efficient_eos:
        response_ids += [tokenizer.eos_token_id]
        kl_response_ids += [tokenizer.eos_token_id]

    prompt_ids, _ = template.mm_plugin.process_token_ids(
        prompt_ids, None, images, videos, audios, tokenizer, processor
    )
    kl_prompt_ids, _ = template.mm_plugin.process_token_ids(
        kl_prompt_ids, None, images, videos, audios, tokenizer, processor
    )

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
    kl_response = examples["_response"][::-1]
    model_inputs = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        input_ids, labels, kl_input_ids, kl_labels, kto_tag = _encode_feedback_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            kl_response=kl_response[i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            audios=examples["_audios"][i] or [],
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
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])
        model_inputs["audios"].append(examples["_audios"][i])

    desirable_num = sum([1 for tag in model_inputs["kto_tags"] if tag])
    undesirable_num = len(model_inputs["kto_tags"]) - desirable_num
    if desirable_num == 0 or undesirable_num == 0:
        logger.warning_rank0("Your dataset only has one preference type.")

    return model_inputs
