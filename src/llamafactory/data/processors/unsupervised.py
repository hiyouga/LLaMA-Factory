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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras.logging import get_logger
from ..data_utils import Role
from .processor_utils import get_paligemma_token_type_ids, get_pixel_values, infer_seqlen, get_pixel_values_videos

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
    exists_images: bool,
    exists_videos: bool,
    cutoff_len: int,
) -> Tuple[List[int], List[int]]:
    if processor is not None:
        processor_class = type(processor).__name__
        if processor_class != 'PaliGemmaProcessor':
            if template.image_token not in prompt[0]["content"] and exists_videos:
                prompt[0]["content"] = template.video_token + prompt[0]["content"]
            if template.video_token not in prompt[0]["content"] and exists_images:
                prompt[0]["content"] = template.image_token + prompt[0]["content"]

    if len(response) == 1:
        messages = prompt + response
    else:
        messages = prompt + [{"role": Role.ASSISTANT.value, "content": ""}]

    if processor is not None and processor_class == 'Idefics2Processor':
        fake_image_token = processor.fake_image_token.content
        image_str = f"{fake_image_token}{template.image_token * processor.image_seq_len}{fake_image_token}"
        image_str = image_str * 5
        for j in range(len(messages)):
            content = messages[j]['content']
            content = content.replace(template.image_token, image_str)
            content = content.replace(f"{fake_image_token}{fake_image_token}", f"{fake_image_token}")
            messages[j]['content'] = content

    input_ids, labels = template.encode_oneturn(tokenizer, messages, system, tools)
    if template.efficient_eos:
        labels += [tokenizer.eos_token_id]

    if processor is not None and processor_class == 'PaliGemmaProcessor':  # paligemma models
        image_token_id = tokenizer.convert_tokens_to_ids(template.image_token)
        input_ids = [image_token_id] * getattr(processor, "image_seq_length") + input_ids

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
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X` and labels with format `Y <eos>`
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    image_keys = template.image_data_key
    video_keys = template.video_data_key
    processor_class = None

    if processor is not None:
        if len(examples["images"][0]):
            for image_key in image_keys:
                model_inputs[image_key] = []

        if len(examples["videos"][0]):
            for video_key in video_keys:
                model_inputs[video_key] = []

        processor_class = type(processor).__name__

        if processor_class == 'PaliGemmaProcessor':  # paligemma models
            model_inputs["token_type_ids"] = []

    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        input_ids, labels = _encode_unsupervised_example(
            prompt=examples["prompt"][i],
            response=examples["response"][i],
            system=examples["system"][i],
            tools=examples["tools"][i],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            exists_images=len(examples["images"][i]) > 0,
            exists_videos=len(examples['videos'][i]) > 0,
            cutoff_len=data_args.cutoff_len,
        )
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)

        if processor is not None:
            if len(examples["images"][i]):
                image_data = get_pixel_values(examples["images"][i], processor, image_keys)
                for image_key in image_keys:
                    image_value = image_data[image_key]
                    if image_value.shape[0] == 1:
                        model_inputs[image_key].append(image_value[0])
                if processor_class == 'PaliGemmaProcessor':  # paligemma models
                    model_inputs["token_type_ids"].append(get_paligemma_token_type_ids(len(input_ids), processor))

            if len(examples["videos"][i]):
                video_data = get_pixel_values_videos(examples["videos"][i], processor, video_keys)
                for video_key in video_keys:
                    video_value = video_data[video_key]
                    if video_value.shape[0] == 1:
                        model_inputs[video_key].append(video_value[0])

    return model_inputs


def print_unsupervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
