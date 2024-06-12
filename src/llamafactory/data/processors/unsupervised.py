from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras.logging import get_logger
from ..data_utils import Role
from .processor_utils import get_paligemma_token_type_ids, get_pixel_values


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
    data_args: "DataArguments",
) -> Tuple[List[int], List[int]]:
    if processor is not None and not hasattr(processor, "image_seq_length"):  # llava-like models
        prompt[0]["content"] = template.image_token + prompt[0]["content"]

    if len(response) == 1:
        messages = prompt + response
    else:
        messages = prompt + [{"role": Role.ASSISTANT.value, "content": ""}]

    input_ids, labels = template.encode_oneturn(
        tokenizer, messages, system, tools, data_args.cutoff_len, data_args.reserved_label_len
    )
    if template.efficient_eos:
        labels += [tokenizer.eos_token_id]

    if processor is not None and hasattr(processor, "image_seq_length"):  # paligemma models
        image_token_id = tokenizer.convert_tokens_to_ids(template.image_token)
        input_ids = [image_token_id] * getattr(processor, "image_seq_length") + input_ids

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
    if processor is not None:
        model_inputs["pixel_values"] = []
        if hasattr(processor, "image_seq_length"):  # paligemma models
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
            data_args=data_args,
        )
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
        if processor is not None:
            model_inputs["pixel_values"].append(get_pixel_values(examples["images"][i], processor))
            if hasattr(processor, "image_seq_length"):  # paligemma models
                model_inputs["token_type_ids"].append(get_paligemma_token_type_ids(len(input_ids), processor))

    return model_inputs


def print_unsupervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
