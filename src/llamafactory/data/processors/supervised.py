import itertools
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from .mm_utils import get_paligemma_token_type_ids, get_pixel_values


if TYPE_CHECKING:
    from transformers import ProcessorMixin
    from transformers.tokenization_utils import PreTrainedTokenizer

    from ...hparams import DataArguments
    from ..template import Template


logger = get_logger(__name__)


def binary_search_for_fit(numbers, capacity):
    """
    Perform binary search to find the largest number that fits into the knapsack with the given capacity.
    """
    left, right = 0, len(numbers) - 1
    result = -1  # If no number fits, return -1

    while left <= right:
        mid = (left + right) // 2
        if numbers[mid] <= capacity:
            result = mid
            left = mid + 1
        else:
            right = mid - 1

    return result


def efficient_greedy_knapsack(numbers, capacity):
    """
    An efficient greedy algorithm with binary search for the knapsack problem.
    """
    numbers.sort()  # Sort numbers in ascending order for binary search
    knapsacks = []

    while numbers:
        current_knapsack = []
        remaining_capacity = capacity

        while True:
            index = binary_search_for_fit(numbers, remaining_capacity)
            if index == -1:
                break  # No more numbers fit in this knapsack

            # Add the found number to the knapsack and update the remaining capacity
            current_knapsack.append(numbers[index])
            remaining_capacity -= numbers[index]

            # Remove the number from the list
            numbers.pop(index)

        knapsacks.append(current_knapsack)

    return knapsacks


def preprocess_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    if processor is not None:
        model_inputs["pixel_values"] = []
        if hasattr(processor, "image_seq_length"):  # paligemma models
            model_inputs["token_type_ids"] = []

    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        if processor is not None and not hasattr(processor, "image_seq_length"):  # llava-like models
            examples["prompt"][i][0]["content"] = template.image_token + examples["prompt"][i][0]["content"]

        messages = examples["prompt"][i] + examples["response"][i]
        input_ids, labels = [], []

        if processor is not None and hasattr(processor, "image_seq_length"):  # paligemma models
            image_token_id = tokenizer.convert_tokens_to_ids(template.image_token)
            input_ids += [image_token_id] * getattr(processor, "image_seq_length")
            labels += [IGNORE_INDEX] * getattr(processor, "image_seq_length")

        for turn_idx, (source_ids, target_ids) in enumerate(
            template.encode_multiturn(
                tokenizer,
                messages,
                examples["system"][i],
                examples["tools"][i],
                data_args.cutoff_len,
                data_args.reserved_label_len,
            )
        ):
            if data_args.train_on_prompt:
                source_mask = source_ids
            elif turn_idx != 0 and template.efficient_eos:
                source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
            else:
                source_mask = [IGNORE_INDEX] * len(source_ids)

            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

        if template.efficient_eos:
            input_ids += [tokenizer.eos_token_id]
            labels += [tokenizer.eos_token_id]

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
        if processor is not None:
            model_inputs["pixel_values"].append(get_pixel_values(examples["images"][i], processor))
            if hasattr(processor, "image_seq_length"):  # paligemma models
                model_inputs["token_type_ids"].append(get_paligemma_token_type_ids(len(input_ids), processor))

    return model_inputs


def preprocess_packed_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
    # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    input_ids, labels = [], []
    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        messages = examples["prompt"][i] + examples["response"][i]
        for source_ids, target_ids in template.encode_multiturn(
            tokenizer, messages, examples["system"][i], examples["tools"][i]
        ):
            if data_args.train_on_prompt:
                source_mask = source_ids
            elif len(input_ids) != 0 and template.efficient_eos:
                source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
            else:
                source_mask = [IGNORE_INDEX] * len(source_ids)

            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

    if template.efficient_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

    # prepare for packing
    lengths = []
    length2examples_idx = defaultdict(list)
    for idx, example in enumerate(input_ids):
        length = len(example)
        if length > data_args.cutoff_len:
            logger.warning("Dropped example with length {} > cutoff_len {}".format(length, data_args.cutoff_len))
            continue
        lengths.append(length)
        length2examples_idx[length].append(idx)

    knapsacks = efficient_greedy_knapsack(lengths, data_args.cutoff_len)

    for knapsack in knapsacks:
        packed_input_ids = []
        packed_labels = []

        total_length = 0
        for length in knapsack:
            total_length += length
            idx = length2examples_idx[length].pop()
            packed_input_ids.append(input_ids[idx])
            packed_labels.append(labels[idx])

        # padding to cutoff_len
        if total_length < data_args.cutoff_len:
            pad_length = data_args.cutoff_len - total_length
            packed_input_ids.append([tokenizer.eos_token_id] * pad_length)
            packed_labels.append([IGNORE_INDEX] * pad_length)
        elif total_length == data_args.cutoff_len:
            pad_length = 0
        else:
            logger.warning(
                "Dropped packed example with total length {} > cutoff_len {}".format(
                    total_length, data_args.cutoff_len
                )
            )
            continue

        # concat all
        model_inputs["input_ids"].append(list(itertools.chain(*packed_input_ids)))

        model_inputs["labels"].append(list(itertools.chain(*packed_labels)))
        model_inputs["attention_mask"].append([1] * total_length + [0] * pad_length)

    return model_inputs


def print_supervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    print("label_ids:\n{}".format(example["labels"]))
    print("labels:\n{}".format(tokenizer.decode(valid_labels, skip_special_tokens=False)))
