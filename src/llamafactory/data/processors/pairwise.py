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


def preprocess_pairwise_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
    model_inputs = {
        "chosen_input_ids": [],
        "chosen_attention_mask": [],
        "chosen_labels": [],
        "rejected_input_ids": [],
        "rejected_attention_mask": [],
        "rejected_labels": [],
    }
    if processor is not None:
        model_inputs["pixel_values"] = []
        if hasattr(processor, "image_seq_length"):  # paligemma models
            model_inputs["chosen_token_type_ids"] = []
            model_inputs["rejected_token_type_ids"] = []

    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) < 2:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        if processor is not None and not hasattr(processor, "image_seq_length"):  # llava-like models
            examples["prompt"][i][0]["content"] = template.image_token + examples["prompt"][i][0]["content"]

        chosen_messages = examples["prompt"][i] + [examples["response"][i][0]]
        rejected_messages = examples["prompt"][i] + [examples["response"][i][1]]
        prompt_ids, chosen_ids = template.encode_oneturn(
            tokenizer,
            chosen_messages,
            examples["system"][i],
            examples["tools"][i],
            data_args.cutoff_len,
            data_args.reserved_label_len,
        )
        _, rejected_ids = template.encode_oneturn(
            tokenizer,
            rejected_messages,
            examples["system"][i],
            examples["tools"][i],
            data_args.cutoff_len,
            data_args.reserved_label_len,
        )

        if template.efficient_eos:
            chosen_ids += [tokenizer.eos_token_id]
            rejected_ids += [tokenizer.eos_token_id]

        if processor is not None and hasattr(processor, "image_seq_length"):  # paligemma models
            image_token_id = tokenizer.convert_tokens_to_ids(template.image_token)
            prompt_ids = [image_token_id] * getattr(processor, "image_seq_length") + prompt_ids

        chosen_input_ids = prompt_ids + chosen_ids
        chosen_labels = [IGNORE_INDEX] * len(prompt_ids) + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        rejected_labels = [IGNORE_INDEX] * len(prompt_ids) + rejected_ids
        model_inputs["chosen_input_ids"].append(chosen_input_ids)
        model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
        model_inputs["chosen_labels"].append(chosen_labels)
        model_inputs["rejected_input_ids"].append(rejected_input_ids)
        model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
        model_inputs["rejected_labels"].append(rejected_labels)
        if processor is not None:
            model_inputs["pixel_values"].append(get_pixel_values(examples["images"][i], processor))
            if hasattr(processor, "image_seq_length"):  # paligemma models
                model_inputs["chosen_token_type_ids"].append(
                    get_paligemma_token_type_ids(len(chosen_input_ids), processor)
                )
                model_inputs["rejected_token_type_ids"].append(
                    get_paligemma_token_type_ids(len(rejected_input_ids), processor)
                )

    return model_inputs


def print_pairwise_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels"]))
    valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels"]))
    print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
    print("chosen_inputs:\n{}".format(tokenizer.decode(example["chosen_input_ids"], skip_special_tokens=False)))
    print("chosen_label_ids:\n{}".format(example["chosen_labels"]))
    print("chosen_labels:\n{}".format(tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)))
    print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
    print("rejected_inputs:\n{}".format(tokenizer.decode(example["rejected_input_ids"], skip_special_tokens=False)))
    print("rejected_label_ids:\n{}".format(example["rejected_labels"]))
    print("rejected_labels:\n{}".format(tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)))
