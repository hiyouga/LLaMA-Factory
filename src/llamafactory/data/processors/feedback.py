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


def preprocess_feedback_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # create unrelated input-output pairs for estimating the KL term by flipping the matched pairs
    kl_response = examples["response"][::-1]
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "kl_input_ids": [],
        "kl_attention_mask": [],
        "kl_labels": [],
        "kto_tags": [],
    }
    if processor is not None:
        model_inputs["pixel_values"] = []
        if hasattr(processor, "image_seq_length"):  # paligemma models
            model_inputs["token_type_ids"] = []
            model_inputs["kl_token_type_ids"] = []

    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) < 2:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        if processor is not None and not hasattr(processor, "image_seq_length"):  # llava-like models
            examples["prompt"][i][0]["content"] = template.image_token + examples["prompt"][i][0]["content"]

        if examples["response"][i][0]["content"]:  # desired example
            kto_tag = True
            messages = examples["prompt"][i] + [examples["response"][i][0]]
        else:  # undesired example
            kto_tag = False
            messages = examples["prompt"][i] + [examples["response"][i][1]]

        if kl_response[i][0]["content"]:
            kl_messages = examples["prompt"][i] + [kl_response[i][0]]
        else:
            kl_messages = examples["prompt"][i] + [kl_response[i][1]]

        prompt_ids, response_ids = template.encode_oneturn(
            tokenizer,
            messages,
            examples["system"][i],
            examples["tools"][i],
            data_args.cutoff_len,
            data_args.reserved_label_len,
        )
        _, kl_response_ids = template.encode_oneturn(
            tokenizer,
            kl_messages,
            examples["system"][i],
            examples["tools"][i],
            data_args.cutoff_len,
            data_args.reserved_label_len,
        )

        if template.efficient_eos:
            response_ids += [tokenizer.eos_token_id]
            kl_response_ids += [tokenizer.eos_token_id]

        if processor is not None and hasattr(processor, "image_seq_length"):  # paligemma models
            image_token_id = tokenizer.convert_tokens_to_ids(template.image_token)
            prompt_ids = [image_token_id] * getattr(processor, "image_seq_length") + prompt_ids

        input_ids = prompt_ids + response_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
        kl_input_ids = prompt_ids + kl_response_ids
        kl_labels = [IGNORE_INDEX] * len(prompt_ids) + kl_response_ids
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
        model_inputs["kl_input_ids"].append(kl_input_ids)
        model_inputs["kl_attention_mask"].append([1] * len(kl_input_ids))
        model_inputs["kl_labels"].append(kl_labels)
        model_inputs["kto_tags"].append(kto_tag)
        if processor is not None:
            model_inputs["pixel_values"].append(get_pixel_values(examples["images"][i], processor))
            if hasattr(processor, "image_seq_length"):  # paligemma models
                model_inputs["token_type_ids"].append(get_paligemma_token_type_ids(len(input_ids), processor))
                model_inputs["kl_token_type_ids"].append(get_paligemma_token_type_ids(len(kl_input_ids), processor))

    desirable_num = sum([1 for tag in model_inputs["kto_tags"] if tag])
    undesirable_num = len(model_inputs["kto_tags"]) - desirable_num
    if desirable_num == 0 or undesirable_num == 0:
        logger.warning("Your dataset only has one preference type.")

    return model_inputs
