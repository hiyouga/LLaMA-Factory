from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

from ..extras.constants import IGNORE_INDEX, IMAGE_TOKEN
from ..extras.logging import get_logger
from ..extras.packages import is_pillow_available
from .utils import Role


if is_pillow_available():
    from PIL import Image


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from PIL.Image import Image as ImageObject
    from transformers import ProcessorMixin, Seq2SeqTrainingArguments
    from transformers.image_processing_utils import BaseImageProcessor
    from transformers.tokenization_utils import PreTrainedTokenizer

    from ..hparams import DataArguments
    from .template import Template


logger = get_logger(__name__)


def _preprocess_visual_inputs(images: Sequence["ImageObject"], processor: "ProcessorMixin") -> "NDArray":
    # process visual inputs (currently only supports a single image)
    image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
    image = images[0] if len(images) != 0 else Image.new("RGB", (100, 100), (255, 255, 255))
    return image_processor(image, return_tensors="pt")["pixel_values"][0]


def preprocess_pretrain_dataset(
    examples: Dict[str, List[Any]], tokenizer: "PreTrainedTokenizer", data_args: "DataArguments"
) -> Dict[str, List[List[int]]]:
    # build grouped texts with format `X1 X2 X3 ...` if packing is enabled
    text_examples = [messages[0]["content"] + tokenizer.eos_token for messages in examples["prompt"]]

    if not data_args.packing:
        if data_args.template == "gemma":
            text_examples = [tokenizer.bos_token + example for example in text_examples]

        result = tokenizer(text_examples, add_special_tokens=False, max_length=data_args.cutoff_len)
    else:
        tokenized_examples = tokenizer(text_examples, add_special_tokens=False)
        concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        block_size = data_args.cutoff_len
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        if data_args.template == "gemma":
            for i in range(len(result["input_ids"])):
                result["input_ids"][i][0] = tokenizer.bos_token_id

    return result


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
        preprocess_visual_inputs = partial(_preprocess_visual_inputs, processor=processor)

    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        if processor is not None and not hasattr(processor, "image_seq_length"):  # llava case
            examples["prompt"][i][0]["content"] = IMAGE_TOKEN + examples["prompt"][i][0]["content"]

        messages = examples["prompt"][i] + examples["response"][i]
        input_ids, labels = [], []

        if processor is not None and hasattr(processor, "image_seq_length"):  # paligemma case
            image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
            input_ids += [image_token_id] * getattr(processor, "image_seq_length")
            labels += [image_token_id] * getattr(processor, "image_seq_length")

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
            model_inputs["pixel_values"].append(preprocess_visual_inputs(examples["images"][i]))

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

    total_length = len(input_ids)
    block_size = data_args.cutoff_len
    # we drop the small remainder, and if the total_length < block_size, we exclude this batch
    total_length = (total_length // block_size) * block_size
    # split by chunks of cutoff_len
    for i in range(0, total_length, block_size):
        if not all(label == IGNORE_INDEX for label in labels[i : i + block_size]):
            model_inputs["input_ids"].append(input_ids[i : i + block_size])
            model_inputs["attention_mask"].append([1] * block_size)
            model_inputs["labels"].append(labels[i : i + block_size])

    return model_inputs


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
        preprocess_visual_inputs = partial(_preprocess_visual_inputs, processor=processor)

    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        if processor is not None and not hasattr(processor, "image_seq_length"):  # llava case
            examples["prompt"][i][0]["content"] = IMAGE_TOKEN + examples["prompt"][i][0]["content"]

        if len(examples["response"][i]) == 1:
            messages = examples["prompt"][i] + examples["response"][i]
        else:
            messages = examples["prompt"][i] + [{"role": Role.ASSISTANT.value, "content": ""}]

        input_ids, labels = template.encode_oneturn(
            tokenizer,
            messages,
            examples["system"][i],
            examples["tools"][i],
            data_args.cutoff_len,
            data_args.reserved_label_len,
        )

        if template.efficient_eos:
            labels += [tokenizer.eos_token_id]

        if processor is not None and hasattr(processor, "image_seq_length"):  # paligemma case
            image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
            input_ids = [image_token_id] * getattr(processor, "image_seq_length") + input_ids

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
        if processor is not None:
            model_inputs["pixel_values"].append(preprocess_visual_inputs(examples["images"][i]))

    return model_inputs


def preprocess_pairwise_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
    model_inputs = {"prompt_ids": [], "chosen_ids": [], "rejected_ids": []}
    if processor is not None:
        model_inputs["pixel_values"] = []
        preprocess_visual_inputs = partial(_preprocess_visual_inputs, processor=processor)

    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) < 2:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        if processor is not None and not hasattr(processor, "image_seq_length"):  # llava case
            examples["prompt"][i][0]["content"] = IMAGE_TOKEN + examples["prompt"][i][0]["content"]

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

        if processor is not None and hasattr(processor, "image_seq_length"):  # paligemma case
            image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
            prompt_ids = [image_token_id] * getattr(processor, "image_seq_length") + prompt_ids

        model_inputs["prompt_ids"].append(prompt_ids)
        model_inputs["chosen_ids"].append(chosen_ids)
        model_inputs["rejected_ids"].append(rejected_ids)
        if processor is not None:
            model_inputs["pixel_values"].append(preprocess_visual_inputs(examples["images"][i]))

    return model_inputs


def preprocess_kto_dataset(
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
        preprocess_visual_inputs = partial(_preprocess_visual_inputs, processor=processor)

    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) < 2:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        if processor is not None and not hasattr(processor, "image_seq_length"):  # llava case
            examples["prompt"][i][0]["content"] = IMAGE_TOKEN + examples["prompt"][i][0]["content"]

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

        if processor is not None and hasattr(processor, "image_seq_length"):  # paligemma case
            image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
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
            model_inputs["pixel_values"].append(preprocess_visual_inputs(examples["images"][i]))

    desirable_num = sum([1 for tag in model_inputs["kto_tags"] if tag])
    undesirable_num = len(model_inputs["kto_tags"]) - desirable_num
    if desirable_num == 0 or undesirable_num == 0:
        logger.warning("Your dataset only has one preference type.")

    return model_inputs


def print_supervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    print("label_ids:\n{}".format(example["labels"]))
    print(
        "labels:\n{}".format(
            tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, example["labels"])), skip_special_tokens=False)
        )
    )


def print_pairwise_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    print("prompt_ids:\n{}".format(example["prompt_ids"]))
    print("prompt:\n{}".format(tokenizer.decode(example["prompt_ids"], skip_special_tokens=False)))
    print("chosen_ids:\n{}".format(example["chosen_ids"]))
    print("chosen:\n{}".format(tokenizer.decode(example["chosen_ids"], skip_special_tokens=False)))
    print("rejected_ids:\n{}".format(example["rejected_ids"]))
    print("rejected:\n{}".format(tokenizer.decode(example["rejected_ids"], skip_special_tokens=False)))


def print_unsupervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))


def get_preprocess_and_print_func(
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
) -> Tuple[Callable, Callable]:
    if stage == "pt":
        preprocess_func = partial(
            preprocess_pretrain_dataset,
            tokenizer=tokenizer,
            data_args=data_args,
        )
        print_function = partial(print_unsupervised_dataset_example, tokenizer=tokenizer)
    elif stage == "sft" and not training_args.predict_with_generate:
        if data_args.packing:
            preprocess_func = partial(
                preprocess_packed_supervised_dataset,
                template=template,
                tokenizer=tokenizer,
                data_args=data_args,
            )
        else:
            preprocess_func = partial(
                preprocess_supervised_dataset,
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                data_args=data_args,
            )

        print_function = partial(print_supervised_dataset_example, tokenizer=tokenizer)
    elif stage == "rm":
        preprocess_func = partial(
            preprocess_pairwise_dataset,
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
        )
        print_function = partial(print_pairwise_dataset_example, tokenizer=tokenizer)
    elif stage == "kto":
        preprocess_func = partial(
            preprocess_kto_dataset,
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
        )
        print_function = partial(print_supervised_dataset_example, tokenizer=tokenizer)
    else:
        preprocess_func = partial(
            preprocess_unsupervised_dataset,
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
        )
        print_function = partial(print_unsupervised_dataset_example, tokenizer=tokenizer)

    return preprocess_func, print_function
