from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple
from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import greedy_knapsack, infer_seqlen
import random

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from ...hparams import DataArguments
    from ..mm_plugin import ImageInput, VideoInput
    from ..template import Template

logger = logging.get_logger(__name__)

def _prepare_raft_context(
    positive_context: List[str],
    negative_context: List[str],
    p: float,
    num_distract: int,
) -> List[str]:
    contexts = []
    use_positive = random.random() < p
    
    if use_positive and positive_context:
        contexts.append(random.choice(positive_context))
        if negative_context:
            n_to_sample = min(num_distract, len(negative_context))
            contexts.extend(random.sample(negative_context, n_to_sample))
    else:
        if negative_context:
            n_to_sample = min(num_distract + 1, len(negative_context))
            contexts.extend(random.sample(negative_context, n_to_sample))
    
    random.shuffle(contexts)
    return contexts

def _encode_raft_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    positive_context: List[str],
    negative_context: List[str],
    system: Optional[str],
    tools: Optional[str],
    images: Sequence["ImageInput"],
    videos: Sequence["VideoInput"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
    train_on_prompt: bool,
    mask_history: bool,
    raft_p: float,
    raft_num_distract: int,
) -> Tuple[List[int], List[int]]:
    """
    Encode a RAFT example with context into token IDs.
    Enhanced version of _encode_supervised_example with RAFT context handling.
    """
    contexts = _prepare_raft_context(positive_context, negative_context, raft_p, raft_num_distract)
    context_str = "\n".join([f"{ctx}\n" for ctx in contexts])
    
    if prompt and isinstance(prompt[0], dict) and "content" in prompt[0]:
        prompt[0]["content"] = f"{context_str}\n{prompt[0]['content']}"
    
    messages = template.mm_plugin.process_messages(prompt + response, images, videos, processor)
    input_ids, labels = template.mm_plugin.process_token_ids([], [], images, videos, tokenizer, processor)
    encoded_pairs = template.encode_multiturn(tokenizer, messages, system, tools)
    total_length = len(input_ids) + (1 if template.efficient_eos else 0)
    
    if mask_history:
        encoded_pairs = encoded_pairs[::-1]

    for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        if total_length >= cutoff_len:
            break

        source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), cutoff_len - total_length)
        source_ids = source_ids[:source_len]
        target_ids = target_ids[:target_len]
        total_length += source_len + target_len

        if train_on_prompt:
            source_label = source_ids
        elif template.efficient_eos:
            source_label = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
        else:
            source_label = [IGNORE_INDEX] * source_len

        if mask_history and turn_idx != 0:
            target_label = [IGNORE_INDEX] * target_len
        else:
            target_label = target_ids

        if mask_history:
            input_ids = source_ids + target_ids + input_ids
            labels = source_label + target_label + labels
        else:
            input_ids += source_ids + target_ids
            labels += source_label + target_label

    if template.efficient_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]
    print(tokenizer.decode(input_ids))
    return input_ids, labels

def preprocess_raft_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    """
    Preprocess a RAFT dataset. Supports all features from supervised preprocessing plus RAFT context handling.
    """
    model_inputs = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        if "_positive_context" not in examples or "_negative_context" not in examples:
            print(examples)
            raise ValueError("RAFT dataset must contain 'positive_context' and 'negative_context' fields")

        input_ids, labels = _encode_raft_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            positive_context=examples["_positive_context"][i],
            negative_context=examples["_negative_context"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len,
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
            raft_p=data_args.raft_p,
            raft_num_distract=data_args.raft_num_distract
        )
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])

    return model_inputs

def preprocess_packed_raft_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    """
    Preprocess a packed RAFT dataset. Supports sequence packing for efficient training.
    """
    valid_num = 0
    batch_input_ids, batch_labels, batch_images, batch_videos = [], [], [], []
    lengths = []
    length2indexes = defaultdict(list)
    
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        input_ids, labels = _encode_raft_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            positive_context=examples["_positive_context"][i],
            negative_context=examples["_negative_context"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len - 1, 
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
            raft_p=data_args.raft_p,
            raft_num_distract=data_args.raft_num_distract
        )

        length = len(input_ids)
        if length > data_args.cutoff_len:
            logger.warning_rank0(f"Dropped lengthy example with length {length} > {data_args.cutoff_len}.")
        else:
            lengths.append(length)
            length2indexes[length].append(valid_num)
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_images.append(examples["_images"][i] or [])
            batch_videos.append(examples["_videos"][i] or [])
            valid_num += 1

    model_inputs = defaultdict(list)
    knapsacks = greedy_knapsack(lengths, data_args.cutoff_len - 1)
    
    for knapsack in knapsacks:
        packed_input_ids, packed_attention_masks, packed_labels = [], [], []
        packed_images, packed_videos = [], []
        
        for i, length in enumerate(knapsack):
            index = length2indexes[length].pop()
            packed_input_ids += batch_input_ids[index]
            packed_labels += batch_labels[index]
            packed_images += batch_images[index]
            packed_videos += batch_videos[index]
            
            if data_args.neat_packing:
                packed_attention_masks += [i + 1] * len(batch_input_ids[index])
            else:
                packed_attention_masks += [1] * len(batch_input_ids[index])

        if len(packed_input_ids) < data_args.cutoff_len:
            pad_length = data_args.cutoff_len - len(packed_input_ids)
            packed_input_ids += [tokenizer.pad_token_id] * pad_length
            packed_labels += [IGNORE_INDEX] * pad_length
            if data_args.neat_packing:
                packed_attention_masks += [0] * pad_length
            else:
                packed_attention_masks += [1] * pad_length

        model_inputs["input_ids"].append(packed_input_ids)
        model_inputs["attention_mask"].append(packed_attention_masks)
        model_inputs["labels"].append(packed_labels)
        model_inputs["images"].append(packed_images or None)
        model_inputs["videos"].append(packed_videos or None)

    return model_inputs

def print_raft_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    """Print a RAFT dataset example."""
    valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    print("label_ids:\n{}".format(example["labels"]))
    print(f"labels:\n{tokenizer.decode(valid_labels, skip_special_tokens=False)}")