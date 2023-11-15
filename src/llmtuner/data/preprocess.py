import os
import tiktoken
from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Tuple, Union

from datasets import load_from_disk

from llmtuner.data.template import get_template_and_fix_tokenizer
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.logging import get_logger

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments
    from transformers.tokenization_utils import PreTrainedTokenizer
    from llmtuner.hparams import DataArguments


logger = get_logger(__name__)


def construct_example(examples: Dict[str, List[Any]]) -> Generator[Any, None, None]:
    for i in range(len(examples["prompt"])):
        query, response = examples["prompt"][i], examples["response"][i]
        query = query + "\n" + examples["query"][i] if "query" in examples and examples["query"][i] else query
        history = examples["history"][i] if "history" in examples else None
        system = examples["system"][i] if "system" in examples else None
        yield query, response, history, system


def infer_max_len(source_len: int, target_len: int, data_args: "DataArguments") -> Tuple[int, int]:
    max_target_len = int(data_args.cutoff_len * (target_len / (source_len + target_len)))
    max_target_len = max(max_target_len, data_args.reserved_label_len)
    max_source_len = data_args.cutoff_len - max_target_len
    return max_source_len, max_target_len


def preprocess_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo"]
) -> Union["Dataset", "IterableDataset"]:
    template = get_template_and_fix_tokenizer(data_args.template, tokenizer)

    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError("Current template does not support `train_on_prompt`.")

    def preprocess_pretrain_dataset(examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        # build grouped texts with format `X1 X2 X3 ...`
        if isinstance(getattr(tokenizer, "tokenizer", None), tiktoken.Encoding): # for tiktoken tokenizer (Qwen)
            kwargs = dict(allowed_special="all")
        else:
            kwargs = dict(add_special_tokens=True)

        if hasattr(tokenizer, "add_eos_token"): # for LLaMA tokenizer
            add_eos_token_flag = getattr(tokenizer, "add_eos_token")
            setattr(tokenizer, "add_eos_token", True)

        tokenized_examples = tokenizer(examples["prompt"], **kwargs)
        concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        block_size = data_args.cutoff_len
        # we drop the small remainder, and if the total_length < block_size, we exclude this batch
        total_length = (total_length // block_size) * block_size
        # split by chunks of cutoff_len
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # make sure the saved tokenizer is the same as the original one
        if hasattr(tokenizer, "add_eos_token"):
            setattr(tokenizer, "add_eos_token", add_eos_token_flag)
        return result

    def preprocess_supervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for query, response, history, system in construct_example(examples):
            if not (isinstance(query, str) and isinstance(response, str) and query != "" and response != ""):
                continue

            input_ids, labels = [], []
            for turn_idx, (source_ids, target_ids) in enumerate(template.encode_multiturn(
                tokenizer, query, response, history, system
            )):
                source_len, target_len = len(source_ids), len(target_ids)
                max_source_len, max_target_len = infer_max_len(source_len, target_len, data_args)
                if source_len > max_source_len:
                    source_ids = source_ids[:max_source_len]
                if target_len > max_target_len:
                    target_ids = target_ids[:max_target_len]

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

            if len(input_ids) > data_args.cutoff_len:
                input_ids = input_ids[:data_args.cutoff_len]
                labels = labels[:data_args.cutoff_len]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

        return model_inputs

    def preprocess_packed_supervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        input_ids, labels = [], []
        for query, response, history, system in construct_example(examples):
            if not (isinstance(query, str) and isinstance(response, str) and query != "" and response != ""):
                continue

            for turn_idx, (source_ids, target_ids) in enumerate(template.encode_multiturn(
                tokenizer, query, response, history, system
            )):
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

        total_length = len(input_ids)
        block_size = data_args.cutoff_len
        # we drop the small remainder, and if the total_length < block_size, we exclude this batch
        total_length = (total_length // block_size) * block_size
        # split by chunks of cutoff_len
        for i in range(0, total_length, block_size):
            model_inputs["input_ids"].append(input_ids[i: i + block_size])
            model_inputs["attention_mask"].append([1] * block_size)
            model_inputs["labels"].append(labels[i: i + block_size])

        return model_inputs

    def preprocess_unsupervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        # build inputs with format `<bos> X` and labels with format `Y <eos>`
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for query, response, history, system in construct_example(examples):
            if not (isinstance(query, str) and query != ""):
                continue

            input_ids, labels = template.encode_oneturn(tokenizer, query, response, history, system)

            if template.efficient_eos:
                labels += [tokenizer.eos_token_id]

            if len(input_ids) > data_args.cutoff_len:
                input_ids = input_ids[:data_args.cutoff_len]
            if len(labels) > data_args.cutoff_len:
                labels = labels[:data_args.cutoff_len]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

        return model_inputs

    def preprocess_pairwise_dataset(examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
        model_inputs = {"prompt_ids": [], "chosen_ids": [], "rejected_ids": []}
        for query, response, history, system in construct_example(examples):
            if not (isinstance(query, str) and isinstance(response, list) and query != "" and len(response) > 1):
                continue

            prompt_ids, chosen_ids = template.encode_oneturn(tokenizer, query, response[0], history, system)
            _, rejected_ids = template.encode_oneturn(tokenizer, query, response[1], history, system)

            if template.efficient_eos:
                chosen_ids += [tokenizer.eos_token_id]
                rejected_ids += [tokenizer.eos_token_id]

            source_len, target_len = len(prompt_ids), max(len(chosen_ids), len(rejected_ids))
            max_source_len, max_target_len = infer_max_len(source_len, target_len, data_args)
            if source_len > max_source_len:
                prompt_ids = prompt_ids[:max_source_len]
            if target_len > max_target_len:
                chosen_ids = chosen_ids[:max_target_len]
                rejected_ids = rejected_ids[:max_target_len]

            model_inputs["prompt_ids"].append(prompt_ids)
            model_inputs["chosen_ids"].append(chosen_ids)
            model_inputs["rejected_ids"].append(rejected_ids)

        return model_inputs

    def print_supervised_dataset_example(example: Dict[str, List[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(
            tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, example["labels"])), skip_special_tokens=False)
        ))

    def print_pairwise_dataset_example(example: Dict[str, List[int]]) -> None:
        print("prompt_ids:\n{}".format(example["prompt_ids"]))
        print("prompt:\n{}".format(tokenizer.decode(example["prompt_ids"], skip_special_tokens=False)))
        print("chosen_ids:\n{}".format(example["chosen_ids"]))
        print("chosen:\n{}".format(tokenizer.decode(example["chosen_ids"], skip_special_tokens=False)))
        print("rejected_ids:\n{}".format(example["rejected_ids"]))
        print("rejected:\n{}".format(tokenizer.decode(example["rejected_ids"], skip_special_tokens=False)))

    def print_unsupervised_dataset_example(example: Dict[str, List[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))

    if stage == "pt":
        preprocess_func = preprocess_pretrain_dataset
        print_function = print_unsupervised_dataset_example
    elif stage == "sft" and not training_args.predict_with_generate:
        preprocess_func = preprocess_packed_supervised_dataset if data_args.sft_packing else preprocess_supervised_dataset
        print_function = print_supervised_dataset_example
    elif stage == "rm":
        preprocess_func = preprocess_pairwise_dataset
        print_function = print_pairwise_dataset_example
    else:
        preprocess_func = preprocess_unsupervised_dataset
        print_function = print_unsupervised_dataset_example

    if data_args.cache_path is not None and os.path.exists(data_args.cache_path):
        logger.warning("Loading dataset from disk will ignore other data arguments.")
        return load_from_disk(data_args.cache_path)

    with training_args.main_process_first(desc="dataset map pre-processing"):
        column_names = list(next(iter(dataset)).keys())
        kwargs = {}
        if not data_args.streaming:
            kwargs = dict(
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=(not data_args.overwrite_cache),
                desc="Running tokenizer on dataset"
            )

        dataset = dataset.map(
            preprocess_func,
            batched=True,
            remove_columns=column_names,
            **kwargs
        )

        if data_args.cache_path is not None and not os.path.exists(data_args.cache_path):
            if training_args.should_save:
                dataset.save_to_disk(data_args.cache_path)
            raise SystemExit("Dataset saved, rerun this script with the same `--cache_path`.")

        if training_args.should_log:
            try:
                print_function(next(iter(dataset)))
            except StopIteration:
                raise RuntimeError("Empty dataset!")

        return dataset
