import tiktoken
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Union
from itertools import chain

from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.template import get_template_and_fix_tokenizer

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments
    from transformers.tokenization_utils import PreTrainedTokenizer
    from llmtuner.hparams import DataArguments


def preprocess_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo"]
) -> Union["Dataset", "IterableDataset"]:
    column_names = list(next(iter(dataset)).keys())
    template = get_template_and_fix_tokenizer(data_args.template, tokenizer)

    if template is not None and template.efficient_eos and data_args.sft_packing:
        raise ValueError("Current template is incompatible with packing.")

    def construct_example(examples: Dict[str, List[Any]]) -> Generator[Any, None, None]:
        for i in range(len(examples["prompt"])):
            query, response = examples["prompt"][i], examples["response"][i]
            query = query + "\n" + examples["query"][i] if "query" in examples and examples["query"][i] else query
            history = examples["history"][i] if "history" in examples else None
            system = examples["system"][i] if "system" in examples else None
            yield query, response, history, system

    def preprocess_pretrain_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build grouped texts with format `X1 X2 X3 ...`
        if isinstance(getattr(tokenizer, "tokenizer", None), tiktoken.Encoding):
            kwargs = dict(allowed_special="all") # for tiktoken tokenizer (Qwen)
        else:
            kwargs = dict(add_special_tokens=True)

        if hasattr(tokenizer, "add_bos_token") and hasattr(tokenizer, "add_eos_token"):
            setattr(tokenizer, "add_bos_token", True) # for LLaMA tokenizer
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
        return result

    def preprocess_supervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for query, response, history, system in construct_example(examples):
            input_ids, labels = [], []

            for turn_idx, (source_ids, target_ids) in enumerate(template.encode_multiturn(
                tokenizer, query, response, history, system
            )):
                total_len = len(source_ids) + len(target_ids)
                max_source_len = int(data_args.cutoff_len * (len(source_ids) / total_len))
                max_target_len = int(data_args.cutoff_len * (len(target_ids) / total_len))

                if len(source_ids) > max_source_len:
                    source_ids = source_ids[:max_source_len]
                if len(target_ids) > max_target_len:
                    target_ids = target_ids[:max_target_len]

                if turn_idx != 0 and template.efficient_eos:
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

    def preprocess_packed_supervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # we do not mask the inputs in packed training.
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        input_ids, labels = [], []
        for query, response, history, system in construct_example(examples):
            for source_ids, target_ids in template.encode_multiturn(tokenizer, query, response, history, system):
                input_ids += source_ids + target_ids
                labels += source_ids + target_ids # TODO: try masking source_ids here

        total_length = len(input_ids)
        block_size = data_args.cutoff_len
        # we drop the small remainder, and if the total_length < block_size, we exclude this batch
        total_length = (total_length // block_size) * block_size
        # split by chunks of cutoff_len
        for i in range(0, total_length, block_size):
            model_inputs["input_ids"].append(input_ids[i: i + block_size])
            model_inputs["attention_mask"].append([1] * len(block_size))
            model_inputs["labels"].append(labels[i: i + block_size])

        return model_inputs

    def preprocess_unsupervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build inputs with format `<bos> X` and labels with format `Y <eos>`
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for query, response, history, system in construct_example(examples):
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

    def preprocess_pairwise_dataset(examples):
        # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
        model_inputs = {"prompt_ids": [], "chosen_ids": [], "rejected_ids": []}
        for query, response, history, system in construct_example(examples):
            prompt_ids, chosen_ids = template.encode_oneturn(tokenizer, query, response[0], history, system)
            _, rejected_ids = template.encode_oneturn(tokenizer, query, response[1], history, system)

            if template.efficient_eos:
                chosen_ids += [tokenizer.eos_token_id]
                rejected_ids += [tokenizer.eos_token_id]

            total_len = len(prompt_ids) + max(len(chosen_ids), len(rejected_ids))
            max_source_len = int(data_args.cutoff_len * (len(prompt_ids) / total_len))
            max_target_len = int(data_args.cutoff_len * (max(len(chosen_ids), len(rejected_ids)) / total_len))

            if len(prompt_ids) > max_source_len:
                prompt_ids = prompt_ids[:max_source_len]
            if len(chosen_ids) > max_target_len:
                chosen_ids = chosen_ids[:max_target_len]
            if len(rejected_ids) > max_target_len:
                rejected_ids = rejected_ids[:max_target_len]

            model_inputs["prompt_ids"].append(prompt_ids)
            model_inputs["chosen_ids"].append(chosen_ids)
            model_inputs["rejected_ids"].append(rejected_ids)
        return model_inputs

    def print_supervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(
            tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, example["labels"])), skip_special_tokens=False)
        ))

    def print_pairwise_dataset_example(example):
        print("prompt_ids:\n{}".format(example["prompt_ids"]))
        print("prompt:\n{}".format(tokenizer.decode(example["prompt_ids"], skip_special_tokens=False)))
        print("chosen_ids:\n{}".format(example["chosen_ids"]))
        print("chosen:\n{}".format(tokenizer.decode(example["chosen_ids"], skip_special_tokens=False)))
        print("rejected_ids:\n{}".format(example["rejected_ids"]))
        print("rejected:\n{}".format(tokenizer.decode(example["rejected_ids"], skip_special_tokens=False)))

    def print_unsupervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))

    if stage == "pt":
        dataset = dataset.filter(lambda example: example["prompt"])
        preprocess_func = preprocess_pretrain_dataset
        print_function = print_unsupervised_dataset_example
    elif stage == "sft" and not training_args.predict_with_generate:
        dataset = dataset.filter(lambda example: example["prompt"] and example["response"])
        preprocess_func = preprocess_packed_supervised_dataset if data_args.sft_packing else preprocess_supervised_dataset
        print_function = print_supervised_dataset_example
    elif stage == "rm":
        dataset = dataset.filter(lambda example: example["prompt"] and len(example["response"]) > 1)
        preprocess_func = preprocess_pairwise_dataset
        print_function = print_pairwise_dataset_example
    else:
        dataset = dataset.filter(lambda example: example["prompt"])
        preprocess_func = preprocess_unsupervised_dataset
        print_function = print_unsupervised_dataset_example

    with training_args.main_process_first(desc="dataset map pre-processing"):
        kwargs = {}
        if not data_args.streaming:
            kwargs = dict(
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset"
            )

        dataset = dataset.map(
            preprocess_func,
            batched=True,            
            remove_columns=column_names,
            **kwargs
        )

        print_function(next(iter(dataset)))
        return dataset
