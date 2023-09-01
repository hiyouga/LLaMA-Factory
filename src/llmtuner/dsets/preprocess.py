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
        block_size = data_args.max_source_length
        # we drop the small remainder, and if the total_length < block_size, we exclude this batch
        total_length = (total_length // block_size) * block_size
        # split by chunks of max_source_length
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    def preprocess_supervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        max_length = data_args.max_source_length + data_args.max_target_length

        for query, response, history, system in construct_example(examples):
            input_ids, labels = [], []

            for source_ids, target_ids in template.encode_multiturn(tokenizer, query, response, history, system):
                if len(source_ids) > data_args.max_source_length:
                    source_ids = source_ids[:data_args.max_source_length]
                if len(target_ids) > data_args.max_target_length:
                    target_ids = target_ids[:data_args.max_target_length]

                if len(input_ids) + len(source_ids) + len(target_ids) > max_length:
                    break

                input_ids += source_ids + target_ids
                labels += [IGNORE_INDEX] * len(source_ids) + target_ids

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

        return model_inputs

    def preprocess_unsupervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build inputs with format `<bos> X` and labels with format `Y <eos>`
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for query, response, history, system in construct_example(examples):
            source_ids, target_ids = template.encode_oneturn(tokenizer, query, response, history, system)

            if len(source_ids) > data_args.max_source_length:
                source_ids = source_ids[:data_args.max_source_length]
            if len(target_ids) > data_args.max_target_length:
                target_ids = target_ids[:data_args.max_target_length]

            model_inputs["input_ids"].append(source_ids)
            model_inputs["attention_mask"].append([1] * len(source_ids))
            model_inputs["labels"].append(target_ids)

        return model_inputs

    def preprocess_pairwise_dataset(examples):
        # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
        model_inputs = {"prompt_ids": [], "chosen_ids": [], "rejected_ids": []}
        for query, response, history, system in construct_example(examples):
            prompt_ids, chosen_ids = template.encode_oneturn(tokenizer, query, response[0], history, system)
            _, rejected_ids = template.encode_oneturn(tokenizer, query, response[1], history, system)

            if len(prompt_ids) > data_args.max_source_length:
                prompt_ids = prompt_ids[:data_args.max_source_length]
            if len(chosen_ids) > data_args.max_target_length:
                chosen_ids = chosen_ids[:data_args.max_target_length]
            if len(rejected_ids) > data_args.max_target_length:
                rejected_ids = rejected_ids[:data_args.max_target_length]

            model_inputs["prompt_ids"].append(prompt_ids)
            model_inputs["chosen_ids"].append(chosen_ids)
            model_inputs["rejected_ids"].append(rejected_ids)
        return model_inputs

    def print_supervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(tokenizer.decode([
            token_id if token_id != IGNORE_INDEX else tokenizer.pad_token_id for token_id in example["labels"]
        ], skip_special_tokens=False)))

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
        preprocess_function = preprocess_pretrain_dataset
        print_function = print_unsupervised_dataset_example
    elif stage == "sft" and not training_args.predict_with_generate:
        dataset = dataset.filter(lambda example: example["prompt"] and example["response"])
        preprocess_function = preprocess_supervised_dataset
        print_function = print_supervised_dataset_example
    elif stage == "rm":
        dataset = dataset.filter(lambda example: example["prompt"] and len(example["response"]) > 1)
        preprocess_function = preprocess_pairwise_dataset
        print_function = print_pairwise_dataset_example
    else:
        dataset = dataset.filter(lambda example: example["prompt"])
        preprocess_function = preprocess_unsupervised_dataset
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
            preprocess_function,
            batched=True,            
            remove_columns=column_names,
            **kwargs
        )

        print_function(next(iter(dataset)))
        return dataset
