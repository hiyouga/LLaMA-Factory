import os
from typing import TYPE_CHECKING, Any, Dict, List, Union

from datasets import concatenate_datasets, interleave_datasets, load_dataset

from llmtuner.dsets.utils import checksum, EXT2TYPE
from llmtuner.extras.logging import get_logger

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from llmtuner.hparams import ModelArguments, DataArguments


logger = get_logger(__name__)


def get_dataset(
    model_args: "ModelArguments",
    data_args: "DataArguments"
) -> Union["Dataset", "IterableDataset"]:
    max_samples = data_args.max_samples
    all_datasets: List[Union["Dataset", "IterableDataset"]] = [] # support multiple datasets

    for dataset_attr in data_args.dataset_list:
        logger.info("Loading dataset {}...".format(dataset_attr))

        if dataset_attr.load_from == "hf_hub":
            data_path = dataset_attr.dataset_name
            data_name = dataset_attr.subset
            data_files = None
        elif dataset_attr.load_from == "script":
            data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
            data_name = dataset_attr.subset
            data_files = None
        elif dataset_attr.load_from == "file":
            data_path, data_name = None, None
            data_files: List[str] = []
            if os.path.isdir(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)): # is directory
                for file_name in os.listdir(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)):
                    data_files.append(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name, file_name))
                    if data_path is None:
                        data_path = EXT2TYPE.get(file_name.split(".")[-1], None)
                    else:
                        assert data_path == EXT2TYPE.get(file_name.split(".")[-1], None), "file types are not identical."
            elif os.path.isfile(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)): # is file
                data_files.append(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name))
                data_path = EXT2TYPE.get(dataset_attr.dataset_name.split(".")[-1], None)
            else:
                raise ValueError("File not found.")

            assert data_path, "File extension must be txt, csv, json or jsonl."
            checksum(data_files, dataset_attr.dataset_sha1)
        else:
            raise NotImplementedError

        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_files=data_files,
            split=data_args.split,
            cache_dir=model_args.cache_dir,
            streaming=data_args.streaming,
            use_auth_token=True if model_args.use_auth_token else None
        )

        if max_samples is not None: # truncate dataset
            dataset = dataset.select(range(min(len(dataset), max_samples)))

        def convert_format(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            # convert dataset from sharegpt format to alpaca format
            outputs = {"prompt": [], "query": [], "response": [], "history": []}
            for msg_list in examples[dataset_attr.messages]:
                msg_list = msg_list[:len(msg_list) // 2 * 2] # should be multiples of 2
                if len(msg_list) == 0:
                    continue

                msg_pairs = []
                user_role, assistant_role = None, None
                for idx in range(0, len(msg_list), 2):
                    if user_role is None and assistant_role is None:
                        user_role = msg_list[idx][dataset_attr.role]
                        assistant_role = msg_list[idx + 1][dataset_attr.role]
                    else:
                        if (
                            msg_list[idx][dataset_attr.role] != user_role
                            or msg_list[idx+1][dataset_attr.role] != assistant_role
                        ):
                            raise ValueError("Only accepts conversation in u/a/u/a/u/a order.")
                    msg_pairs.append((msg_list[idx][dataset_attr.content], msg_list[idx + 1][dataset_attr.content]))

                if len(msg_pairs) != 0:
                    outputs["prompt"].append(msg_pairs[-1][0])
                    outputs["query"].append("")
                    outputs["response"].append(msg_pairs[-1][1])
                    outputs["history"].append(msg_pairs[:-1])

            return outputs

        if dataset_attr.formatting == "sharegpt": # convert format
            column_names = list(next(iter(dataset)).keys())
            kwargs = {}
            if not data_args.streaming:
                kwargs = dict(
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=(not data_args.overwrite_cache),
                    desc="Converting format of dataset"
                )

            dataset = dataset.map(
                convert_format,
                batched=True,
                remove_columns=column_names,
                **kwargs
            )
        else:
            for column_name in ["prompt", "query", "response", "history"]: # align dataset
                if getattr(dataset_attr, column_name) and getattr(dataset_attr, column_name) != column_name:
                    dataset = dataset.rename_column(getattr(dataset_attr, column_name), column_name)

        if dataset_attr.system_prompt: # add system prompt
            system_prompt = dataset_attr.system_prompt
            if data_args.streaming:
                dataset = dataset.map(lambda _: {"system": system_prompt})
            else:
                dataset = dataset.add_column("system", [system_prompt] * len(dataset))

        all_datasets.append(dataset)

    if len(data_args.dataset_list) == 1:
        return all_datasets[0]
    elif data_args.mix_strategy == "concat":
        if data_args.streaming:
            logger.warning("The samples between different datasets will not be mixed in streaming mode.")
        return concatenate_datasets(all_datasets)
    elif data_args.mix_strategy.startswith("interleave"):
        if not data_args.streaming:
            logger.warning("We recommend using `mix_strategy=concat` in non-streaming mode.")
        return interleave_datasets(
            datasets=all_datasets,
            probabilities=data_args.interleave_probs,
            seed=data_args.seed,
            stopping_strategy="first_exhausted" if data_args.mix_strategy.endswith("under") else "all_exhausted"
        )
    else:
        raise ValueError("Unknown mixing strategy.")
