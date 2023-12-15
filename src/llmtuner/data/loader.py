import os
from typing import TYPE_CHECKING, Any, Dict, List, Union

from datasets import concatenate_datasets, interleave_datasets, load_dataset

from llmtuner.data.utils import checksum
from llmtuner.extras.constants import FILEEXT2TYPE
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

        data_path, data_name, data_dir, data_files = None, None, None, None
        if dataset_attr.load_from in ["hf_hub", "ms_hub"]:
            data_path = dataset_attr.dataset_name
            data_name = dataset_attr.subset
            data_dir = dataset_attr.folder
        elif dataset_attr.load_from == "script":
            data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
            data_name = dataset_attr.subset
        elif dataset_attr.load_from == "file":
            data_files = []
            local_path: str = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
            if os.path.isdir(local_path): # is directory
                for file_name in os.listdir(local_path):
                    data_files.append(os.path.join(local_path, file_name))
                    if data_path is None:
                        data_path = FILEEXT2TYPE.get(file_name.split(".")[-1], None)
                    else:
                        assert data_path == FILEEXT2TYPE.get(file_name.split(".")[-1], None), "file types are not identical."
            elif os.path.isfile(local_path): # is file
                data_files.append(local_path)
                data_path = FILEEXT2TYPE.get(local_path.split(".")[-1], None)
            else:
                raise ValueError("File not found.")

            assert data_path, "File extension must be txt, csv, json or jsonl."
            checksum(data_files, dataset_attr.dataset_sha1)
        else:
            raise NotImplementedError

        if dataset_attr.load_from == "ms_hub":
            try:
                from modelscope import MsDataset # type: ignore
                from modelscope.utils.config_ds import MS_DATASETS_CACHE # type: ignore

                cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
                dataset = MsDataset.load(
                    dataset_name=data_path,
                    subset_name=data_name,
                    data_dir=data_dir,
                    data_files=data_files,
                    split=data_args.split,
                    cache_dir=cache_dir,
                    token=model_args.ms_hub_token,
                    use_streaming=(data_args.streaming and (dataset_attr.load_from != "file")),
                ).to_hf_dataset()
            except ImportError:
                raise ImportError("Please install modelscope via `pip install modelscope -U`")
        else:
            dataset = load_dataset(
                path=data_path,
                name=data_name,
                data_dir=data_dir,
                data_files=data_files,
                split=data_args.split,
                cache_dir=model_args.cache_dir,
                token=model_args.hf_hub_token,
                streaming=(data_args.streaming and (dataset_attr.load_from != "file"))
            )

        if data_args.streaming and (dataset_attr.load_from == "file"): # faster than specifying streaming=True
            dataset = dataset.to_iterable_dataset() # TODO: add num shards parameter

        if max_samples is not None: # truncate dataset
            dataset = dataset.select(range(min(len(dataset), max_samples)))

        def convert_format(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            # convert dataset from sharegpt format to alpaca format
            outputs = {"prompt": [], "query": [], "response": [], "history": [], "system": []}
            for i, msg_list in enumerate(examples[dataset_attr.messages]):
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
                    outputs["history"].append(msg_pairs[:-1] if len(msg_pairs) > 1 else None)
                    outputs["system"].append(examples[dataset_attr.system][i] if dataset_attr.system else "")

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
            for column_name in ["prompt", "query", "response", "history", "system"]: # align dataset
                if getattr(dataset_attr, column_name) and getattr(dataset_attr, column_name) != column_name:
                    dataset = dataset.rename_column(getattr(dataset_attr, column_name), column_name)

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
