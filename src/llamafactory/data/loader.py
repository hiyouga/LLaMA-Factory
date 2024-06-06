import inspect
import os
import sys
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
from datasets import load_dataset, load_from_disk

from ..extras.constants import FILEEXT2TYPE
from ..extras.logging import get_logger
from ..extras.misc import has_tokenized_data
from .aligner import align_dataset
from .parser import get_dataset_list
from .preprocess import get_preprocess_and_print_func
from .template import get_template_and_fix_tokenizer
from .utils import merge_dataset


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import ProcessorMixin, Seq2SeqTrainingArguments
    from transformers.tokenization_utils import PreTrainedTokenizer

    from ..hparams import DataArguments, ModelArguments
    from .parser import DatasetAttr


logger = get_logger(__name__)


def load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
) -> Union["Dataset", "IterableDataset"]:
    logger.info("Loading dataset {}...".format(dataset_attr))
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
                if data_path is None:
                    data_path = FILEEXT2TYPE.get(file_name.split(".")[-1], None)
                elif data_path != FILEEXT2TYPE.get(file_name.split(".")[-1], None):
                    raise ValueError("File types should be identical.")
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
            data_path = FILEEXT2TYPE.get(local_path.split(".")[-1], None)
        else:
            raise ValueError("File {} not found.".format(local_path))

        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))
    else:
        raise NotImplementedError("Unknown load type: {}.".format(dataset_attr.load_from))

    if dataset_attr.load_from == "ms_hub":
        try:
            from modelscope import MsDataset
            from modelscope.utils.config_ds import MS_DATASETS_CACHE

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
            )
            if isinstance(dataset, MsDataset):
                dataset = dataset.to_hf_dataset()
        except ImportError:
            raise ImportError("Please install modelscope via `pip install modelscope -U`")
    else:
        if "trust_remote_code" in inspect.signature(load_dataset).parameters:  # for datasets==2.16.0
            kwargs = {"trust_remote_code": True}
        else:
            kwargs = {}

        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=data_args.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            streaming=(data_args.streaming and (dataset_attr.load_from != "file")),
            **kwargs,
        )

    if data_args.streaming and (dataset_attr.load_from == "file"):  # faster than specifying streaming=True
        dataset = dataset.to_iterable_dataset()  # TODO: add num shards parameter

    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info("Sampled {} examples from dataset {}.".format(dataset_attr.num_samples, dataset_attr))

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args)


def get_dataset(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> Union["Dataset", "IterableDataset"]:
    template = get_template_and_fix_tokenizer(tokenizer, data_args.template)
    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError("Current template does not support `train_on_prompt`.")

    # Load tokenized dataset
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning("Loading dataset from disk will ignore other data arguments.")
            dataset = load_from_disk(data_args.tokenized_path)
            logger.info("Loaded tokenized dataset from {}.".format(data_args.tokenized_path))
            if data_args.streaming:
                dataset = dataset.to_iterable_dataset()
            return dataset

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    with training_args.main_process_first(desc="load dataset"):
        all_datasets = []
        for dataset_attr in get_dataset_list(data_args):
            if (stage == "rm" and dataset_attr.ranking is False) or (stage != "rm" and dataset_attr.ranking is True):
                raise ValueError("The dataset is not applicable in the current training stage.")

            all_datasets.append(load_single_dataset(dataset_attr, model_args, data_args))
        dataset = merge_dataset(all_datasets, data_args, training_args)

    with training_args.main_process_first(desc="pre-process dataset"):
        preprocess_func, print_function = get_preprocess_and_print_func(
            data_args, training_args, stage, template, tokenizer, processor
        )
        column_names = list(next(iter(dataset)).keys())
        kwargs = {}
        if not data_args.streaming:
            kwargs = dict(
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=(not data_args.overwrite_cache),
                desc="Running tokenizer on dataset",
            )

        dataset = dataset.map(preprocess_func, batched=True, remove_columns=column_names, **kwargs)

        if data_args.tokenized_path is not None:
            if training_args.should_save:
                dataset.save_to_disk(data_args.tokenized_path)
                logger.info("Tokenized dataset saved at {}.".format(data_args.tokenized_path))
                logger.info("Please restart the training with `tokenized_path: {}`.".format(data_args.tokenized_path))

            sys.exit(0)

        if training_args.should_log:
            try:
                print_function(next(iter(dataset)))
            except StopIteration:
                if stage == "pt":
                    raise RuntimeError("Cannot find sufficient samples, consider increasing dataset size.")
                else:
                    raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

        return dataset
