# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import functools
from typing import TYPE_CHECKING, Dict, Literal, Optional, Sequence, Union

import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers.utils.versions import require_version

from ..extras import logging
from ..extras.constants import FILEEXT2TYPE, IGNORE_INDEX
from ..extras.misc import has_tokenized_data
from .aligner import align_dataset
from .data_utils import merge_dataset, split_dataset, preprocess_sp_dataset
from .parser import get_dataset_list
from .preprocess import get_preprocess_and_print_func


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from ..hparams import DataArguments, ModelArguments
    from .data_utils import DatasetModule
    from .parser import DatasetAttr
    from .template import Template


logger = logging.get_logger(__name__)


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Loads a single dataset and aligns it to the standard format.
    """
    logger.info_rank0(f"Loading dataset {dataset_attr}...")
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub", "om_hub"]:
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
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
        else:
            raise ValueError(f"File {local_path} not found.")

        data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))

        if any(data_path != FILEEXT2TYPE.get(os.path.splitext(data_file)[-1][1:], None) for data_file in data_files):
            raise ValueError("File types should be identical.")
    else:
        raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")

    if dataset_attr.load_from == "ms_hub":
        require_version("modelscope>=1.11.0", "To fix: pip install modelscope>=1.11.0")
        from modelscope import MsDataset  # type: ignore
        from modelscope.utils.config_ds import MS_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
        dataset = MsDataset.load(
            dataset_name=data_path,
            subset_name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.ms_hub_token,
            use_streaming=data_args.streaming,
        )
        if isinstance(dataset, MsDataset):
            dataset = dataset.to_hf_dataset()

    elif dataset_attr.load_from == "om_hub":
        require_version("openmind>=0.8.0", "To fix: pip install openmind>=0.8.0")
        from openmind import OmDataset  # type: ignore
        from openmind.utils.hub import OM_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or OM_DATASETS_CACHE
        dataset = OmDataset.load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.om_hub_token,
            streaming=data_args.streaming,
        )
    else:
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            streaming=data_args.streaming,
            trust_remote_code=True,
        )

    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info_rank0(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args, training_args)


def _get_merged_dataset(
    dataset_names: Optional[Sequence[str]],
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Gets the merged datasets in the standard format.
    """
    if dataset_names is None:
        return None

    datasets = []
    for dataset_attr in get_dataset_list(dataset_names, data_args.dataset_dir):
        if (stage == "rm" and dataset_attr.ranking is False) or (stage != "rm" and dataset_attr.ranking is True):
            raise ValueError("The dataset is not applicable in the current training stage.")

        datasets.append(_load_single_dataset(dataset_attr, model_args, data_args, training_args))

    return merge_dataset(datasets, data_args, seed=training_args.seed)


def _get_preprocessed_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Preprocesses the dataset, including format checking and tokenization.
    """
    if dataset is None:
        return None

    preprocess_func, print_function = get_preprocess_and_print_func(
        data_args, stage, template, tokenizer, processor, do_generate=(training_args.predict_with_generate and is_eval)
    )
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
        )

    dataset = dataset.map(
        preprocess_func,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        **kwargs,
    )

    if training_args.should_log:
        try:
            print("eval example:" if is_eval else "training example:")
            print_function(next(iter(dataset)))
        except StopIteration:
            if stage == "pt":
                raise RuntimeError("Cannot find sufficient samples, consider increasing dataset size.")
            else:
                raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

    return dataset


def sequence_parallel_decorator(get_dataset):
    @functools.wraps(get_dataset)
    def sequence_parallel_processor(*args, **kwargs):
        dataset_module = get_dataset(*args, **kwargs)
        # get arguments
        # NOTE: hard-coded indexing seems inevitable in such decorator style implementation?
        model_args, data_args, training_args = args[1], args[2], args[3]
        tokenizer = kwargs['tokenizer']        
        if model_args.sequence_parallel_size > 1:
            def pad_sequence(examples):
                max_length = data_args.cutoff_len
                input_pad_token_id = tokenizer.pad_token_id
                assert data_args.ignore_pad_token_for_loss
                label_pad_token_id = IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

                for k, v in examples.items():
                    if k.endswith('input_ids'):
                        pad_token_id = input_pad_token_id
                    elif k.endswith('labels'):
                        pad_token_id = label_pad_token_id
                        # shift labels here
                        v = [seq[1:] for seq in v]
                    elif k.endswith('attention_mask'):
                        pad_token_id = 0
                    elif k.endswith('position_ids'):
                        pad_token_id = max_length - 1  # pad the max position id
                    elif k == 'images' or k == 'videos':
                        pad_token_id = -1
                        continue  # TODO: haven't tested multi-modal yet
                    else:
                        raise NotImplementedError(f"Unexpected dataset key: {k}")
                    examples[k] = [seq + [pad_token_id] * (max_length - len(seq)) for seq in v]

                return examples

            # sp for Sequence Parallel
            def sp_split(examples):
                for k, v in examples.items():
                    chunks = list()
                    for row in v:
                        if row is None:
                            chunks += [None] * model_args.sequence_parallel_size
                        else:
                            chunks += preprocess_sp_dataset(row, model_args.sequence_parallel_size, model_args.sequence_parallel_mode)
                    examples[k] = chunks
                return examples

            # padding then split
            for k in dataset_module:
                dataset = dataset_module[k]
                if data_args.shuffle_for_sequence_parallel:
                    dataset = dataset.shuffle(seed=training_args.seed)
                kwargs = dict(
                    num_proc=data_args.preprocessing_num_workers,
                    # load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
                    desc="Running padding split on dataset",
                )
                padded_dataset = dataset.map(pad_sequence, batched=True, **kwargs)
                kwargs = dict(
                    num_proc=data_args.preprocessing_num_workers,
                    # load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
                    desc="Running sequence parallel split on dataset",
                )
                sp_dataset = padded_dataset.map(sp_split, batched=True, **kwargs)
                dataset_module[k] = sp_dataset

        else:
            # no sequence parallelism
            pass
        
        return dataset_module
    
    return sequence_parallel_processor


@sequence_parallel_decorator
def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""
    Gets the train dataset and optionally gets the evaluation dataset.
    """
    # Load tokenized dataset
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning_rank0("Loading dataset from disk will ignore other data arguments.")
            dataset_dict: "DatasetDict" = load_from_disk(data_args.tokenized_path)
            logger.info_rank0(f"Loaded tokenized dataset from {data_args.tokenized_path}.")

            dataset_module: Dict[str, "Dataset"] = {}
            if "train" in dataset_dict:
                dataset_module["train_dataset"] = dataset_dict["train"]

            if "validation" in dataset_dict:
                dataset_module["eval_dataset"] = dataset_dict["validation"]

            if data_args.streaming:
                dataset_module = {k: v.to_iterable_dataset() for k, v in dataset_module.items()}

            return dataset_module

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    # Load and preprocess dataset
    with training_args.main_process_first(desc="load dataset"):
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)
        eval_dataset = _get_merged_dataset(data_args.eval_dataset, model_args, data_args, training_args, stage)

    with training_args.main_process_first(desc="pre-process dataset"):
        dataset = _get_preprocessed_dataset(
            dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
        )
        eval_dataset = _get_preprocessed_dataset(
            eval_dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
        )

        if data_args.val_size > 1e-6:
            dataset_dict = split_dataset(dataset, data_args, seed=training_args.seed)
        else:
            dataset_dict = {}
            if dataset is not None:
                if data_args.streaming:
                    dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)

                dataset_dict["train"] = dataset

            if eval_dataset is not None:
                if data_args.streaming:
                    eval_dataset = eval_dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)

                dataset_dict["validation"] = eval_dataset

            dataset_dict = DatasetDict(dataset_dict)

        if data_args.tokenized_path is not None:
            if training_args.should_save:
                dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info_rank0(f"Tokenized dataset saved at {data_args.tokenized_path}.")
                logger.info_rank0(f"Please restart the training with `tokenized_path: {data_args.tokenized_path}`.")

            sys.exit(0)

        dataset_module = {}
        if "train" in dataset_dict:
            dataset_module["train_dataset"] = dataset_dict["train"]

        if "validation" in dataset_dict:
            dataset_module["eval_dataset"] = dataset_dict["validation"]

        return dataset_module
