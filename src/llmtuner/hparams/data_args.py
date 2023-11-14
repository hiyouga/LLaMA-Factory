import os
import json
from typing import List, Literal, Optional
from dataclasses import dataclass, field


@dataclass
class DatasetAttr:

    load_from: str
    dataset_name: Optional[str] = None
    dataset_sha1: Optional[str] = None
    system_prompt: Optional[str] = None
    subset: Optional[str] = None
    ranking: Optional[bool] = False
    formatting: Optional[Literal["alpaca", "sharegpt"]] = "alpaca"

    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    history: Optional[str] = None
    messages: Optional[str] = "conversations"
    role: Optional[str] = "from"
    content: Optional[str] = "value"

    def __repr__(self) -> str:
        return self.dataset_name


@dataclass
class DataArguments:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    template: Optional[str] = field(
        default=None,
        metadata={"help": "Which template to use for constructing prompts in training and inference."}
    )
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The name of provided dataset(s) to use. Use commas to separate multiple datasets."}
    )
    dataset_dir: Optional[str] = field(
        default="data",
        metadata={"help": "Path to the folder containing the datasets."}
    )
    split: Optional[str] = field(
        default="train",
        metadata={"help": "Which dataset split to use for training and evaluation."}
    )
    cutoff_len: Optional[int] = field(
        default=1024,
        metadata={"help": "The maximum length of the model inputs after tokenization."}
    )
    reserved_label_len: Optional[int] = field(
        default=1,
        metadata={"help": "The maximum length reserved for label after tokenization."}
    )
    train_on_prompt: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to disable the mask on the prompt or not."}
    )
    streaming: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable dataset streaming."}
    )
    buffer_size: Optional[int] = field(
        default=16384,
        metadata={"help": "Size of the buffer to randomly sample examples from in dataset streaming."}
    )
    mix_strategy: Optional[Literal["concat", "interleave_under", "interleave_over"]] = field(
        default="concat",
        metadata={"help": "Strategy to use in dataset mixing (concat/interleave) (undersampling/oversampling)."}
    )
    interleave_probs: Optional[str] = field(
        default=None,
        metadata={"help": "Probabilities to sample data from datasets. Use commas to separate multiple datasets."}
    )
    overwrite_cache: Optional[bool] = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples for each dataset."}
    )
    eval_num_beams: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`"}
    )
    ignore_pad_token_for_loss: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "System prompt to add before the user query. Use `|` to separate multiple prompts in training."}
    )
    val_size: Optional[float] = field(
        default=0,
        metadata={"help": "Size of the development set, should be an integer or a float in range `[0,1)`."}
    )
    sft_packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Packing the questions and answers in the supervised fine-tuning stage."}
    )
    cache_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save or load the preprocessed datasets."}
    )

    def __post_init__(self):
        if self.reserved_label_len >= self.cutoff_len:
            raise ValueError("`reserved_label_len` must be smaller than `cutoff_len`.")

        if self.streaming and self.val_size > 1e-6 and self.val_size < 1:
            raise ValueError("Streaming mode should have an integer val size.")

        if self.streaming and self.max_samples is not None:
            raise ValueError("`max_samples` is incompatible with `streaming`.")

        if self.streaming and self.cache_path:
            raise ValueError("`cache_path` is incompatible with `streaming`.")

    def init_for_training(self, seed: int): # support mixing multiple datasets
        self.seed = seed
        dataset_names = [ds.strip() for ds in self.dataset.split(",")] if self.dataset is not None else []
        try:
            with open(os.path.join(self.dataset_dir, "dataset_info.json"), "r") as f:
                dataset_info = json.load(f)
        except Exception:
            if self.dataset is not None:
                raise ValueError("Cannot find dataset_info.json in `dataset_dir`.")
            dataset_info = None

        prompt_list = self.system_prompt.split("|") if self.system_prompt else [None]
        prompt_list = prompt_list * (len(dataset_names) // len(prompt_list))
        assert len(prompt_list) == len(dataset_names), "Number of system prompts should be equal to datasets or 1."

        if self.interleave_probs is not None:
            self.interleave_probs = [float(prob.strip()) for prob in self.interleave_probs.split(",")]

        self.dataset_list: List[DatasetAttr] = []
        for i, name in enumerate(dataset_names):
            if name not in dataset_info:
                raise ValueError("Undefined dataset {} in dataset_info.json.".format(name))

            if "hf_hub_url" in dataset_info[name]:
                dataset_attr = DatasetAttr("hf_hub", dataset_name=dataset_info[name]["hf_hub_url"])
            elif "script_url" in dataset_info[name]:
                dataset_attr = DatasetAttr("script", dataset_name=dataset_info[name]["script_url"])
            else:
                dataset_attr = DatasetAttr(
                    "file",
                    dataset_name=dataset_info[name]["file_name"],
                    dataset_sha1=dataset_info[name].get("file_sha1", None)
                )

            if "columns" in dataset_info[name]:
                dataset_attr.prompt = dataset_info[name]["columns"].get("prompt", None)
                dataset_attr.query = dataset_info[name]["columns"].get("query", None)
                dataset_attr.response = dataset_info[name]["columns"].get("response", None)
                dataset_attr.history = dataset_info[name]["columns"].get("history", None)
                dataset_attr.messages = dataset_info[name]["columns"].get("messages", None)
                dataset_attr.role = dataset_info[name]["columns"].get("role", None)
                dataset_attr.content = dataset_info[name]["columns"].get("content", None)

            dataset_attr.subset = dataset_info[name].get("subset", None)
            dataset_attr.ranking = dataset_info[name].get("ranking", False)
            dataset_attr.formatting = dataset_info[name].get("formatting", "alpaca")
            dataset_attr.system_prompt = prompt_list[i]
            self.dataset_list.append(dataset_attr)
