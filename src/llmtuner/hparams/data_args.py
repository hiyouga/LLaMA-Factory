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
    ranking: Optional[bool] = False
    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    history: Optional[str] = None

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
        default="alpaca_en",
        metadata={"help": "The name of provided dataset(s) to use. Use commas to separate multiple datasets."}
    )
    dataset_dir: Optional[str] = field(
        default="data",
        metadata={"help": "The name of the folder containing datasets."}
    )
    split: Optional[str] = field(
        default="train",
        metadata={"help": "Which dataset split to use for training and evaluation."}
    )
    streaming: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable streaming mode."}
    )
    buffer_size: Optional[int] = field(
        default=1024,
        metadata={"help": "Size of the buffer to randomly sample examples from in streaming mode."}
    )
    mix_strategy: Optional[Literal["concat", "interleave_under", "interleave_over"]] = field(
        default="concat",
        metadata={"help": "Strategy to use in dataset mixing."}
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
    max_source_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    max_target_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum total output sequence length after tokenization."}
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

    def init_for_training(self): # support mixing multiple datasets
        dataset_names = [ds.strip() for ds in self.dataset.split(",")]
        with open(os.path.join(self.dataset_dir, "dataset_info.json"), "r") as f:
            dataset_info = json.load(f)

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

            dataset_attr.ranking = dataset_info[name].get("ranking", False)
            dataset_attr.system_prompt = prompt_list[i]
            self.dataset_list.append(dataset_attr)
