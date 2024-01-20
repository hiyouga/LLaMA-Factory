import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal, Optional

from ..extras.constants import DATA_CONFIG
from ..extras.misc import use_modelscope


if TYPE_CHECKING:
    from ..hparams import DataArguments


@dataclass
class DatasetAttr:
    load_from: Literal["hf_hub", "ms_hub", "script", "file"]
    dataset_name: Optional[str] = None
    dataset_sha1: Optional[str] = None
    subset: Optional[str] = None
    folder: Optional[str] = None
    ranking: Optional[bool] = False
    formatting: Optional[Literal["alpaca", "sharegpt"]] = "alpaca"

    system: Optional[str] = None

    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    history: Optional[str] = None

    messages: Optional[str] = "conversations"
    tools: Optional[str] = None

    role_tag: Optional[str] = "from"
    content_tag: Optional[str] = "value"
    user_tag: Optional[str] = "human"
    assistant_tag: Optional[str] = "gpt"
    observation_tag: Optional[str] = "observation"
    function_tag: Optional[str] = "function_call"

    def __repr__(self) -> str:
        return self.dataset_name


def get_dataset_list(data_args: "DataArguments") -> List["DatasetAttr"]:
    dataset_names = [ds.strip() for ds in data_args.dataset.split(",")] if data_args.dataset is not None else []
    try:
        with open(os.path.join(data_args.dataset_dir, DATA_CONFIG), "r") as f:
            dataset_info = json.load(f)
    except Exception as err:
        if data_args.dataset is not None:
            raise ValueError(
                "Cannot open {} due to {}.".format(os.path.join(data_args.dataset_dir, DATA_CONFIG), str(err))
            )
        dataset_info = None

    if data_args.interleave_probs is not None:
        data_args.interleave_probs = [float(prob.strip()) for prob in data_args.interleave_probs.split(",")]

    dataset_list: List[DatasetAttr] = []
    for name in dataset_names:
        if name not in dataset_info:
            raise ValueError("Undefined dataset {} in {}.".format(name, DATA_CONFIG))

        has_hf_url = "hf_hub_url" in dataset_info[name]
        has_ms_url = "ms_hub_url" in dataset_info[name]

        if has_hf_url or has_ms_url:
            if (use_modelscope() and has_ms_url) or (not has_hf_url):
                dataset_attr = DatasetAttr("ms_hub", dataset_name=dataset_info[name]["ms_hub_url"])
            else:
                dataset_attr = DatasetAttr("hf_hub", dataset_name=dataset_info[name]["hf_hub_url"])
        elif "script_url" in dataset_info[name]:
            dataset_attr = DatasetAttr("script", dataset_name=dataset_info[name]["script_url"])
        else:
            dataset_attr = DatasetAttr(
                "file",
                dataset_name=dataset_info[name]["file_name"],
                dataset_sha1=dataset_info[name].get("file_sha1", None),
            )

        dataset_attr.subset = dataset_info[name].get("subset", None)
        dataset_attr.folder = dataset_info[name].get("folder", None)
        dataset_attr.ranking = dataset_info[name].get("ranking", False)
        dataset_attr.formatting = dataset_info[name].get("formatting", "alpaca")

        if "columns" in dataset_info[name]:
            if dataset_attr.formatting == "alpaca":
                column_names = ["prompt", "query", "response", "history"]
            else:
                column_names = ["messages", "tools"]

            column_names += ["system"]
            for column_name in column_names:
                setattr(dataset_attr, column_name, dataset_info[name]["columns"].get(column_name, None))

        if dataset_attr.formatting == "sharegpt" and "tags" in dataset_info[name]:
            for tag in ["role_tag", "content_tag", "user_tag", "assistant_tag", "observation_tag", "function_tag"]:
                setattr(dataset_attr, tag, dataset_info[name]["tags"].get(tag, None))

        dataset_list.append(dataset_attr)

    return dataset_list
