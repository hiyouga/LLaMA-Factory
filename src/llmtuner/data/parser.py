import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from ..extras.constants import DATA_CONFIG
from ..extras.misc import use_modelscope


if TYPE_CHECKING:
    from ..hparams import DataArguments


@dataclass
class DatasetAttr:
    r"""
    Dataset attributes.
    """

    """ basic configs """
    load_from: Literal["hf_hub", "ms_hub", "script", "file"]
    dataset_name: str
    """ extra configs """
    file_sha1: Optional[str] = None
    subset: Optional[str] = None
    folder: Optional[str] = None
    ranking: bool = False
    formatting: Literal["alpaca", "sharegpt"] = "alpaca"
    """ columns """
    system: Optional[str] = None
    """ columns for the alpaca format """
    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    history: Optional[str] = None
    """ columns for the sharegpt format """
    messages: Optional[str] = "conversations"
    tools: Optional[str] = None
    """ tags for the sharegpt format """
    role_tag: Optional[str] = "from"
    content_tag: Optional[str] = "value"
    user_tag: Optional[str] = "human"
    assistant_tag: Optional[str] = "gpt"
    observation_tag: Optional[str] = "observation"
    function_tag: Optional[str] = "function_call"
    system_tag: Optional[str] = "system"

    def __repr__(self) -> str:
        return self.dataset_name

    def set_attr(self, key: str, obj: Dict[str, Any], default: Optional[Any] = None) -> None:
        setattr(self, key, obj.get(key, default))


def get_dataset_list(data_args: "DataArguments") -> List["DatasetAttr"]:
    if data_args.dataset is not None:
        dataset_names = [ds.strip() for ds in data_args.dataset.split(",")]
    else:
        dataset_names = []

    if data_args.dataset_dir == "ONLINE":
        dataset_info = None
    else:
        try:
            with open(os.path.join(data_args.dataset_dir, DATA_CONFIG), "r") as f:
                dataset_info = json.load(f)
        except Exception as err:
            if len(dataset_names) != 0:
                raise ValueError(
                    "Cannot open {} due to {}.".format(os.path.join(data_args.dataset_dir, DATA_CONFIG), str(err))
                )
            dataset_info = None

    if data_args.interleave_probs is not None:
        data_args.interleave_probs = [float(prob.strip()) for prob in data_args.interleave_probs.split(",")]

    dataset_list: List[DatasetAttr] = []
    for name in dataset_names:
        if dataset_info is None:
            load_from = "ms_hub" if use_modelscope() else "hf_hub"
            dataset_attr = DatasetAttr(load_from, dataset_name=name)
            dataset_list.append(dataset_attr)
            continue

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
            dataset_attr = DatasetAttr("file", dataset_name=dataset_info[name]["file_name"])

        dataset_attr.set_attr("file_sha1", dataset_info[name])
        dataset_attr.set_attr("subset", dataset_info[name])
        dataset_attr.set_attr("folder", dataset_info[name])
        dataset_attr.set_attr("ranking", dataset_info[name], default=False)
        dataset_attr.set_attr("formatting", dataset_info[name], default="alpaca")

        if "columns" in dataset_info[name]:
            column_names = ["system"]
            if dataset_attr.formatting == "alpaca":
                column_names.extend(["prompt", "query", "response", "history"])
            else:
                column_names.extend(["messages", "tools"])

            for column_name in column_names:
                dataset_attr.set_attr(column_name, dataset_info[name]["columns"])

        if dataset_attr.formatting == "sharegpt" and "tags" in dataset_info[name]:
            tag_names = (
                "role_tag",
                "content_tag",
                "user_tag",
                "assistant_tag",
                "observation_tag",
                "function_tag",
                "system_tag",
            )
            for tag in tag_names:
                dataset_attr.set_attr(tag, dataset_info[name]["tags"])

        dataset_list.append(dataset_attr)

    return dataset_list
