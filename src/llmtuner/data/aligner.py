from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Union

from datasets import Features

from .utils import Role


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from ..hparams import DataArguments
    from .parser import DatasetAttr


def convert_alpaca(examples: Dict[str, List[Any]], dataset_attr: "DatasetAttr") -> Dict[str, List[Any]]:
    outputs = {"prompt": [], "response": [], "system": [], "tools": []}
    for i in range(len(examples[dataset_attr.prompt])):
        prompt = []
        if dataset_attr.history and isinstance(examples[dataset_attr.history][i], list):
            for old_prompt, old_response in examples[dataset_attr.history][i]:
                prompt.append({"role": Role.USER.value, "content": old_prompt})
                prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

        content = []
        if dataset_attr.prompt and examples[dataset_attr.prompt][i]:
            content.append(examples[dataset_attr.prompt][i])

        if dataset_attr.query and examples[dataset_attr.query][i]:
            content.append(examples[dataset_attr.query][i])

        prompt.append({"role": Role.USER.value, "content": "\n".join(content)})

        if dataset_attr.response and isinstance(examples[dataset_attr.response][i], list):
            response = [
                {"role": Role.ASSISTANT.value, "content": content} for content in examples[dataset_attr.response][i]
            ]
        elif dataset_attr.response and isinstance(examples[dataset_attr.response][i], str):
            response = [{"role": Role.ASSISTANT.value, "content": examples[dataset_attr.response][i]}]
        else:
            response = []

        outputs["prompt"].append(prompt)
        outputs["response"].append(response)
        outputs["system"].append(examples[dataset_attr.system][i] if dataset_attr.system else "")
        outputs["tools"].append("")

    return outputs


def convert_sharegpt(examples: Dict[str, List[Any]], dataset_attr: "DatasetAttr") -> Dict[str, List[Any]]:
    outputs = {"prompt": [], "response": [], "system": [], "tools": []}
    tag_mapping = {
        dataset_attr.user_tag: Role.USER.value,
        dataset_attr.assistant_tag: Role.ASSISTANT.value,
        dataset_attr.observation_tag: Role.OBSERVATION.value,
        dataset_attr.function_tag: Role.FUNCTION.value,
        dataset_attr.system_tag: Role.SYSTEM.value,
    }
    odd_tags = (dataset_attr.user_tag, dataset_attr.observation_tag)
    even_tags = (dataset_attr.assistant_tag, dataset_attr.function_tag)
    accept_tags = (odd_tags, even_tags)
    for i, messages in enumerate(examples[dataset_attr.messages]):
        if dataset_attr.system_tag and messages[0][dataset_attr.role_tag] == dataset_attr.system_tag:
            system = messages[0][dataset_attr.content_tag]
            messages = messages[1:]
        else:
            system = examples[dataset_attr.system][i] if dataset_attr.system else ""

        messages = messages[: len(messages) // 2 * 2]  # should be multiples of 2
        if len(messages) == 0:
            continue

        aligned_messages = []
        for turn_idx, message in enumerate(messages):
            if message[dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
                raise ValueError("Invalid role tag in {}.".format(messages))

            aligned_messages.append(
                {"role": tag_mapping[message[dataset_attr.role_tag]], "content": message[dataset_attr.content_tag]}
            )

        outputs["prompt"].append(aligned_messages[:-1])
        outputs["response"].append(aligned_messages[-1:])
        outputs["system"].append(system)
        outputs["tools"].append(examples[dataset_attr.tools][i] if dataset_attr.tools else "")

    return outputs


def align_dataset(
    dataset: Union["Dataset", "IterableDataset"], dataset_attr: "DatasetAttr", data_args: "DataArguments"
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Aligned dataset:
        prompt: [{"role": "user", "content": "..."}] * (2T - 1)
        response: [{"role": "assistant", "content": "..."}] * N (N > 1 for ranking dataset)
        system: "..."
        tools: "..."
    """
    if dataset_attr.formatting == "alpaca":
        convert_func = partial(convert_alpaca, dataset_attr=dataset_attr)
    else:
        convert_func = partial(convert_sharegpt, dataset_attr=dataset_attr)

    column_names = list(next(iter(dataset)).keys())
    features = Features.from_dict(
        {
            "prompt": [
                {"role": {"dtype": "string", "_type": "Value"}, "content": {"dtype": "string", "_type": "Value"}}
            ],
            "response": [
                {"role": {"dtype": "string", "_type": "Value"}, "content": {"dtype": "string", "_type": "Value"}}
            ],
            "system": {"dtype": "string", "_type": "Value"},
            "tools": {"dtype": "string", "_type": "Value"},
        }
    )
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache),
            desc="Converting format of dataset",
        )

    return dataset.map(
        convert_func,
        batched=True,
        remove_columns=column_names,
        features=features,
        **kwargs,
    )
