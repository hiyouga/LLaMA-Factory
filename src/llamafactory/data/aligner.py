import os
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Union

from datasets import Features

from ..extras.logging import get_logger
from .utils import Role


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from ..hparams import DataArguments
    from .parser import DatasetAttr


logger = get_logger(__name__)


def _convert_images(images: List[Any], dataset_attr: "DatasetAttr", data_args: "DataArguments") -> List[Any]:
    r"""
    Optionally concatenates image path to dataset dir when loading from local disk.
    """
    outputs = []
    if dataset_attr.load_from in ["script", "file"]:
        for image in images:
            if isinstance(image, str) and os.path.isfile(os.path.join(data_args.dataset_dir, image)):
                outputs.append(os.path.join(data_args.dataset_dir, image))
            else:
                outputs.append(image)

    return outputs


def convert_alpaca(
    examples: Dict[str, List[Any]], dataset_attr: "DatasetAttr", data_args: "DataArguments"
) -> Dict[str, List[Any]]:
    r"""
    Converts alpaca format dataset to the standard format.
    """
    outputs = {"prompt": [], "response": [], "system": [], "tools": [], "images": []}
    convert_images = partial(_convert_images, dataset_attr=dataset_attr, data_args=data_args)
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

        prompt.append({"role": Role.USER.value, "content": "\n".join(content)})  # "prompt\nquery"

        if dataset_attr.kto_tag and isinstance(examples[dataset_attr.kto_tag][i], bool):  # kto example
            response = [{"role": Role.ASSISTANT.value, "content": examples[dataset_attr.response][i]}]
            if examples[dataset_attr.kto_tag][i]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            dataset_attr.ranking
            and isinstance(examples[dataset_attr.chosen][i], str)
            and isinstance(examples[dataset_attr.rejected][i], str)
        ):  # pairwise example
            response = [
                {"role": Role.ASSISTANT.value, "content": examples[dataset_attr.chosen][i]},
                {"role": Role.ASSISTANT.value, "content": examples[dataset_attr.rejected][i]},
            ]
        elif dataset_attr.response and isinstance(examples[dataset_attr.response][i], str):  # normal example
            response = [{"role": Role.ASSISTANT.value, "content": examples[dataset_attr.response][i]}]
        else:  # unsupervised
            response = []

        outputs["prompt"].append(prompt)
        outputs["response"].append(response)
        outputs["system"].append(examples[dataset_attr.system][i] if dataset_attr.system else "")
        outputs["tools"].append(examples[dataset_attr.tools][i] if dataset_attr.tools else "")
        outputs["images"].append(convert_images(examples[dataset_attr.images][i]) if dataset_attr.images else [])

    return outputs


def convert_sharegpt(
    examples: Dict[str, List[Any]], dataset_attr: "DatasetAttr", data_args: "DataArguments"
) -> Dict[str, List[Any]]:
    r"""
    Converts sharegpt format dataset to the standard format.
    """
    outputs = {"prompt": [], "response": [], "system": [], "tools": [], "images": []}
    convert_images = partial(_convert_images, dataset_attr=dataset_attr, data_args=data_args)
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

        if len(messages) == 0:
            continue

        aligned_messages = []
        broken_data = False
        for turn_idx, message in enumerate(messages):
            if message[dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
                logger.warning("Invalid role tag in {}.".format(messages))
                broken_data = True

            aligned_messages.append(
                {"role": tag_mapping[message[dataset_attr.role_tag]], "content": message[dataset_attr.content_tag]}
            )

        if (not dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
            dataset_attr.ranking and len(aligned_messages) % 2 == 0
        ):
            logger.warning("Invalid message count in {}.".format(messages))
            broken_data = True

        if dataset_attr.kto_tag and isinstance(examples[dataset_attr.kto_tag][i], bool):  # kto example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]
            if examples[dataset_attr.kto_tag][i]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            dataset_attr.ranking
            and isinstance(examples[dataset_attr.chosen][i], dict)
            and isinstance(examples[dataset_attr.rejected][i], dict)
        ):  # pairwise example
            chosen = examples[dataset_attr.chosen][i]
            rejected = examples[dataset_attr.rejected][i]
            if (
                chosen[dataset_attr.role_tag] not in accept_tags[-1]
                or rejected[dataset_attr.role_tag] not in accept_tags[-1]
            ):
                logger.warning("Invalid role tag in {}.".format([chosen, rejected]))
                broken_data = True

            prompt = aligned_messages
            response = [
                {"role": tag_mapping[chosen[dataset_attr.role_tag]], "content": chosen[dataset_attr.content_tag]},
                {"role": tag_mapping[rejected[dataset_attr.role_tag]], "content": rejected[dataset_attr.content_tag]},
            ]
        else:  # normal example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]

        if broken_data:
            logger.warning("Skipping this abnormal example.")
            continue

        outputs["prompt"].append(prompt)
        outputs["response"].append(response)
        outputs["system"].append(system)
        outputs["tools"].append(examples[dataset_attr.tools][i] if dataset_attr.tools else "")
        outputs["images"].append(convert_images(examples[dataset_attr.images][i]) if dataset_attr.images else [])

    return outputs


def align_dataset(
    dataset: Union["Dataset", "IterableDataset"], dataset_attr: "DatasetAttr", data_args: "DataArguments"
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Aligned dataset:
        prompt: [{"role": "user", "content": "..."}] * (2T - 1)
        response: [{"role": "assistant", "content": "..."}] * N (N > 1 for ranking dataset)
        system: "..."
        tools: "...",
        images: [],
    """
    if dataset_attr.formatting == "alpaca":
        convert_func = partial(convert_alpaca, dataset_attr=dataset_attr, data_args=data_args)
    else:
        convert_func = partial(convert_sharegpt, dataset_attr=dataset_attr, data_args=data_args)

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
            "images": [{"_type": "Image"}],
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
