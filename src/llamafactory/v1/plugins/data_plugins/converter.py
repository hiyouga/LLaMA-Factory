# Copyright 2025 the LlamaFactory team.
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


from typing import Callable, TypedDict

from typing_extensions import NotRequired, Required

from ....extras import logging
from ...extras.types import DPOSample, Sample, SFTSample


logger = logging.get_logger(__name__)


class AlpacaSample(TypedDict, total=False):
    system: NotRequired[str]
    instruction: NotRequired[str]
    input: NotRequired[str]
    output: NotRequired[str]


ShareGPTMessage = TypedDict(
    "ShareGPTMessage",
    {
        "from": Required[str],  # Role of the message sender (e.g., "human", "gpt", "system")
        "value": Required[str],  # Content of the message
    },
)


class ShareGPTSample(TypedDict, total=False):
    """Type definition for raw ShareGPT sample."""

    conversations: Required[list[ShareGPTMessage]]


class PairSample(TypedDict, total=False):
    prompt: NotRequired[str]
    chosen: NotRequired[list[dict]]
    rejected: NotRequired[list[dict]]


def alpaca_converter(raw_sample: AlpacaSample) -> SFTSample:
    """Convert Alpaca sample to SFT sample.

    Args:
        raw_sample (AlpacaSample): Alpaca sample.

    Returns:
        SFTSample: SFT sample.
    """
    messages = []
    if "system" in raw_sample:
        messages.append(
            {"role": "system", "content": [{"type": "text", "value": raw_sample["system"]}], "loss_weight": 0.0}
        )

    if "history" in raw_sample:
        for idx, item in enumerate(raw_sample["history"]):
            if len(item) != 2:
                logger.warning_rank0(
                    f"Warning: History item at index {idx} has invalid length (expected 2, got {len(item)}). Skipping."
                )
                continue

            old_prompt, old_response = item
            messages.append({"role": "user", "content": [{"type": "text", "value": old_prompt}], "loss_weight": 0.0})
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "value": old_response}], "loss_weight": 1.0}
            )

    if "instruction" in raw_sample or "input" in raw_sample:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "value": raw_sample.get("instruction", "") + raw_sample.get("input", "")}
                ],
                "loss_weight": 0.0,
            }
        )

    if "output" in raw_sample:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "value": raw_sample["output"]}], "loss_weight": 1.0}
        )

    return {"messages": messages}


def sharegpt_converter(raw_sample: ShareGPTSample) -> SFTSample:
    """Converts a raw ShareGPT sample into a formatted SFT (Supervised Fine-Tuning) sample.

    Retains only SFT-relevant scenarios and removes parity checks.

    Args:
        raw_sample (ShareGPTSample): A raw sample in ShareGPT format.

    Returns:
        dict: A dictionary containing the formatted 'messages' list for SFT training.
              Returns an empty list if the input data is invalid.
    """
    tag_mapping = {
        "human": "user",
        "gpt": "assistant",
        "observation": "observation",
        "function_call": "function",
    }
    messages = raw_sample.get("conversations", [])
    aligned_messages = []
    system_content = ""

    # Extract system message if present (typically the first message)
    if messages and messages[0]["from"] == "system":
        system_content = messages[0]["value"]
        messages = messages[1:]

    if system_content:
        aligned_messages.append(
            {"role": "system", "content": [{"type": "text", "value": system_content}], "loss_weight": 0.0}
        )

    has_invalid_role = False
    for message in messages:
        sender = message["from"]
        # validate sender is in supported tags
        if sender not in tag_mapping:
            logger.warning_rank0(f"Unsupported role tag '{sender}' in message: {message}")
            has_invalid_role = True
            break

        aligned_messages.append(
            {
                "role": tag_mapping[sender],
                "content": [{"type": "text", "value": message["value"]}],
                "loss_weight": 0.0 if sender in ("human", "observation") else 1.0,
            }
        )

    if has_invalid_role:
        logger.warning_rank0("Skipping invalid example due to unsupported role tags.")
        return {"messages": []}

    return {"messages": aligned_messages}


def pair_converter(raw_sample: PairSample) -> DPOSample:
    """Convert Pair sample to standard DPO sample.

    Args:
        raw_sample (PairSample): pair sample with prompt, chosen, rejected fields.
        see raw example at: https://huggingface.co/datasets/HuggingFaceH4/orca_dpo_pairs

    Returns:
        DPOSample: DPO sample with chosen_messages and rejected_messages.
        see the standard DPO sample at: https://huggingface.co/datasets/frozenleaves/v1-dpo-demo/raw/main/v1-dpo-demo.jsonl
    """
    chosen_messages = []
    assert "chosen" in raw_sample, "chosen field is required in pair sample."
    assert "rejected" in raw_sample, "rejected field is required in pair sample."
    assert isinstance(raw_sample["chosen"], list) and isinstance(raw_sample["rejected"], list), (
        "chosen and rejected field should be a list[dict], or you may need to implement your custom converter."
    )

    if "chosen" in raw_sample:
        value = raw_sample.get("chosen", "")
        for item in value:
            if item.get("role", "") == "system":
                chosen_messages.append(
                    {
                        "role": "system",
                        "content": [{"type": "text", "value": item.get("content", "")}],
                        "loss_weight": 0.0,
                    }
                )
            if item.get("role", "") == "user":
                chosen_messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "value": item.get("content", "")}],
                        "loss_weight": 0.0,
                    }
                )
            if item.get("role", "") == "assistant":
                chosen_messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "value": item.get("content", "")}],
                        "loss_weight": 1.0,
                    }
                )

    rejected_messages = []
    if "rejected" in raw_sample:
        value = raw_sample.get("rejected", "")
        for item in value:
            if item.get("role", "") == "system":
                rejected_messages.append(
                    {
                        "role": "system",
                        "content": [{"type": "text", "value": item.get("content", "")}],
                        "loss_weight": 0.0,
                    }
                )
            if item.get("role", "") == "user":
                rejected_messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "value": item.get("content", "")}],
                        "loss_weight": 0.0,
                    }
                )
            if item.get("role", "") == "assistant":
                rejected_messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "value": item.get("content", "")}],
                        "loss_weight": 1.0,
                    }
                )

    return {"chosen_messages": chosen_messages, "rejected_messages": rejected_messages}


CONVERTERS = {
    "alpaca": alpaca_converter,
    "pair": pair_converter,
    "sharegpt": sharegpt_converter,
}


def get_converter(converter_name: str) -> Callable[[dict], Sample]:
    if converter_name not in CONVERTERS:
        raise ValueError(f"Converter {converter_name} not found.")

    return CONVERTERS[converter_name]
