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


from typing import Any, Literal, NotRequired, TypedDict

from ...utils import logging
from ...utils.plugin import BasePlugin
from ...utils.types import DPOSample, Sample, SFTSample


logger = logging.get_logger(__name__)


class AlpacaSample(TypedDict, total=False):
    system: NotRequired[str]
    instruction: str
    input: NotRequired[str]
    output: str


SharegptMessage = TypedDict(
    "SharegptMessage", {"from": Literal["human", "gpt", "system", "function_call", "observation"], "value": str}
)


class SharegptSample(TypedDict, total=False):
    conversations: list[SharegptMessage]
    tools: NotRequired[str]


class OpenaiMessage(TypedDict, total=False):
    role: Literal["user", "assistant", "tool"]
    content: str


class OpenaiSample(TypedDict, total=False):
    messages: list[OpenaiMessage]


class PairSample(TypedDict, total=False):
    chosen: list[OpenaiMessage]
    rejected: list[OpenaiMessage]


class DataConverterPlugin(BasePlugin):
    """Plugin for data converters."""

    def __call__(self, raw_sample: dict[str, Any]) -> Sample:
        return super().__call__(raw_sample)


@DataConverterPlugin("alpaca").register
def alpaca_converter(raw_sample: AlpacaSample) -> SFTSample:
    """Convert Alpaca sample to SFT sample.

    See raw example at: https://huggingface.co/datasets/llamafactory/alpaca_gpt4_en

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


@DataConverterPlugin("sharegpt").register
def sharegpt_converter(raw_sample: SharegptSample) -> SFTSample:
    """Convert ShareGPT sample to SFT sample.

    See raw example at: https://huggingface.co/datasets/llamafactory/glaive_toolcall_en

    Args:
        raw_sample (SharegptSample): ShareGPT sample.

    Returns:
        SFTSample: SFT sample.
    """
    tag_mapping = {
        "system": "system",
        "human": "user",
        "gpt": "assistant",
        "observation": "tool",
        "function_call": "assistant",
    }
    messages = []
    tools = raw_sample.get("tools", "")

    for message in raw_sample.get("conversations", []):
        tag = message["from"]
        if tag not in tag_mapping:
            logger.warning_rank0(f"Unsupported role tag {tag} in message: {message}")
        elif tag == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "tool_calls", "value": message["value"]}],
                    "loss_weight": 1.0,
                }
            )
        else:
            messages.append(
                {
                    "role": tag_mapping[tag],
                    "content": [{"type": "text", "value": message["value"]}],
                    "loss_weight": 1.0 if tag == "gpt" else 0.0,
                }
            )

    if tools:
        if messages and messages[0]["role"] == "system":
            messages[0]["content"].append({"type": "tools", "value": tools})
        else:
            messages.insert(0, {"role": "system", "content": [{"type": "tools", "value": tools}], "loss_weight": 0.0})

    return {"messages": messages}


@DataConverterPlugin("pair").register
def pair_converter(raw_sample: PairSample) -> DPOSample:
    """Convert Pair sample to DPO sample.

    See raw example at: https://huggingface.co/datasets/HuggingFaceH4/orca_dpo_pairs

    Args:
        raw_sample (PairSample): pair sample with chosen, rejected fields.

    Returns:
        DPOSample: DPO sample with chosen_messages and rejected_messages.
    """

    def process_message(raw_messages: list[OpenaiMessage]):
        messages = []
        for message in raw_messages:
            messages.append(
                {
                    "role": message["role"],
                    "content": [{"type": "text", "value": message["content"]}],
                    "loss_weight": 1.0 if message["role"] == "assistant" else 0.0,
                }
            )

        return messages

    chosen_messages = process_message(raw_sample.get("chosen", []))
    rejected_messages = process_message(raw_sample.get("rejected", []))

    return {"chosen_messages": chosen_messages, "rejected_messages": rejected_messages}
