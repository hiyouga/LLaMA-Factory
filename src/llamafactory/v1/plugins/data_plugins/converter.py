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

from typing_extensions import NotRequired

from ...extras.types import Sample, SFTSample, DPOSample


class AlpacaSample(TypedDict, total=False):
    system: NotRequired[str]
    instruction: NotRequired[str]
    input: NotRequired[str]
    output: NotRequired[str]


class PairSample(TypedDict, total=False):
    prompt: NotRequired[str]
    chosen: NotRequired[str]
    rejected: NotRequired[str]


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


def pair_converter(raw_sample: PairSample) -> DPOSample:
    """Convert Pair sample to standard DPO sample.

    Args:
        raw_sample (PairSample): pair sample with prompt, chosen, rejected fields.

    Returns:
        DPOSample: DPO sample with chosen_messages and rejected_messages.
    """
    chosen_messages = []
    if "prompt" in raw_sample:
        chosen_messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "value": raw_sample["prompt"]}],
                "loss_weight": 0.0,
            }
        )

    if "chosen" in raw_sample:
        chosen_messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "value": raw_sample["chosen"]}],
                "loss_weight": 1.0,
            }
        )

    rejected_messages = []
    if "prompt" in raw_sample:
        rejected_messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "value": raw_sample["prompt"]}],
                "loss_weight": 0.0,
            }
        )

    if "rejected" in raw_sample:
        rejected_messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "value": raw_sample["rejected"]}],
                "loss_weight": 1.0,
            }
        )

    return {
        "chosen_messages": chosen_messages,
        "rejected_messages": rejected_messages
    }


CONVERTERS = {
    "alpaca": alpaca_converter,
    'pair': pair_converter,
}


def get_converter(converter_name: str) -> Callable[[dict], Sample]:
    if converter_name not in CONVERTERS:
        raise ValueError(f"Converter {converter_name} not found.")

    return CONVERTERS[converter_name]
