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

"""Rendering: turn a v1 ``Sample`` into a tokenized ``ModelInput``.

This module is the orchestration + public API (``Renderer``). The mechanical pieces live in
sibling modules:
  - ``format``  -- v1<->HF message conversion
  - ``escape``  -- special-token escaping (prompt-injection hardening)

Assistant supervision is located WITHOUT a per-model marker table: a training sample is rendered
so that its last message is the supervised assistant turn, and that turn's token span is recovered
by a single prompt/full difference -- encode the prompt (everything up to and including the
assistant role header, via ``add_generation_prompt=True``) and the full sequence, then the tail of
the full sequence that the prompt does not cover is exactly this turn. Multi-turn conversations are
split into one sample per supervised turn (see ``process_samples``) so the supervised turn is always
the last one; this keeps the diff on the only boundary that is prefix-stable across chat templates
(appending the final assistant turn never restripts earlier turns), so models with reasoning-history
stripping (e.g. Qwen3 ``<think>``) are handled correctly without hard-coding role markers.
"""

import json

from ...utils.constants import IGNORE_INDEX
from ...utils.helper import get_tokenizer
from ...utils.types import Message, ModelInput, Processor, Sample
from .escape import _escape_special, _escape_special_in_messages, _special_token_strings
from .format import _FALLBACK_CHATML_JINJA, _to_hf_messages


def _render_messages(
    processor: Processor,
    messages: list[Message],
    tools: str | None = None,
    is_generate: bool = False,
    **kwargs,
) -> ModelInput:
    r"""Render messages using the model's own chat template.

    Note: ``position_ids`` are not produced here; ``process_samples`` assigns a 1-based range.
    """

    tokenizer = get_tokenizer(processor)
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = _FALLBACK_CHATML_JINJA

    # 0. Neutralize special-token strings in user-controlled text (no-op for normal data).
    specials = _special_token_strings(tokenizer)
    special_ids = {tid for tid, t in tokenizer.added_tokens_decoder.items() if getattr(t, "special", False)}
    messages = _escape_special_in_messages(messages, specials, special_ids, tokenizer)

    hf_messages = _to_hf_messages(messages)

    tools_parsed = None
    if tools:
        tools = _escape_special(tools, specials, special_ids, tokenizer)  # E3: tools text is user-controlled
        try:
            tools_parsed = json.loads(tools)
        except json.JSONDecodeError as e:
            raise ValueError(f"tools is not valid JSON: {tools!r}") from e
        if not isinstance(tools_parsed, list):
            tools_parsed = [tools_parsed]
    if not is_generate and hf_messages and hf_messages[-1].get("reasoning_content"):
        kwargs["enable_thinking"] = True
    def _encode(msgs: list[dict], add_generation_prompt: bool) -> list[int]:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt, tools=tools_parsed, **kwargs
        )
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    # 1. Full sequence, used verbatim.
    input_ids = _encode(hf_messages, add_generation_prompt=is_generate)
    n = len(input_ids)

    if is_generate:
        # Generation prompt only -- nothing is supervised.
        return ModelInput(
            input_ids=input_ids,
            attention_mask=[1] * n,
            labels=[IGNORE_INDEX] * n,
            loss_weights=[0.0] * n,
        )

    # 2. Locate the supervised (last) assistant turn by a prompt/full diff (no marker table).
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError(
            "training render expects the last message to be the supervised assistant turn; "
            "multi-turn conversations are split per turn in process_samples."
        )

    prompt_ids = _encode(hf_messages[:-1], add_generation_prompt=True)
    if input_ids[: len(prompt_ids)] != prompt_ids:
        # The prompt must be a token-prefix of the full sequence for the diff to be valid. If a
        # template re-renders earlier turns when the final turn is appended, fail loud rather than
        # mislabel.
        raise ValueError(
            "prompt is not a token-prefix of the full sequence; the chat template is not "
            "prefix-stable for this turn, so diff-based labeling is unsafe."
        )

    weight = messages[-1].get("loss_weight", 1.0)
    supervised = weight > 1e-6
    labels = [IGNORE_INDEX] * len(prompt_ids)
    loss_weights = [0.0] * len(prompt_ids)
    for tid in input_ids[len(prompt_ids) :]:
        labels.append(tid if supervised else IGNORE_INDEX)
        loss_weights.append(weight)

    return ModelInput(
        input_ids=input_ids,
        attention_mask=[1] * n,
        labels=labels,
        loss_weights=loss_weights,
    )


class Renderer:
    def __init__(self, processor: Processor) -> None:
        self.processor = processor

    def render_messages(
        self,
        messages: list[Message],
        tools: str | None = None,
        is_generate: bool = False,
        **kwargs,
    ) -> ModelInput:
        """Render messages to model input using apply_chat_template.

        Args:
            messages: The messages to render. For training the last message must be the supervised
                assistant turn (use ``process_samples`` to split multi-turn conversations).
            tools: JSON string of tool definitions.
            is_generate: Whether to render for generation (adds generation prompt, no supervision).
            **kwargs: Extra chat-template kwargs (e.g. ``enable_thinking``) forwarded verbatim to
                ``apply_chat_template``; unset ones fall back to the template's own defaults. A
                supervised assistant turn carrying reasoning forces ``enable_thinking=True``.

        Returns:
            ModelInput with input_ids, attention_mask, labels, and loss_weights.
        """
        return _render_messages(
            self.processor,
            messages,
            tools,
            is_generate,
            **kwargs
        )

    def process_samples(self, samples: list[Sample]) -> list[ModelInput]:
        """Process samples to model input.

        Multi-turn SFT conversations are already prefix-split in the data layer (DataEngine), so each
        ``messages`` sample is rendered once -- the diff-based renderer supervises only its last
        assistant turn.

        Args:
            samples: The samples to process.

        Returns:
            List of processed model inputs.
        """
        model_inputs = []
        for sample in samples:
            rendered: list[ModelInput] = []
            if "messages" in sample:
                model_input = self.render_messages(sample["messages"], sample.get("tools"))
                model_input["position_ids"] = list(range(1, len(model_input["input_ids"]) + 1))
                rendered.append(model_input)
            elif "chosen_messages" in sample and "rejected_messages" in sample:
                chosen_input = self.render_messages(sample["chosen_messages"], sample.get("tools"))
                rejected_input = self.render_messages(sample["rejected_messages"], sample.get("tools"))
                chosen_input["token_type_ids"] = [1] * len(chosen_input["input_ids"])
                rejected_input["token_type_ids"] = [2] * len(rejected_input["input_ids"])
                model_input = ModelInput(
                    input_ids=chosen_input["input_ids"] + rejected_input["input_ids"],
                    attention_mask=chosen_input["attention_mask"] + rejected_input["attention_mask"],
                    labels=chosen_input["labels"] + rejected_input["labels"],
                    loss_weights=chosen_input["loss_weights"] + rejected_input["loss_weights"],
                    token_type_ids=chosen_input["token_type_ids"] + rejected_input["token_type_ids"],
                )
                # chosen and rejected are independent sequences; position ids must restart at 1 for
                # each (a single continuous range would offset rejected's positional embeddings).
                model_input["position_ids"] = list(range(1, len(chosen_input["input_ids"]) + 1)) + list(
                    range(1, len(rejected_input["input_ids"]) + 1)
                )
                rendered.append(model_input)
            else:
                raise ValueError("No valid messages or chosen_messages/rejected_messages found in sample.")

            for model_input in rendered:
                if "extra_info" in sample:
                    model_input["extra_info"] = sample["extra_info"]
                if "_dataset_name" in sample:
                    model_input["_dataset_name"] = sample["_dataset_name"]
                model_inputs.append(model_input)

        return model_inputs
