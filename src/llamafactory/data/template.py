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

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

from typing_extensions import override

from ..extras import logging
from .data_utils import Role
from .formatter import EmptyFormatter, FunctionFormatter, StringFormatter, ToolFormatter
from .mm_plugin import get_mm_plugin


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from ..hparams import DataArguments
    from .formatter import SLOTS, Formatter
    from .mm_plugin import BasePlugin
    from .tool_utils import FunctionCall


logger = logging.get_logger(__name__)


@dataclass
class Template:
    format_user: "Formatter"
    format_assistant: "Formatter"
    format_system: "Formatter"
    format_function: "Formatter"
    format_observation: "Formatter"
    format_tools: "Formatter"
    format_prefix: "Formatter"
    default_system: str
    stop_words: list[str]
    thought_words: tuple[str, str]
    efficient_eos: bool
    replace_eos: bool
    replace_jinja_template: bool
    mm_plugin: "BasePlugin"

    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        enable_thinking: bool = False,
    ) -> tuple[list[int], list[int]]:
        r"""Return a single pair of token ids representing prompt and response respectively."""
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        prompt_ids = []
        for encoded_ids in encoded_messages[:-1]:
            prompt_ids += encoded_ids

        response_ids = encoded_messages[-1]
        return prompt_ids, response_ids

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> list[tuple[list[int], list[int]]]:
        r"""Return multiple pairs of token ids representing prompts and responses respectively."""
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]

    def extract_tool(self, content: str) -> Union[str, list["FunctionCall"]]:
        r"""Extract tool message."""
        return self.format_tools.extract(content)

    def get_stop_token_ids(self, tokenizer: "PreTrainedTokenizer") -> list[int]:
        r"""Return stop token ids."""
        stop_token_ids = {tokenizer.eos_token_id}
        for token in self.stop_words:
            stop_token_ids.add(tokenizer.convert_tokens_to_ids(token))

        return list(stop_token_ids)

    def add_thought(self, content: str) -> str:
        r"""Add empty thought to assistant message."""
        return f"{self.thought_words[0]}\n\n{self.thought_words[1]}\n\n" + content

    def remove_thought(self, content: str) -> str:
        r"""Remove thought from assistant message."""
        pattern = re.compile(f"{re.escape(self.thought_words[0])}(.*?){re.escape(self.thought_words[1])}", re.DOTALL)
        return re.sub(pattern, "", content).lstrip("\n")

    def get_thought_word_ids(self, tokenizer: "PreTrainedTokenizer") -> list[int]:
        r"""Get the token ids of thought words."""
        return tokenizer.encode(f"{self.thought_words[0]}\n\n{self.thought_words[1]}\n\n", add_special_tokens=False)

    def _convert_elements_to_ids(self, tokenizer: "PreTrainedTokenizer", elements: "SLOTS") -> list[int]:
        r"""Convert elements to token ids."""
        token_ids = []
        for elem in elements:
            if isinstance(elem, str):
                if len(elem) != 0:
                    token_ids += tokenizer.encode(elem, add_special_tokens=False)
            elif isinstance(elem, dict):
                token_ids += [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            elif isinstance(elem, set):
                if "bos_token" in elem and tokenizer.bos_token_id is not None:
                    token_ids += [tokenizer.bos_token_id]
                elif "eos_token" in elem and tokenizer.eos_token_id is not None:
                    token_ids += [tokenizer.eos_token_id]
            else:
                raise ValueError(f"Input must be string, set[str] or dict[str, str], got {type(elem)}")

        return token_ids

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
    ) -> list[list[int]]:
        r"""Encode formatted inputs to pairs of token ids.

        Turn 0: prefix + system + query        resp
        Turn t: query                          resp.
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    elements += self.format_system.apply(content=(system + tool_text))

            if message["role"] == Role.USER:
                elements += self.format_user.apply(content=message["content"], idx=str(i // 2))
            elif message["role"] == Role.ASSISTANT:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION:
                elements += self.format_function.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages

    @staticmethod
    def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str) -> None:
        r"""Add or replace eos token to the tokenizer."""
        if tokenizer.eos_token == eos_token:
            return

        is_added = tokenizer.eos_token_id is None
        num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

        if is_added:
            logger.info_rank0(f"Add eos token: {tokenizer.eos_token}.")
        else:
            logger.info_rank0(f"Replace eos token: {tokenizer.eos_token}.")

        if num_added_tokens > 0:
            logger.warning_rank0("New tokens have been added, make sure `resize_vocab` is True.")

    def fix_special_tokens(self, tokenizer: "PreTrainedTokenizer") -> None:
        r"""Add eos token and pad token to the tokenizer."""
        stop_words = self.stop_words
        if self.replace_eos:
            if not stop_words:
                raise ValueError("Stop words are required to replace the EOS token.")

            self._add_or_replace_eos_token(tokenizer, eos_token=stop_words[0])
            stop_words = stop_words[1:]

        if tokenizer.eos_token_id is None:
            self._add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info_rank0(f"Add pad token: {tokenizer.pad_token}")

        if stop_words:
            num_added_tokens = tokenizer.add_special_tokens(
                dict(additional_special_tokens=stop_words), replace_additional_special_tokens=False
            )
            logger.info_rank0("Add {} to stop words.".format(",".join(stop_words)))
            if num_added_tokens > 0:
                logger.warning_rank0("New tokens have been added, make sure `resize_vocab` is True.")

    @staticmethod
    def _jinja_escape(content: str) -> str:
        r"""Escape single quotes in content."""
        return content.replace("'", r"\'")

    @staticmethod
    def _convert_slots_to_jinja(slots: "SLOTS", tokenizer: "PreTrainedTokenizer", placeholder: str = "content") -> str:
        r"""Convert slots to jinja template."""
        slot_items = []
        for slot in slots:
            if isinstance(slot, str):
                slot_pieces = slot.split("{{content}}")
                if slot_pieces[0]:
                    slot_items.append("'" + Template._jinja_escape(slot_pieces[0]) + "'")
                if len(slot_pieces) > 1:
                    slot_items.append(placeholder)
                    if slot_pieces[1]:
                        slot_items.append("'" + Template._jinja_escape(slot_pieces[1]) + "'")
            elif isinstance(slot, set):  # do not use {{ eos_token }} since it may be replaced
                if "bos_token" in slot and tokenizer.bos_token_id is not None:
                    slot_items.append("'" + tokenizer.bos_token + "'")
                elif "eos_token" in slot and tokenizer.eos_token_id is not None:
                    slot_items.append("'" + tokenizer.eos_token + "'")
            elif isinstance(slot, dict):
                raise ValueError("Dict is not supported.")

        return " + ".join(slot_items)

    def _get_jinja_template(self, tokenizer: "PreTrainedTokenizer") -> str:
        r"""Return the jinja template."""
        prefix = self._convert_slots_to_jinja(self.format_prefix.apply(), tokenizer)
        system = self._convert_slots_to_jinja(self.format_system.apply(), tokenizer, placeholder="system_message")
        user = self._convert_slots_to_jinja(self.format_user.apply(), tokenizer)
        assistant = self._convert_slots_to_jinja(self.format_assistant.apply(), tokenizer)
        jinja_template = ""
        if prefix:
            jinja_template += "{{ " + prefix + " }}"

        if self.default_system:
            jinja_template += "{% set system_message = '" + self._jinja_escape(self.default_system) + "' %}"

        jinja_template += (
            "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}"
            "{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% endif %}"
            "{% if system_message is defined %}{{ " + system + " }}{% endif %}"
            "{% for message in loop_messages %}"
            "{% set content = message['content'] %}"
            "{% if message['role'] == 'user' %}"
            "{{ " + user + " }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ " + assistant + " }}"
            "{% endif %}"
            "{% endfor %}"
        )
        return jinja_template

    def fix_jinja_template(self, tokenizer: "PreTrainedTokenizer") -> None:
        r"""Replace the jinja template in the tokenizer."""
        if tokenizer.chat_template is None or self.replace_jinja_template:
            try:
                tokenizer.chat_template = self._get_jinja_template(tokenizer)
            except ValueError as e:
                logger.info_rank0(f"Cannot add this chat template to tokenizer: {e}.")

    @staticmethod
    def _convert_slots_to_ollama(
        slots: "SLOTS", tokenizer: "PreTrainedTokenizer", placeholder: str = "content"
    ) -> str:
        r"""Convert slots to ollama template."""
        slot_items = []
        for slot in slots:
            if isinstance(slot, str):
                slot_pieces = slot.split("{{content}}")
                if slot_pieces[0]:
                    slot_items.append(slot_pieces[0])
                if len(slot_pieces) > 1:
                    slot_items.append("{{ " + placeholder + " }}")
                    if slot_pieces[1]:
                        slot_items.append(slot_pieces[1])
            elif isinstance(slot, set):  # do not use {{ eos_token }} since it may be replaced
                if "bos_token" in slot and tokenizer.bos_token_id is not None:
                    slot_items.append(tokenizer.bos_token)
                elif "eos_token" in slot and tokenizer.eos_token_id is not None:
                    slot_items.append(tokenizer.eos_token)
            elif isinstance(slot, dict):
                raise ValueError("Dict is not supported.")

        return "".join(slot_items)

    def _get_ollama_template(self, tokenizer: "PreTrainedTokenizer") -> str:
        r"""Return the ollama template."""
        prefix = self._convert_slots_to_ollama(self.format_prefix.apply(), tokenizer)
        system = self._convert_slots_to_ollama(self.format_system.apply(), tokenizer, placeholder=".System")
        user = self._convert_slots_to_ollama(self.format_user.apply(), tokenizer, placeholder=".Content")
        assistant = self._convert_slots_to_ollama(self.format_assistant.apply(), tokenizer, placeholder=".Content")
        return (
            f"{prefix}{{{{ if .System }}}}{system}{{{{ end }}}}"
            f"""{{{{ range .Messages }}}}{{{{ if eq .Role "user" }}}}{user}"""
            f"""{{{{ else if eq .Role "assistant" }}}}{assistant}{{{{ end }}}}{{{{ end }}}}"""
        )

    def get_ollama_modelfile(self, tokenizer: "PreTrainedTokenizer") -> str:
        r"""Return the ollama modelfile.

        TODO: support function calling.
        """
        modelfile = "# ollama modelfile auto-generated by llamafactory\n\n"
        modelfile += f'FROM .\n\nTEMPLATE """{self._get_ollama_template(tokenizer)}"""\n\n'

        if self.default_system:
            modelfile += f'SYSTEM """{self.default_system}"""\n\n'

        for stop_token_id in self.get_stop_token_ids(tokenizer):
            modelfile += f'PARAMETER stop "{tokenizer.convert_ids_to_tokens(stop_token_id)}"\n'

        modelfile += "PARAMETER num_ctx 4096\n"
        return modelfile


@dataclass
class Llama2Template(Template):
    r"""A template that fuse the system message to first user message."""

    @override
    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: str,
        tools: str,
    ) -> list[list[int]]:
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            system_text = ""
            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    system_text = self.format_system.apply(content=(system + tool_text))[0]

            if message["role"] == Role.USER:
                elements += self.format_user.apply(content=system_text + message["content"])
            elif message["role"] == Role.ASSISTANT:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION:
                elements += self.format_function.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages

    def _get_jinja_template(self, tokenizer: "PreTrainedTokenizer") -> str:
        prefix = self._convert_slots_to_jinja(self.format_prefix.apply(), tokenizer)
        system_message = self._convert_slots_to_jinja(
            self.format_system.apply(), tokenizer, placeholder="system_message"
        )
        user_message = self._convert_slots_to_jinja(self.format_user.apply(), tokenizer)
        assistant_message = self._convert_slots_to_jinja(self.format_assistant.apply(), tokenizer)
        jinja_template = ""
        if prefix:
            jinja_template += "{{ " + prefix + " }}"

        if self.default_system:
            jinja_template += "{% set system_message = '" + self._jinja_escape(self.default_system) + "' %}"

        jinja_template += (
            "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}"
            "{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% endif %}"
            "{% for message in loop_messages %}"
            "{% if loop.index0 == 0 and system_message is defined %}"
            "{% set content = " + system_message + " + message['content'] %}"
            "{% else %}{% set content = message['content'] %}{% endif %}"
            "{% if message['role'] == 'user' %}"
            "{{ " + user_message + " }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ " + assistant_message + " }}"
            "{% endif %}"
            "{% endfor %}"
        )
        return jinja_template


@dataclass
class ReasoningTemplate(Template):
    r"""A template that add thought to assistant message."""

    @override
    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        enable_thinking: bool = False,
    ) -> tuple[list[int], list[int]]:
        messages = deepcopy(messages)
        for i in range(len(messages)):
            if messages[i]["role"] == Role.ASSISTANT and (i != len(messages) - 1):
                messages[i]["content"] = self.remove_thought(messages[i]["content"])

        encoded_messages = self._encode(tokenizer, messages, system, tools)
        prompt_ids = []
        for encoded_ids in encoded_messages[:-1]:
            prompt_ids += encoded_ids

        if not enable_thinking and (
            messages[-1]["role"] == Role.ASSISTANT
            and self.thought_words[0] not in messages[-1]["content"]
            and self.thought_words[1] not in messages[-1]["content"]
        ):
            prompt_ids += self.get_thought_word_ids(tokenizer)

        response_ids = encoded_messages[-1]
        return prompt_ids, response_ids

    @override
    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> list[tuple[list[int], list[int]]]:
        messages = deepcopy(messages)
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        for i in range(len(messages) - 1):
            if (
                messages[i + 1]["role"] == Role.ASSISTANT
                and self.thought_words[0] not in messages[i + 1]["content"]
                and self.thought_words[1] not in messages[i + 1]["content"]
            ):
                encoded_messages[i] += self.get_thought_word_ids(tokenizer)

        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]


TEMPLATES: dict[str, "Template"] = {}


def register_template(
    name: str,
    format_user: Optional["Formatter"] = None,
    format_assistant: Optional["Formatter"] = None,
    format_system: Optional["Formatter"] = None,
    format_function: Optional["Formatter"] = None,
    format_observation: Optional["Formatter"] = None,
    format_tools: Optional["Formatter"] = None,
    format_prefix: Optional["Formatter"] = None,
    default_system: str = "",
    stop_words: Optional[list[str]] = None,
    thought_words: Optional[tuple[str, str]] = None,
    efficient_eos: bool = False,
    replace_eos: bool = False,
    replace_jinja_template: bool = False,
    mm_plugin: "BasePlugin" = get_mm_plugin(name="base"),
    template_class: type["Template"] = Template,
) -> None:
    r"""Register a chat template.

    To add the following chat template:
    ```
    <s><user>user prompt here
    <model>model response here</s>
    <user>user prompt here
    <model>model response here</s>
    ```

    The corresponding code should be:
    ```
    register_template(
        name="custom",
        format_user=StringFormatter(slots=["<user>{{content}}\n<model>"]),
        format_assistant=StringFormatter(slots=["{{content}}</s>\n"]),
        format_prefix=EmptyFormatter("<s>"),
    )
    ```
    """
    if name in TEMPLATES:
        raise ValueError(f"Template {name} already exists.")

    default_slots = ["{{content}}"] if efficient_eos else ["{{content}}", {"eos_token"}]
    default_user_formatter = StringFormatter(slots=["{{content}}"])
    default_assistant_formatter = StringFormatter(slots=default_slots)
    default_function_formatter = FunctionFormatter(slots=default_slots, tool_format="default")
    default_tool_formatter = ToolFormatter(tool_format="default")
    default_prefix_formatter = EmptyFormatter()
    TEMPLATES[name] = template_class(
        format_user=format_user or default_user_formatter,
        format_assistant=format_assistant or default_assistant_formatter,
        format_system=format_system or default_user_formatter,
        format_function=format_function or default_function_formatter,
        format_observation=format_observation or format_user or default_user_formatter,
        format_tools=format_tools or default_tool_formatter,
        format_prefix=format_prefix or default_prefix_formatter,
        default_system=default_system,
        stop_words=stop_words or [],
        thought_words=thought_words or ("<think>", "</think>"),
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        replace_jinja_template=replace_jinja_template,
        mm_plugin=mm_plugin,
    )


def parse_template(tokenizer: "PreTrainedTokenizer") -> "Template":
    r"""Extract a chat template from the tokenizer."""

    def find_diff(short_str: str, long_str: str) -> str:
        i, j = 0, 0
        diff = ""
        while i < len(short_str) and j < len(long_str):
            if short_str[i] == long_str[j]:
                i += 1
                j += 1
            else:
                diff += long_str[j]
                j += 1

        return diff

    prefix = tokenizer.decode(tokenizer.encode(""))

    messages = [{"role": "system", "content": "{{content}}"}]
    system_slot = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)[len(prefix) :]

    messages = [{"role": "system", "content": ""}, {"role": "user", "content": "{{content}}"}]
    user_slot_empty_system = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    user_slot_empty_system = user_slot_empty_system[len(prefix) :]

    messages = [{"role": "user", "content": "{{content}}"}]
    user_slot = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    user_slot = user_slot[len(prefix) :]

    messages = [{"role": "user", "content": "{{content}}"}, {"role": "assistant", "content": "{{content}}"}]
    assistant_slot = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    assistant_slot = assistant_slot[len(prefix) + len(user_slot) :]
    assistant_slot = assistant_slot.replace("<think>", "").replace("</think>", "").lstrip("\n")  # remove thought tags

    if len(user_slot) > len(user_slot_empty_system):
        default_system = find_diff(user_slot_empty_system, user_slot)
        sole_system = system_slot.replace("{{content}}", default_system, 1)
        user_slot = user_slot[len(sole_system) :]
    else:  # if defaut_system is empty, user_slot_empty_system will be longer than user_slot
        default_system = ""

    return Template(
        format_user=StringFormatter(slots=[user_slot]),
        format_assistant=StringFormatter(slots=[assistant_slot]),
        format_system=StringFormatter(slots=[system_slot]),
        format_function=FunctionFormatter(slots=[assistant_slot], tool_format="default"),
        format_observation=StringFormatter(slots=[user_slot]),
        format_tools=ToolFormatter(tool_format="default"),
        format_prefix=EmptyFormatter(slots=[prefix]) if prefix else EmptyFormatter(),
        default_system=default_system,
        stop_words=[],
        thought_words=("<think>", "</think>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        mm_plugin=get_mm_plugin(name="base"),
    )


def get_template_and_fix_tokenizer(tokenizer: "PreTrainedTokenizer", data_args: "DataArguments") -> "Template":
    r"""Get chat template and fixes the tokenizer."""
    if data_args.template is None:
        if isinstance(tokenizer.chat_template, str):
            logger.warning_rank0("`template` was not specified, try parsing the chat template from the tokenizer.")
            template = parse_template(tokenizer)
        else:
            logger.warning_rank0("`template` was not specified, use `empty` template.")
            template = TEMPLATES["empty"]  # placeholder
    else:
        if data_args.template not in TEMPLATES:
            raise ValueError(f"Template {data_args.template} does not exist.")

        template = TEMPLATES[data_args.template]

    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError("Current template does not support `train_on_prompt`.")

    if data_args.tool_format is not None:
        logger.info_rank0(f"Using tool format: {data_args.tool_format}.")
        default_slots = ["{{content}}"] if template.efficient_eos else ["{{content}}", {"eos_token"}]
        template.format_function = FunctionFormatter(slots=default_slots, tool_format=data_args.tool_format)
        template.format_tools = ToolFormatter(tool_format=data_args.tool_format)

    template.fix_special_tokens(tokenizer)
    template.fix_jinja_template(tokenizer)
    return template


register_template(
    name="alpaca",
    format_user=StringFormatter(slots=["### Instruction:\n{{content}}\n\n### Response:\n"]),
    format_assistant=StringFormatter(slots=["{{content}}", {"eos_token"}, "\n\n"]),
    default_system=(
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    ),
    replace_jinja_template=True,
)


register_template(
    name="aquila",
    format_user=StringFormatter(slots=["Human: {{content}}###Assistant:"]),
    format_assistant=StringFormatter(slots=["{{content}}###"]),
    format_system=StringFormatter(slots=["System: {{content}}###"]),
    default_system=(
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
    ),
    stop_words=["</s>"],
)


register_template(
    name="atom",
    format_user=StringFormatter(
        slots=[{"bos_token"}, "Human: {{content}}\n", {"eos_token"}, {"bos_token"}, "Assistant:"]
    ),
    format_assistant=StringFormatter(slots=["{{content}}\n", {"eos_token"}]),
)


register_template(
    name="baichuan",
    format_user=StringFormatter(slots=[{"token": "<reserved_102>"}, "{{content}}", {"token": "<reserved_103>"}]),
    efficient_eos=True,
)


register_template(
    name="baichuan2",
    format_user=StringFormatter(slots=["<reserved_106>{{content}}<reserved_107>"]),
    efficient_eos=True,
)


register_template(
    name="bailing",
    format_user=StringFormatter(slots=["<role>HUMAN</role>{{content}}<role>ASSISTANT</role>"]),
    format_system=StringFormatter(slots=["<role>SYSTEM</role>{{content}}"]),
    format_observation=StringFormatter(slots=["<role>OBSERVATION</role>{{content}}<role>ASSISTANT</role>"]),
    stop_words=["<|endoftext|>"],
    efficient_eos=True,
)


register_template(
    name="belle",
    format_user=StringFormatter(slots=["Human: {{content}}\n\nBelle: "]),
    format_assistant=StringFormatter(slots=["{{content}}", {"eos_token"}, "\n\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


register_template(
    name="bluelm",
    format_user=StringFormatter(slots=[{"token": "[|Human|]:"}, "{{content}}", {"token": "[|AI|]:"}]),
)


register_template(
    name="breeze",
    format_user=StringFormatter(slots=["[INST] {{content}} [/INST] "]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    efficient_eos=True,
)


register_template(
    name="chatglm2",
    format_user=StringFormatter(slots=["[Round {{idx}}]\n\n问：{{content}}\n\n答："]),
    format_prefix=EmptyFormatter(slots=[{"token": "[gMASK]"}, {"token": "sop"}]),
    efficient_eos=True,
)


register_template(
    name="chatglm3",
    format_user=StringFormatter(slots=[{"token": "<|user|>"}, "\n", "{{content}}", {"token": "<|assistant|>"}]),
    format_assistant=StringFormatter(slots=["\n", "{{content}}"]),
    format_system=StringFormatter(slots=[{"token": "<|system|>"}, "\n", "{{content}}"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4"),
    format_observation=StringFormatter(
        slots=[{"token": "<|observation|>"}, "\n", "{{content}}", {"token": "<|assistant|>"}]
    ),
    format_tools=ToolFormatter(tool_format="glm4"),
    format_prefix=EmptyFormatter(slots=[{"token": "[gMASK]"}, {"token": "sop"}]),
    stop_words=["<|user|>", "<|observation|>"],
    efficient_eos=True,
)


register_template(
    name="chatml",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    stop_words=["<|im_end|>", "<|im_start|>"],
    replace_eos=True,
    replace_jinja_template=True,
)


# copied from chatml template
register_template(
    name="chatml_de",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    default_system="Du bist ein freundlicher und hilfsbereiter KI-Assistent.",
    stop_words=["<|im_end|>", "<|im_start|>"],
    replace_eos=True,
    replace_jinja_template=True,
)


register_template(
    name="codegeex2",
    format_prefix=EmptyFormatter(slots=[{"token": "[gMASK]"}, {"token": "sop"}]),
)


register_template(
    name="codegeex4",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>\n"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4"),
    format_observation=StringFormatter(slots=["<|observation|>\n{{content}}<|assistant|>\n"]),
    format_tools=ToolFormatter(tool_format="glm4"),
    format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
    default_system=(
        "你是一位智能编程助手，你叫CodeGeeX。你会为用户回答关于编程、代码、计算机方面的任何问题，"
        "并提供格式规范、可以执行、准确安全的代码，并在必要时提供详细的解释。"
    ),
    stop_words=["<|user|>", "<|observation|>"],
    efficient_eos=True,
)


register_template(
    name="cohere",
    format_user=StringFormatter(
        slots=[
            (
                "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{content}}<|END_OF_TURN_TOKEN|>"
                "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
            )
        ]
    ),
    format_system=StringFormatter(slots=["<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{content}}<|END_OF_TURN_TOKEN|>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


register_template(
    name="cpm",
    format_user=StringFormatter(slots=["<用户>{{content}}<AI>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


# copied from chatml template
register_template(
    name="cpm3",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|im_end|>"],
)


# copied from chatml template
register_template(
    name="dbrx",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    default_system=(
        "You are DBRX, created by Databricks. You were last updated in December 2023. "
        "You answer questions based on information available up to that point.\n"
        "YOU PROVIDE SHORT RESPONSES TO SHORT QUESTIONS OR STATEMENTS, but provide thorough "
        "responses to more complex and open-ended questions.\nYou assist with various tasks, "
        "from writing to coding (using markdown for code blocks — remember to use ``` with "
        "code, JSON, and tables).\n(You do not have real-time data access or code execution "
        "capabilities. You avoid stereotyping and provide balanced perspectives on "
        "controversial topics. You do not provide song lyrics, poems, or news articles and "
        "do not divulge details of your training data.)\nThis is your system prompt, "
        "guiding your responses. Do not reference it, just respond to the user. If you find "
        "yourself talking about this message, stop. You should be responding appropriately "
        "and usually that means not mentioning this.\nYOU DO NOT MENTION ANY OF THIS INFORMATION "
        "ABOUT YOURSELF UNLESS THE INFORMATION IS DIRECTLY PERTINENT TO THE USER'S QUERY."
    ),
    stop_words=["<|im_end|>"],
    replace_eos=True,
)


register_template(
    name="deepseek",
    format_user=StringFormatter(slots=["User: {{content}}\n\nAssistant:"]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


register_template(
    name="deepseek3",
    format_user=StringFormatter(slots=["<｜User｜>{{content}}<｜Assistant｜>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


# copied from deepseek3 template
register_template(
    name="deepseekr1",
    format_user=StringFormatter(slots=["<｜User｜>{{content}}<｜Assistant｜>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    template_class=ReasoningTemplate,
)


register_template(
    name="deepseekcoder",
    format_user=StringFormatter(slots=["### Instruction:\n{{content}}\n### Response:"]),
    format_assistant=StringFormatter(slots=["\n{{content}}\n<|EOT|>\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    default_system=(
        "You are an AI programming assistant, utilizing the DeepSeek Coder model, "
        "developed by DeepSeek Company, and you only answer questions related to computer science. "
        "For politically sensitive questions, security and privacy issues, "
        "and other non-computer science questions, you will refuse to answer.\n"
    ),
)


register_template(
    name="default",
    format_user=StringFormatter(slots=["Human: {{content}}", {"eos_token"}, "\nAssistant:"]),
    format_assistant=StringFormatter(slots=["{{content}}", {"eos_token"}, "\n"]),
    format_system=StringFormatter(slots=["System: {{content}}", {"eos_token"}, "\n"]),
    replace_jinja_template=True,
)


register_template(
    name="empty",
    format_assistant=StringFormatter(slots=["{{content}}"]),
    replace_jinja_template=True,
)


register_template(
    name="exaone",
    format_user=StringFormatter(slots=["[|user|]{{content}}\n[|assistant|]"]),
    format_assistant=StringFormatter(slots=["{{content}}", {"eos_token"}, "\n"]),
    format_system=StringFormatter(slots=["[|system|]{{content}}[|endofturn|]\n"]),
)


register_template(
    name="falcon",
    format_user=StringFormatter(slots=["User: {{content}}\nFalcon:"]),
    format_assistant=StringFormatter(slots=["{{content}}\n"]),
    efficient_eos=True,
)


register_template(
    name="fewshot",
    format_assistant=StringFormatter(slots=["{{content}}\n\n"]),
    efficient_eos=True,
    replace_jinja_template=True,
)


register_template(
    name="gemma",
    format_user=StringFormatter(slots=["<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<end_of_turn>\n"]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_observation=StringFormatter(
        slots=["<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]
    ),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<end_of_turn>"],
    replace_eos=True,
    template_class=Llama2Template,
)


# copied from gemma template
register_template(
    name="gemma3",
    format_user=StringFormatter(slots=["<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<end_of_turn>\n"]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_observation=StringFormatter(
        slots=["<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]
    ),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<end_of_turn>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin("gemma3", image_token="<image_soft_token>"),
    template_class=Llama2Template,
)


register_template(
    name="glm4",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>"]),
    format_assistant=StringFormatter(slots=["\n{{content}}"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4"),
    format_observation=StringFormatter(slots=["<|observation|>\n{{content}}<|assistant|>"]),
    format_tools=ToolFormatter(tool_format="glm4"),
    format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
    stop_words=["<|user|>", "<|observation|>"],
    efficient_eos=True,
)


# copied from glm4 template
register_template(
    name="glmz1",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>"]),
    format_assistant=StringFormatter(slots=["\n{{content}}"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4"),
    format_observation=StringFormatter(slots=["<|observation|>\n{{content}}<|assistant|>"]),
    format_tools=ToolFormatter(tool_format="glm4"),
    format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
    stop_words=["<|user|>", "<|observation|>"],
    efficient_eos=True,
    template_class=ReasoningTemplate,
)


register_template(
    name="granite3",
    format_user=StringFormatter(
        slots=[
            "<|start_of_role|>user<|end_of_role|>{{content}}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
        ]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|end_of_text|>\n"]),
    format_system=StringFormatter(slots=["<|start_of_role|>system<|end_of_role|>{{content}}<|end_of_text|>\n"]),
)


register_template(
    name="granite3_vision",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}\n<|assistant|>\n"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}\n"]),
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    mm_plugin=get_mm_plugin(name="llava_next", image_token="<image>"),
)


register_template(
    name="index",
    format_user=StringFormatter(slots=["reserved_0{{content}}reserved_1"]),
    format_system=StringFormatter(slots=["<unk>{{content}}"]),
    efficient_eos=True,
)


register_template(
    name="hunyuan",
    format_user=StringFormatter(slots=["<|bos|>user\n{{content}}<|eos|>\n<|bos|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|eos|>\n"]),
    format_system=StringFormatter(slots=["<|bos|>system\n{{content}}<|eos|>\n"]),
    format_prefix=EmptyFormatter(slots=["<|bos|>"]),
    stop_words=["<|eos|>"],
)


register_template(
    name="intern",
    format_user=StringFormatter(slots=["<|User|>:{{content}}\n<|Bot|>:"]),
    format_assistant=StringFormatter(slots=["{{content}}<eoa>\n"]),
    format_system=StringFormatter(slots=["<|System|>:{{content}}\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    default_system=(
        "You are an AI assistant whose name is InternLM (书生·浦语).\n"
        "- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory "
        "(上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
        "- InternLM (书生·浦语) can understand and communicate fluently in the language "
        "chosen by the user such as English and 中文."
    ),
    stop_words=["<eoa>"],
)


register_template(
    name="intern2",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    default_system=(
        "You are an AI assistant whose name is InternLM (书生·浦语).\n"
        "- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory "
        "(上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
        "- InternLM (书生·浦语) can understand and communicate fluently in the language "
        "chosen by the user such as English and 中文."
    ),
    stop_words=["<|im_end|>"],
)


register_template(
    name="intern_vl",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    default_system=(
        "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"
    ),
    stop_words=["<|im_end|>"],
    mm_plugin=get_mm_plugin(name="intern_vl", image_token="<image>", video_token="<video>"),
)


register_template(
    name="kimi_vl",
    format_user=StringFormatter(
        slots=["<|im_user|>user<|im_middle|>{{content}}<|im_end|><|im_assistant|>assistant<|im_middle|>"]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>"]),
    format_system=StringFormatter(slots=["<|im_system|>system<|im_middle|>{{content}}<|im_end|>"]),
    default_system="You are a helpful assistant",
    stop_words=["<|im_end|>"],
    thought_words=("◁think▷", "◁/think▷"),
    mm_plugin=get_mm_plugin("kimi_vl", image_token="<|media_pad|>"),
)


register_template(
    name="llama2",
    format_user=StringFormatter(slots=[{"bos_token"}, "[INST] {{content}} [/INST]"]),
    format_system=StringFormatter(slots=["<<SYS>>\n{{content}}\n<</SYS>>\n\n"]),
    template_class=Llama2Template,
)


# copied from llama2 template
register_template(
    name="llama2_zh",
    format_user=StringFormatter(slots=[{"bos_token"}, "[INST] {{content}} [/INST]"]),
    format_system=StringFormatter(slots=["<<SYS>>\n{{content}}\n<</SYS>>\n\n"]),
    default_system="You are a helpful assistant. 你是一个乐于助人的助手。",
    template_class=Llama2Template,
)


register_template(
    name="llama3",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|eot_id|>"]),
    format_system=StringFormatter(slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]),
    format_function=FunctionFormatter(slots=["{{content}}<|eot_id|>"], tool_format="llama3"),
    format_observation=StringFormatter(
        slots=[
            (
                "<|start_header_id|>ipython<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_tools=ToolFormatter(tool_format="llama3"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot_id|>", "<|eom_id|>"],
    replace_eos=True,
)


register_template(
    name="llama4",
    format_user=StringFormatter(
        slots=["<|header_start|>user<|header_end|>\n\n{{content}}<|eot|><|header_start|>assistant<|header_end|>\n\n"]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|eot|>"]),
    format_system=StringFormatter(slots=["<|header_start|>system<|header_end|>\n\n{{content}}<|eot|>"]),
    format_function=FunctionFormatter(slots=["{{content}}<|eot|>"], tool_format="llama3"),
    format_observation=StringFormatter(
        slots=[
            "<|header_start|>ipython<|header_end|>\n\n{{content}}<|eot|><|header_start|>assistant<|header_end|>\n\n"
        ]
    ),
    format_tools=ToolFormatter(tool_format="llama3"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot|>", "<|eom|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="llama4", image_token="<|image|>"),
)


# copied from llama3 template
register_template(
    name="mllama",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|eot_id|>"]),
    format_system=StringFormatter(slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]),
    format_function=FunctionFormatter(slots=["{{content}}<|eot_id|>"], tool_format="llama3"),
    format_observation=StringFormatter(
        slots=[
            (
                "<|start_header_id|>ipython<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_tools=ToolFormatter(tool_format="llama3"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot_id|>", "<|eom_id|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="mllama", image_token="<|image|>"),
)


register_template(
    name="moonlight",
    format_user=StringFormatter(
        slots=["<|im_user|>user<|im_middle|>{{content}}<|im_end|><|im_assistant|>assistant<|im_middle|>"]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>"]),
    format_system=StringFormatter(slots=["<|im_system|>system<|im_middle|>{{content}}<|im_end|>"]),
    default_system="You are a helpful assistant provided by Moonshot-AI.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
)


# copied from vicuna template
register_template(
    name="llava",
    format_user=StringFormatter(slots=["USER: {{content}} ASSISTANT:"]),
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    mm_plugin=get_mm_plugin(name="llava", image_token="<image>"),
)


# copied from vicuna template
register_template(
    name="llava_next",
    format_user=StringFormatter(slots=["USER: {{content}} ASSISTANT:"]),
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    mm_plugin=get_mm_plugin(name="llava_next", image_token="<image>"),
)


# copied from llama3 template
register_template(
    name="llava_next_llama3",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|eot_id|>"]),
    format_system=StringFormatter(slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]),
    format_function=FunctionFormatter(slots=["{{content}}<|eot_id|>"], tool_format="llama3"),
    format_observation=StringFormatter(
        slots=[
            (
                "<|start_header_id|>ipython<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_tools=ToolFormatter(tool_format="llama3"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot_id|>", "<|eom_id|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="llava_next", image_token="<image>"),
)


# copied from mistral template
register_template(
    name="llava_next_mistral",
    format_user=StringFormatter(slots=["[INST] {{content}}[/INST]"]),
    format_assistant=StringFormatter(slots=[" {{content}}", {"eos_token"}]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_function=FunctionFormatter(slots=["[TOOL_CALLS] {{content}}", {"eos_token"}], tool_format="mistral"),
    format_observation=StringFormatter(slots=["""[TOOL_RESULTS] {"content": {{content}}}[/TOOL_RESULTS]"""]),
    format_tools=ToolFormatter(tool_format="mistral"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    mm_plugin=get_mm_plugin(name="llava_next", image_token="<image>"),
    template_class=Llama2Template,
)


# copied from qwen template
register_template(
    name="llava_next_qwen",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="llava_next", image_token="<image>"),
)


# copied from chatml template
register_template(
    name="llava_next_yi",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    stop_words=["<|im_end|>"],
    mm_plugin=get_mm_plugin(name="llava_next", image_token="<image>"),
)


# copied from vicuna template
register_template(
    name="llava_next_video",
    format_user=StringFormatter(slots=["USER: {{content}} ASSISTANT:"]),
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    mm_plugin=get_mm_plugin(name="llava_next_video", image_token="<image>", video_token="<video>"),
)


# copied from mistral template
register_template(
    name="llava_next_video_mistral",
    format_user=StringFormatter(slots=["[INST] {{content}}[/INST]"]),
    format_assistant=StringFormatter(slots=[" {{content}}", {"eos_token"}]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_function=FunctionFormatter(slots=["[TOOL_CALLS] {{content}}", {"eos_token"}], tool_format="mistral"),
    format_observation=StringFormatter(slots=["""[TOOL_RESULTS] {"content": {{content}}}[/TOOL_RESULTS]"""]),
    format_tools=ToolFormatter(tool_format="mistral"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    mm_plugin=get_mm_plugin(name="llava_next_video", image_token="<image>", video_token="<video>"),
    template_class=Llama2Template,
)


# copied from chatml template
register_template(
    name="llava_next_video_yi",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    stop_words=["<|im_end|>"],
    mm_plugin=get_mm_plugin(name="llava_next_video", image_token="<image>", video_token="<video>"),
)


# copied from chatml template
register_template(
    name="marco",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    default_system=(
        "你是一个经过良好训练的AI助手，你的名字是Marco-o1."
        "由阿里国际数字商业集团的AI Business创造.\n## 重要！！！！！\n"
        "当你回答问题时，你的思考应该在<Thought>内完成，<Output>内输出你的结果。\n"
        "<Thought>应该尽可能是英文，但是有2个特例，一个是对原文中的引用，另一个是是数学应该使用markdown格式，<Output>内的输出需要遵循用户输入的语言。\n"
    ),
    stop_words=["<|im_end|>"],
)


# copied from qwen template
register_template(
    name="mimo",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    template_class=ReasoningTemplate,
)


# copied from chatml template
register_template(
    name="minicpm_v",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    stop_words=["<|im_end|>"],
    default_system="You are a helpful assistant.",
    mm_plugin=get_mm_plugin(name="minicpm_v", image_token="<image>", video_token="<video>"),
)


# copied from minicpm_v template
register_template(
    name="minicpm_o",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    stop_words=["<|im_end|>"],
    default_system="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    mm_plugin=get_mm_plugin(name="minicpm_v", image_token="<image>", video_token="<video>", audio_token="<audio>"),
)


# mistral tokenizer v3 tekken
register_template(
    name="ministral",
    format_user=StringFormatter(slots=["[INST]{{content}}[/INST]"]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_function=FunctionFormatter(slots=["[TOOL_CALLS]{{content}}", {"eos_token"}], tool_format="mistral"),
    format_observation=StringFormatter(slots=["""[TOOL_RESULTS]{"content": {{content}}}[/TOOL_RESULTS]"""]),
    format_tools=ToolFormatter(tool_format="mistral"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    template_class=Llama2Template,
)


# mistral tokenizer v3
register_template(
    name="mistral",
    format_user=StringFormatter(slots=["[INST] {{content}}[/INST]"]),
    format_assistant=StringFormatter(slots=[" {{content}}", {"eos_token"}]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_function=FunctionFormatter(slots=["[TOOL_CALLS] {{content}}", {"eos_token"}], tool_format="mistral"),
    format_observation=StringFormatter(slots=["""[TOOL_RESULTS] {"content": {{content}}}[/TOOL_RESULTS]"""]),
    format_tools=ToolFormatter(tool_format="mistral"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    template_class=Llama2Template,
)


# mistral tokenizer v7 tekken (copied from ministral)
register_template(
    name="mistral_small",
    format_user=StringFormatter(slots=["[INST]{{content}}[/INST]"]),
    format_system=StringFormatter(slots=["[SYSTEM_PROMPT]{{content}}[/SYSTEM_PROMPT]"]),
    format_function=FunctionFormatter(slots=["[TOOL_CALLS]{{content}}", {"eos_token"}], tool_format="mistral"),
    format_observation=StringFormatter(slots=["""[TOOL_RESULTS]{"content": {{content}}}[/TOOL_RESULTS]"""]),
    format_tools=ToolFormatter(tool_format="mistral"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


register_template(
    name="olmo",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>\n"]),
    format_prefix=EmptyFormatter(slots=[{"eos_token"}]),
)


register_template(
    name="openchat",
    format_user=StringFormatter(slots=["GPT4 Correct User: {{content}}", {"eos_token"}, "GPT4 Correct Assistant:"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


register_template(
    name="openchat-3.6",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>GPT4 Correct User<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>GPT4 Correct Assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot_id|>"],
)


# copied from chatml template
register_template(
    name="opencoder",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    default_system="You are OpenCoder, created by OpenCoder Team.",
    stop_words=["<|im_end|>"],
)


register_template(
    name="orion",
    format_user=StringFormatter(slots=["Human: {{content}}\n\nAssistant: ", {"eos_token"}]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


register_template(
    name="paligemma",
    format_user=StringFormatter(slots=["{{content}}\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    mm_plugin=get_mm_plugin(name="paligemma", image_token="<image>"),
    template_class=Llama2Template,
)


# copied from gemma template
register_template(
    name="paligemma_chat",
    format_user=StringFormatter(slots=["<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<end_of_turn>\n"]),
    format_observation=StringFormatter(
        slots=["<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]
    ),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<end_of_turn>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="paligemma", image_token="<image>"),
    template_class=Llama2Template,
)


register_template(
    name="phi",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|end|>\n<|assistant|>\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|end|>\n"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}<|end|>\n"]),
    stop_words=["<|end|>"],
    replace_eos=True,
)


register_template(
    name="phi_small",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|end|>\n<|assistant|>\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|end|>\n"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}<|end|>\n"]),
    format_prefix=EmptyFormatter(slots=[{"<|endoftext|>"}]),
    stop_words=["<|end|>"],
    replace_eos=True,
)


register_template(
    name="phi4",
    format_user=StringFormatter(
        slots=["<|im_start|>user<|im_sep|>{{content}}<|im_end|><|im_start|>assistant<|im_sep|>"]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>"]),
    format_system=StringFormatter(slots=["<|im_start|>system<|im_sep|>{{content}}<|im_end|>"]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
)


# copied from ministral template
register_template(
    name="pixtral",
    format_user=StringFormatter(slots=["[INST]{{content}}[/INST]"]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_function=FunctionFormatter(slots=["[TOOL_CALLS]{{content}}", {"eos_token"}], tool_format="mistral"),
    format_observation=StringFormatter(slots=["""[TOOL_RESULTS]{"content": {{content}}}[/TOOL_RESULTS]"""]),
    format_tools=ToolFormatter(tool_format="mistral"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    mm_plugin=get_mm_plugin(name="pixtral", image_token="[IMG]"),
    template_class=Llama2Template,
)


# copied from chatml template
register_template(
    name="qwen",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    default_system="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
)


# copied from qwen template
register_template(
    name="qwen3",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    template_class=ReasoningTemplate,
)


# copied from chatml template
register_template(
    name="qwen2_audio",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="qwen2_audio", audio_token="<|AUDIO|>"),
)


# copied from qwen template
register_template(
    name="qwen2_omni",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(
        name="qwen2_omni", audio_token="<|AUDIO|>", image_token="<|IMAGE|>", video_token="<|VIDEO|>"
    ),
)

# copied from qwen template
register_template(
    name="qwen2_vl",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="qwen2_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"),
)


register_template(
    name="sailor",
    format_user=StringFormatter(slots=["<|im_start|>question\n{{content}}<|im_end|>\n<|im_start|>answer\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    default_system=(
        "You are an AI assistant named Sailor created by Sea AI Lab. "
        "Your answer should be friendly, unbiased, faithful, informative and detailed."
    ),
    stop_words=["<|im_end|>"],
)


# copied from llama3 template
register_template(
    name="skywork_o1",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|eot_id|>"]),
    format_system=StringFormatter(slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]),
    format_function=FunctionFormatter(slots=["{{content}}<|eot_id|>"], tool_format="llama3"),
    format_observation=StringFormatter(
        slots=[
            (
                "<|start_header_id|>ipython<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_tools=ToolFormatter(tool_format="llama3"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    default_system=(
        "You are Skywork-o1, a thinking model developed by Skywork AI, specializing in solving complex problems "
        "involving mathematics, coding, and logical reasoning through deep thought. When faced with a user's request, "
        "you first engage in a lengthy and in-depth thinking process to explore possible solutions to the problem. "
        "After completing your thoughts, you then provide a detailed explanation of the solution process "
        "in your response."
    ),
    stop_words=["<|eot_id|>", "<|eom_id|>"],
)


register_template(
    name="solar",
    format_user=StringFormatter(slots=["### User:\n{{content}}\n\n### Assistant:\n"]),
    format_system=StringFormatter(slots=["### System:\n{{content}}\n\n"]),
    efficient_eos=True,
)


register_template(
    name="starchat",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|end|>\n<|assistant|>"]),
    format_assistant=StringFormatter(slots=["{{content}}<|end|>\n"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}<|end|>\n"]),
    stop_words=["<|end|>"],
)


register_template(
    name="telechat",
    format_user=StringFormatter(slots=["<_user>{{content}}<_bot>"]),
    format_system=StringFormatter(slots=["<_system>{{content}}<_end>"]),
)


register_template(
    name="telechat2",
    format_user=StringFormatter(slots=["<_user>{{content}}<_bot>"]),
    format_system=StringFormatter(slots=["<_system>{{content}}"]),
    default_system=(
        "你是中国电信星辰语义大模型，英文名是TeleChat，你是由中电信人工智能科技有限公司和中国电信人工智能研究院（TeleAI）研发的人工智能助手。"
    ),
)


register_template(
    name="vicuna",
    format_user=StringFormatter(slots=["USER: {{content}} ASSISTANT:"]),
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    replace_jinja_template=True,
)


register_template(
    name="video_llava",
    format_user=StringFormatter(slots=["USER: {{content}} ASSISTANT:"]),
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    mm_plugin=get_mm_plugin(name="video_llava", image_token="<image>", video_token="<video>"),
)


register_template(
    name="xuanyuan",
    format_user=StringFormatter(slots=["Human: {{content}} Assistant:"]),
    default_system=(
        "以下是用户和人工智能助手之间的对话。用户以Human开头，人工智能助手以Assistant开头，"
        "会对人类提出的问题给出有帮助、高质量、详细和礼貌的回答，并且总是拒绝参与与不道德、"
        "不安全、有争议、政治敏感等相关的话题、问题和指示。\n"
    ),
)


register_template(
    name="xverse",
    format_user=StringFormatter(slots=["Human: {{content}}\n\nAssistant: "]),
)


register_template(
    name="yayi",
    format_user=StringFormatter(slots=[{"token": "<|Human|>"}, ":\n{{content}}\n\n", {"token": "<|YaYi|>"}, ":"]),
    format_assistant=StringFormatter(slots=["{{content}}\n\n"]),
    format_system=StringFormatter(slots=[{"token": "<|System|>"}, ":\n{{content}}\n\n"]),
    default_system=(
        "You are a helpful, respectful and honest assistant named YaYi "
        "developed by Beijing Wenge Technology Co.,Ltd. "
        "Always answer as helpfully as possible, while being safe.  "
        "Your answers should not include any harmful, unethical, "
        "racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, "
        "explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information."
    ),
    stop_words=["<|End|>"],
)


# copied from chatml template
register_template(
    name="yi",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    stop_words=["<|im_end|>"],
)


register_template(
    name="yi_vl",
    format_user=StringFormatter(slots=["### Human: {{content}}\n### Assistant:"]),
    format_assistant=StringFormatter(slots=["{{content}}\n"]),
    default_system=(
        "This is a chat between an inquisitive human and an AI assistant. "
        "Assume the role of the AI assistant. Read all the images carefully, "
        "and respond to the human's questions with informative, helpful, detailed and polite answers. "
        "这是一个好奇的人类和一个人工智能助手之间的对话。假设你扮演这个AI助手的角色。"
        "仔细阅读所有的图像，并对人类的问题做出信息丰富、有帮助、详细的和礼貌的回答。\n\n"
    ),
    stop_words=["###"],
    efficient_eos=True,
    mm_plugin=get_mm_plugin(name="llava", image_token="<image>"),
)


register_template(
    name="yuan",
    format_user=StringFormatter(slots=["{{content}}", {"token": "<sep>"}]),
    format_assistant=StringFormatter(slots=["{{content}}<eod>\n"]),
    stop_words=["<eod>"],
)


register_template(
    name="zephyr",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}", {"eos_token"}, "<|assistant|>\n"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}", {"eos_token"}]),
    default_system="You are Zephyr, a helpful assistant.",
)


register_template(
    name="ziya",
    format_user=StringFormatter(slots=["<human>:{{content}}\n<bot>:"]),
    format_assistant=StringFormatter(slots=["{{content}}\n"]),
)
