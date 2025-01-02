from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

from ..extras.logging import get_logger
from .formatter import EmptyFormatter, FunctionFormatter, StringFormatter, ToolFormatter
from .utils import Role, infer_max_len


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from .formatter import SLOTS, Formatter


logger = get_logger(__name__)


@dataclass
class Template:
    format_user: "Formatter"
    format_assistant: "Formatter"
    format_system: "Formatter"
    format_function: "Formatter"
    format_observation: "Formatter"
    format_tools: "Formatter"
    format_separator: "Formatter"
    default_system: str
    stop_words: List[str]
    image_token: str
    efficient_eos: bool
    replace_eos: bool
    force_system: bool

    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        cutoff_len: int = 1_000_000,
        reserved_label_len: int = 1,
    ) -> Tuple[List[int], List[int]]:
        r"""
        Returns a single pair of token ids representing prompt and response respectively.
        """
        encoded_pairs = self._encode(tokenizer, messages, system, tools, cutoff_len, reserved_label_len)
        prompt_ids = []
        for query_ids, resp_ids in encoded_pairs[:-1]:
            prompt_ids += query_ids + resp_ids
        prompt_ids = prompt_ids + encoded_pairs[-1][0]
        answer_ids = encoded_pairs[-1][1]
        return prompt_ids, answer_ids

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        cutoff_len: int = 1_000_000,
        reserved_label_len: int = 1,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        r"""
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        return self._encode(tokenizer, messages, system, tools, cutoff_len, reserved_label_len)

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        cutoff_len: int,
        reserved_label_len: int,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: system + query        resp
        Turn t: sep + query           resp
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []
            if i == 0 and (system or tools or self.force_system):
                tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                elements += self.format_system.apply(content=(system + tool_text))
            elif i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message["role"] == Role.USER.value:
                elements += self.format_user.apply(content=message["content"], idx=str(i // 2))
            elif message["role"] == Role.ASSISTANT.value:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION.value:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION.value:
                elements += self.format_function.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return self._make_pairs(encoded_messages, cutoff_len, reserved_label_len)

    def _convert_elements_to_ids(
        self, tokenizer: "PreTrainedTokenizer", elements: List[Union[str, Dict[str, str]]]
    ) -> List[int]:
        r"""
        Converts elements to token ids.
        """
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
                raise ValueError("Input must be string, set[str] or dict[str, str], got {}".format(type(elem)))

        return token_ids

    def _make_pairs(
        self,
        encoded_messages: Sequence[List[int]],
        cutoff_len: int,
        reserved_label_len: int,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        encoded_pairs = []
        total_length = 0
        for i in range(0, len(encoded_messages), 2):
            if total_length >= cutoff_len:
                break

            max_source_len, max_target_len = infer_max_len(
                source_len=len(encoded_messages[i]),
                target_len=len(encoded_messages[i + 1]),
                max_len=(cutoff_len - total_length),
                reserved_label_len=reserved_label_len,
            )
            source_ids = encoded_messages[i][:max_source_len]
            target_ids = encoded_messages[i + 1][:max_target_len]
            total_length += len(source_ids) + len(target_ids)
            encoded_pairs.append((source_ids, target_ids))

        return encoded_pairs


@dataclass
class Llama2Template(Template):
    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        system: str,
        tools: str,
        cutoff_len: int,
        reserved_label_len: int,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: system + query        resp
        Turn t: sep + query           resp
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []
            system_text = ""
            if i == 0 and (system or tools or self.force_system):
                tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                system_text = self.format_system.apply(content=(system + tool_text))[0]
            elif i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message["role"] == Role.USER.value:
                elements += self.format_user.apply(content=system_text + message["content"])
            elif message["role"] == Role.ASSISTANT.value:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION.value:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION.value:
                elements += self.format_function.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return self._make_pairs(encoded_messages, cutoff_len, reserved_label_len)


templates: Dict[str, Template] = {}


def _register_template(
    name: str,
    format_user: Optional["Formatter"] = None,
    format_assistant: Optional["Formatter"] = None,
    format_system: Optional["Formatter"] = None,
    format_function: Optional["Formatter"] = None,
    format_observation: Optional["Formatter"] = None,
    format_tools: Optional["Formatter"] = None,
    format_separator: Optional["Formatter"] = None,
    default_system: str = "",
    stop_words: List[str] = [],
    image_token: str = "<image>",
    efficient_eos: bool = False,
    replace_eos: bool = False,
    force_system: bool = False,
) -> None:
    r"""
    Registers a chat template.

    To add the following chat template:
    ```
    [HUMAN]:
    user prompt here
    [AI]:
    model response here

    [HUMAN]:
    user prompt here
    [AI]:
    model response here
    ```

    The corresponding code should be:
    ```
    _register_template(
        name="custom",
        format_user=StringFormatter(slots=["[HUMAN]:\n{{content}}\n[AI]:\n"]),
        format_separator=EmptyFormatter(slots=["\n\n"]),
        efficient_eos=True,
    )
    ```
    """
    eos_slots = [] if efficient_eos else [{"eos_token"}]
    template_class = Llama2Template if name.startswith("llama2") else Template
    default_user_formatter = StringFormatter(slots=["{{content}}"])
    default_assistant_formatter = StringFormatter(slots=["{{content}}"] + eos_slots)
    default_function_formatter = FunctionFormatter(slots=["Action: {{name}}\nAction Input: {{arguments}}"] + eos_slots)
    default_tool_formatter = ToolFormatter(tool_format="default")
    default_separator_formatter = EmptyFormatter()
    templates[name] = template_class(
        format_user=format_user or default_user_formatter,
        format_assistant=format_assistant or default_assistant_formatter,
        format_system=format_system or default_user_formatter,
        format_function=format_function or default_function_formatter,
        format_observation=format_observation or format_user or default_user_formatter,
        format_tools=format_tools or default_tool_formatter,
        format_separator=format_separator or default_separator_formatter,
        default_system=default_system,
        stop_words=stop_words,
        image_token=image_token,
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        force_system=force_system,
    )


def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.warning("New tokens have been added, make sure `resize_vocab` is True.")


def _jinja_escape(content: str) -> str:
    return content.replace("'", r"\'")


def _convert_slots_to_jinja(slots: "SLOTS", tokenizer: "PreTrainedTokenizer", placeholder: str = "content") -> str:
    slot_items = []
    for slot in slots:
        if isinstance(slot, str):
            slot_pieces = slot.split("{{content}}")
            if slot_pieces[0]:
                slot_items.append("'" + _jinja_escape(slot_pieces[0]) + "'")
            if len(slot_pieces) > 1:
                slot_items.append(placeholder)
                if slot_pieces[1]:
                    slot_items.append("'" + _jinja_escape(slot_pieces[1]) + "'")
        elif isinstance(slot, set):  # do not use {{ eos_token }} since it may be replaced
            if "bos_token" in slot and tokenizer.bos_token_id is not None:
                slot_items.append("'" + tokenizer.bos_token + "'")
            elif "eos_token" in slot and tokenizer.eos_token_id is not None:
                slot_items.append("'" + tokenizer.eos_token + "'")
        elif isinstance(slot, dict):
            raise ValueError("Dict is not supported.")

    return " + ".join(slot_items)


def _get_jinja_template(template: "Template", tokenizer: "PreTrainedTokenizer") -> str:
    jinja_template = ""

    if template.default_system:
        jinja_template += "{% set system_message = '" + _jinja_escape(template.default_system) + "' %}"

    jinja_template += (
        "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}"
    )

    system_message = _convert_slots_to_jinja(template.format_system.apply(), tokenizer, placeholder="system_message")
    if isinstance(template, Llama2Template):
        pass
    elif template.force_system:
        jinja_template += "{{ " + system_message + " }}"
    else:
        jinja_template += "{% if system_message is defined %}{{ " + system_message + " }}{% endif %}"

    jinja_template += "{% for message in messages %}"
    jinja_template += "{% set content = message['content'] %}"
    if isinstance(template, Llama2Template):
        jinja_template += "{% if loop.index0 == 0 and system_message is defined %}"
        jinja_template += "{% set content = " + system_message + " + message['content'] %}"
        jinja_template += "{% endif %}"

    jinja_template += "{% if message['role'] == 'user' %}"
    user_message = _convert_slots_to_jinja(template.format_user.apply(), tokenizer)
    jinja_template += "{{ " + user_message + " }}"

    jinja_template += "{% elif message['role'] == 'assistant' %}"
    assistant_message = _convert_slots_to_jinja(
        template.format_assistant.apply() + template.format_separator.apply(), tokenizer
    )
    jinja_template += "{{ " + assistant_message + " }}"
    jinja_template += "{% endif %}"
    jinja_template += "{% endfor %}"
    return jinja_template


def get_template_and_fix_tokenizer(
    tokenizer: "PreTrainedTokenizer",
    name: Optional[str] = None,
) -> Template:
    if name is None:
        template = templates["empty"]  # placeholder
    else:
        template = templates.get(name, None)
        if template is None:
            raise ValueError("Template {} does not exist.".format(name))

    stop_words = template.stop_words
    if template.replace_eos:
        if not stop_words:
            raise ValueError("Stop words are required to replace the EOS token.")

        _add_or_replace_eos_token(tokenizer, eos_token=stop_words[0])
        stop_words = stop_words[1:]

    if tokenizer.eos_token_id is None:
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))

    if stop_words:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=stop_words), replace_additional_special_tokens=False
        )
        logger.info("Add {} to stop words.".format(",".join(stop_words)))
        if num_added_tokens > 0:
            logger.warning("New tokens have been added, make sure `resize_vocab` is True.")

    try:
        tokenizer.chat_template = _get_jinja_template(template, tokenizer)
    except ValueError:
        logger.info("Cannot add this chat template to tokenizer.")

    return template


_register_template(
    name="alpaca",
    format_user=StringFormatter(slots=["### Instruction:\n{{content}}\n\n### Response:\n"]),
    format_separator=EmptyFormatter(slots=["\n\n"]),
    default_system=(
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
    ),
)


_register_template(
    name="aquila",
    format_user=StringFormatter(slots=["Human: {{content}}###Assistant:"]),
    format_separator=EmptyFormatter(slots=["###"]),
    default_system=(
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
    ),
    stop_words=["</s>"],
    efficient_eos=True,
)


_register_template(
    name="atom",
    format_user=StringFormatter(
        slots=[{"bos_token"}, "Human: {{content}}\n", {"eos_token"}, {"bos_token"}, "Assistant:"]
    ),
    format_assistant=StringFormatter(slots=["{{content}}\n", {"eos_token"}]),
)


_register_template(
    name="baichuan",
    format_user=StringFormatter(slots=[{"token": "<reserved_102>"}, "{{content}}", {"token": "<reserved_103>"}]),
    efficient_eos=True,
)


_register_template(
    name="baichuan2",
    format_user=StringFormatter(slots=["<reserved_106>{{content}}<reserved_107>"]),
    efficient_eos=True,
)


_register_template(
    name="belle",
    format_user=StringFormatter(slots=["Human: {{content}}\n\nBelle: "]),
    format_system=StringFormatter(slots=[{"bos_token"}, "{{content}}"]),
    format_separator=EmptyFormatter(slots=["\n\n"]),
    force_system=True,
)


_register_template(
    name="bluelm",
    format_user=StringFormatter(slots=[{"token": "[|Human|]:"}, "{{content}}", {"token": "[|AI|]:"}]),
)


_register_template(
    name="breeze",
    format_user=StringFormatter(slots=["[INST] {{content}} [/INST] "]),
    format_system=StringFormatter(slots=[{"bos_token"}, "{{content}}"]),
    default_system=(
        "You are a helpful AI assistant built by MediaTek Research. "
        "The user you are helping speaks Traditional Chinese and comes from Taiwan."
    ),
    efficient_eos=True,
)


_register_template(
    name="chatglm2",
    format_user=StringFormatter(slots=["[Round {{idx}}]\n\n问：{{content}}\n\n答："]),
    format_system=StringFormatter(slots=[{"token": "[gMASK]"}, {"token": "sop"}, "{{content}}"]),
    format_separator=EmptyFormatter(slots=["\n\n"]),
    efficient_eos=True,
    force_system=True,
)


_register_template(
    name="chatglm3",
    format_user=StringFormatter(slots=[{"token": "<|user|>"}, "\n", "{{content}}", {"token": "<|assistant|>"}]),
    format_assistant=StringFormatter(slots=["\n", "{{content}}"]),
    format_system=StringFormatter(slots=[{"token": "[gMASK]"}, {"token": "sop"}, "{{content}}"]),
    format_function=FunctionFormatter(slots=["{{name}}\n{{arguments}}"]),
    format_observation=StringFormatter(
        slots=[{"token": "<|observation|>"}, "\n", "{{content}}", {"token": "<|assistant|>"}]
    ),
    stop_words=["<|user|>", "<|observation|>"],
    efficient_eos=True,
    force_system=True,
)


_register_template(
    name="chatglm3_system",
    format_user=StringFormatter(slots=[{"token": "<|user|>"}, "\n", "{{content}}", {"token": "<|assistant|>"}]),
    format_assistant=StringFormatter(slots=["\n", "{{content}}"]),
    format_system=StringFormatter(
        slots=[{"token": "[gMASK]"}, {"token": "sop"}, {"token": "<|system|>"}, "\n", "{{content}}"]
    ),
    format_function=FunctionFormatter(slots=["{{name}}\n{{arguments}}"]),
    format_observation=StringFormatter(
        slots=[{"token": "<|observation|>"}, "\n", "{{content}}", {"token": "<|assistant|>"}]
    ),
    default_system=(
        "You are ChatGLM3, a large language model trained by Zhipu.AI. "
        "Follow the user's instructions carefully. Respond using markdown."
    ),
    stop_words=["<|user|>", "<|observation|>"],
    efficient_eos=True,
)


_register_template(
    name="chatml",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|im_end|>", "<|im_start|>"],
    replace_eos=True,
)


_register_template(
    name="chatml_de",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="Du bist ein freundlicher und hilfsbereiter KI-Assistent.",
    stop_words=["<|im_end|>", "<|im_start|>"],
    replace_eos=True,
)


_register_template(
    name="codegeex2",
    format_system=StringFormatter(slots=[{"token": "[gMASK]"}, {"token": "sop"}, "{{content}}"]),
    force_system=True,
)


_register_template(
    name="cohere",
    format_user=StringFormatter(
        slots=[
            (
                "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{content}}<|END_OF_TURN_TOKEN|>"
                "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
            )
        ]
    ),
    format_system=StringFormatter(
        slots=[{"bos_token"}, "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{content}}<|END_OF_TURN_TOKEN|>"]
    ),
    default_system=(
        "You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users "
        "by providing thorough responses. You are trained by Cohere."
    ),
)


_register_template(
    name="cpm",
    format_user=StringFormatter(slots=["<用户>{{content}}<AI>"]),
    format_system=StringFormatter(slots=[{"bos_token"}, "{{content}}"]),
    force_system=True,
)


_register_template(
    name="dbrx",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
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


_register_template(
    name="deepseek",
    format_user=StringFormatter(slots=["User: {{content}}\n\nAssistant:"]),
    format_system=StringFormatter(slots=[{"bos_token"}, "{{content}}"]),
    force_system=True,
)


_register_template(
    name="deepseekcoder",
    format_user=StringFormatter(slots=["### Instruction:\n{{content}}\n### Response:"]),
    format_assistant=StringFormatter(slots=["\n", "{{content}}"]),
    format_separator=EmptyFormatter(slots=["\n<|EOT|>\n"]),
    default_system=(
        "You are an AI programming assistant, utilizing the Deepseek Coder model, "
        "developed by Deepseek Company, and you only answer questions related to computer science. "
        "For politically sensitive questions, security and privacy issues, "
        "and other non-computer science questions, you will refuse to answer\n"
    ),
    stop_words=["<|EOT|>"],
    efficient_eos=True,
)


_register_template(
    name="default",
    format_user=StringFormatter(slots=["Human: {{content}}\nAssistant: "]),
    format_system=StringFormatter(slots=["{{content}}\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
)


_register_template(
    name="empty",
    format_user=StringFormatter(slots=["{{content}}"]),
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_system=StringFormatter(slots=[{"bos_token"}, "{{content}}"]),
    efficient_eos=True,
    force_system=True,
)


_register_template(
    name="falcon",
    format_user=StringFormatter(slots=["User: {{content}}\nFalcon:"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    efficient_eos=True,
)


_register_template(
    name="fewshot",
    format_separator=EmptyFormatter(slots=["\n\n"]),
    efficient_eos=True,
)


_register_template(
    name="gemma",
    format_user=StringFormatter(slots=["<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]),
    format_system=StringFormatter(slots=[{"bos_token"}, "{{content}}"]),
    format_observation=StringFormatter(
        slots=["<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]
    ),
    format_separator=EmptyFormatter(slots=["<end_of_turn>\n"]),
    efficient_eos=True,
    force_system=True,
)


_register_template(
    name="intern",
    format_user=StringFormatter(slots=["<|User|>:{{content}}", {"token": "<eoh>"}, "\n<|Bot|>:"]),
    format_separator=EmptyFormatter(slots=[{"token": "<eoa>"}, "\n"]),
    stop_words=["<eoa>"],
    efficient_eos=True,
)


_register_template(
    name="intern2",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=[{"bos_token"}, "<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system=(
        "You are an AI assistant whose name is InternLM (书生·浦语).\n"
        "- InternLM (书生·浦语) is a conversational language model that is developed "
        "by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
        "- InternLM (书生·浦语) can understand and communicate fluently in the language chosen "
        "by the user such as English and 中文."
    ),
    stop_words=["<|im_end|>"],
    efficient_eos=True,  # internlm2 tokenizer cannot set eos_token_id
)


_register_template(
    name="llama2",
    format_user=StringFormatter(slots=[{"bos_token"}, "[INST] {{content}} [/INST]"]),
    format_system=StringFormatter(slots=["<<SYS>>\n{{content}}\n<</SYS>>\n\n"]),
    default_system=(
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe. "
        "Your answers should not include any harmful, unethical, "
        "racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, "
        "explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information."
    ),
)


_register_template(
    name="llama2_zh",
    format_user=StringFormatter(slots=[{"bos_token"}, "[INST] {{content}} [/INST]"]),
    format_system=StringFormatter(slots=["<<SYS>>\n{{content}}\n<</SYS>>\n\n"]),
    default_system="You are a helpful assistant. 你是一个乐于助人的助手。",
)


_register_template(
    name="llama3",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_system=StringFormatter(
        slots=[{"bos_token"}, "<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]
    ),
    format_observation=StringFormatter(
        slots=[
            (
                "<|start_header_id|>tool<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    default_system="You are a helpful assistant.",
    stop_words=["<|eot_id|>"],
    replace_eos=True,
)


_register_template(
    name="llama3-text-to-sql",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>"
            )
        ]
    ),
    format_system=StringFormatter(
        slots=[{"bos_token"}, "<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]
    ),
    format_observation=StringFormatter(
        slots=[
            (
                "<|start_header_id|>tool<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    default_system="당신은 주어진 `입력 텍스트:`와 `DDL Statements`를 참고하여 SQL 쿼리를 작성하는 SQL 쿼리 작성기입니다.\n`쿼리 작성:`이라고 하면 바로 SQL 쿼리를 작성하세요.",
    stop_words=["<|eot_id|>"],
    replace_eos=True,
)

_register_template(
    name="llama3-company-extract",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>"
            )
        ]
    ),
    format_system=StringFormatter(
        slots=[{"bos_token"}, "<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]
    ),
    default_system="당신은 회사명을 찾는 챗봇입니다. \n\n ## 지시 사항 ## \n1. 주어진 입력에 나와있는 회사명을 찾으세요 \n2. 회사명을 찾을수 없다면, 빈 파이썬 리스트를 출력하고, 임의로 답변하려고 시도하지 마세요. \n3. 출력은 반드시 파이썬의 리스트 형태로 생성하십시오.\n4. 뉴스데이터:이 주어지면 회사리스트: 다음에 파이썬 리스트 형태로 작성하세요.",
    stop_words=["<|eot_id|>"],
    replace_eos=True,
)

_register_template(
    name="llama3-rag",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>"
            )
        ]
    ),
    format_system=StringFormatter(
        slots=[{"bos_token"}, "<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]
    ),
    default_system="당신은 검색된 문서를 바탕으로 질문에 답변하는 챗봇입니다.\n\n### 지시 사항 ###\n1. 검색 결과에 있는 내용을 바탕으로 질문에 답하세요.\n2. 만약 당신의 답변에서 검색 결과의 내용을 인용했다면, 답변 끝에 인용한 출처의 문서 id를 꼭 추가해 주세요. 문서 id는 검색 결과에 나와 있는 값을 사용하고, 이중 리스트 형태로 사용됩니다. 예시: [[doc1], [doc8]]\n3. 질문에 대한 답을 검색 결과에서 찾을 수 없다면 검색 결과에서는 해당 내용을 찾을 수 없다고 답변하고, 임의로 답변하려고 시도하지 마세요.",
    stop_words=["<|eot_id|>"],
    replace_eos=True,
)

_register_template(
    name="llama3-hate-speech-detection",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>"
            )
        ]
    ),
    format_system=StringFormatter(
        slots=[{"bos_token"}, "<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]
    ),
    default_system="당신은 주어진 입력이 어떤 혐오 카테고리에 속하는지를 분류하는 분류기입니다.\n\n1. 아래의 카테고리 중에서만 고르세요.\n['출신차별', '외모차별', '정치성향차별', '혐오욕설', '연령차별', '성차별', '인종차별', '종교차별', '혐오아님']\n2. 주어진 입력은 위 8개의 카테고리 중 여러 개에 속할 수 있습니다.\n3. 출력은 반드시 파이썬의 리스트 형태로 생성하십시오.\n4. 입력:이 주어지면 분류 결과: 다음에 파이썬 리스트 형태로 작성하세요.",
    stop_words=["<|eot_id|>"],
    replace_eos=True,
)

_register_template(
    name="llama3-esg",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>"
            )
        ]
    ),
    format_system=StringFormatter(
        slots=[{"bos_token"}, "<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]
    ),
    default_system="당신은 ESG에 대해서 점수를 계산하는 모델입니다. 평가 기준은 다음과 같습니다.\n\n## 지시사항\n주어진 평가 기준을 바탕으로 뉴스 기사로부터 환경, 사회, 지배 구조에 대해서 점수를 측정하십시오.\n\n## 평가 기준\n환경\n1점: 환경 오염 유발, 법적 제재, 규제 위반 등의 사건이 발생한 경우. 예: 심각한 오염 사고, 규제 위반으로 인한 벌금 부과\n2점: 환경 보호에 대한 소극적인 태도 또는 일회성 활동에 그치는 경우. 예: 임시적인 환경 캠페인 참여, 적극적인 감축 목표 미설정\n3점: 환경 보호를 위한 노력이 있으나 체계적인 전략이 부족한 경우. 예: 온실가스 감축 계획 발표, 일부 친환경 인증 획득\n4점: 체계적인 환경 관리 계획이 있으며, 목표 달성에 꾸준히 투자하는 경우. 예: 지속 가능한 에너지 사용, 체계적인 폐기물 관리 프로그램 운영\n5점: 적극적인 환경 정책과 뛰어난 실적 결과를 보이며 업계 선도적 활동을 하는 경우. 예: 탄소 배출 네트 제로(Net Zero) 목표 실현, 친환경 기술 선도\n\n사회\n1점: 인권, 근로자 안전 문제 발생, 지역 사회와의 갈등 등 부정적 이슈 발생. 예: 직원 착취 문제, 안전 관리 소홀로 인한 사고\n2점: 직원 복지나 지역 사회 기여에 대한 제한적 활동 또는 단기적인 참여. 예: 일회성 기부, 사내 복지 프로그램 부족\n3점: 직원 복지 및 사회적 기여에 관심을 가지며 일부 프로그램을 운영하는 경우. 예: 사내 복지 프로그램 운영, 특정 지역 사회 활동 참여\n4점: 직원 복지, 다양성, 지역 사회 공헌에 적극적으로 나서며 책임감 있게 운영하는 경우. 예: 다양성 강화 프로그램, 정기적인 사회공헌 활동\n5점: 사회적 가치 창출에 지속적으로 기여하며 업계 모범적인 활동을 보여주는 경우. 예: 폭넓은 사회적 책임 활동, 소외 계층 지원 프로그램\n\n지배구조\n1점: 지배구조의 투명성 부족, 부패 사건 또는 규제 위반이 발생한 경우. 예: 부패 스캔들, 재무 조작\n2점: 지배구조 개선 노력이 부족하며, 외부 감사나 투명성 문제에 있는 경우. 예: 의사 결정 과정 불투명, 주주와의 소통 부족\n3점: 기본적인 지배구조 원칙을 따르나, 일부 개선의 여지가 있는 경우. 예: 주주와의 소통 강화 중, 투명성 개선 노력\n4점: 투명하고 윤리적인 지배구조를 실천하고 있으며, 책임 있는 경영을 하는 경우. 예: 정기적인 외부 감사, 투명한 의사결정 구조\n5점: 업계에서 모범이 되는 윤리적 지배구조를 유지하고 지속적 개선을 보이는 경우. 예: 독립적 이사회 구성, 주주 및 이해관계자와의 적극적 소통\n\n## 답변 형식\n답변 형식은 반드시 아래의 형식을 따르십시오. 근거:, E:, S:, G: 외에는 그 어떠한 문장도 덧붙이지 마십시오.\n\n근거: 해당 뉴스 기사에 대해서 ESG 점수를 작성한 근거를 작성합니다.\nE: 1~5 사이의 정수\nS: 1~5 사이의 정수\nG: 1~5 사이의 정수\n\n## 뉴스 기사",
    stop_words=["<|eot_id|>"],
    replace_eos=True,
)

_register_template(
    name="llama3-esg-binary-classification",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>"
            )
        ]
    ),
    format_system=StringFormatter(
        slots=[{"bos_token"}, "<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]
    ),
    default_system="1. 상장사의 뉴스를 보고 ESG 관련 기사인지 판단하시오.\n2. ESG 각각에 대한 핵심 키워드는 아래의 10개 키워드를 참고해서 판단하시오.\n- 환경 (Environmental) : 탄소 배출 감소 (Carbon Emission Reduction), 재생 에너지 사용 (Renewable Energy Usage), 자원 효율성 (Resource Efficiency), 폐기물 관리 (Waste Management), 친환경 제품 개발 (Eco-Friendly Product Development), 기후 변화 대응 (Climate Change Mitigation), 오염 방지 (Pollution Prevention), 생물 다양성 보호 (Biodiversity Protection), 물 관리 (Water Management)\n순환 경제 (Circular Economy)\n- 사회 (Social) : 노동 인권 보호 (Labor Rights Protection), 다양성 및 포용성 (Diversity and Inclusion), 직원 안전 및 건강 (Employee Health & Safety), 공정한 노동 관행 (Fair Labor Practices), 커뮤니티 참여 및 개발 (Community Engagement & Development), 제품 책임 및 안전 (Product Responsibility & Safety), 공급망 책임 (Supply Chain Responsibility), 데이터 보호 및 프라이버시 (Data Protection & Privacy), 인재 개발 및 교육 (Talent Development & Education), 고객 만족도 (Customer Satisfaction)\n- 지배구조 (Governance) : 이사회 독립성 (Board Independence), 윤리적 경영 (Ethical Management), 주주 권리 보호 (Shareholder Rights Protection), 내부 통제 및 감사 (Internal Controls & Audits), 리스크 관리 (Risk Management), 투명한 정보 공개 (Transparent Disclosure), 기업 윤리 및 부패 방지 (Corporate Ethics & Anti-Corruption), 경영진 보상 구조 (Executive Compensation Structure), 컴플라이언스 (Regulatory Compliance), 이해관계자 참여 (Stakeholder Engagement)\n3. 입력이 주어지면 근거를 작성하고 ESG 관련 기사이면 => True 아니면 => False를 작성합니다.\n4. 답변은 항상 '=> True' 또는 '=> False'로 끝나야 하며 그 뒤에 아무 것도 적지마세요.\n5. 지역의 정책이나 장소에 대한 내용이거나, 개인의 사건과 같이 특정 기업에 대한 이야기가 아닌 경우에는 아니라고 판단하세요. 저는 상장 '기업'의 내용에 집중하고 있습니다.\n6. 기업의 경영권 분쟁과 같은 경우에는 단순히 개인의 이야기라고 볼 수 없습니다. 이 경우에도 ESG라고 판단할 수 있습니다.\n7. 개인적인 부고의 기사는 ESG와 관련이 없다고 판단하세요.",
    stop_words=["<|eot_id|>"],
    replace_eos=True,
)

_register_template(
    name="mistral",
    format_user=StringFormatter(slots=["[INST] {{content}} [/INST]"]),
    format_system=StringFormatter(slots=[{"bos_token"}, "{{content}}"]),
    force_system=True,
)


_register_template(
    name="olmo",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>\n"]),
    format_system=StringFormatter(slots=[{"eos_token"}, "{{content}}"]),
    force_system=True,
)


_register_template(
    name="openchat",
    format_user=StringFormatter(slots=["GPT4 Correct User: {{content}}", {"eos_token"}, "GPT4 Correct Assistant:"]),
    format_system=StringFormatter(slots=[{"bos_token"}, "{{content}}"]),
    force_system=True,
)


_register_template(
    name="openchat-3.6",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>GPT4 Correct User<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>GPT4 Correct Assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_system=StringFormatter(slots=[{"bos_token"}, "{{content}}"]),
    stop_words=["<|eot_id|>"],
    replace_eos=True,
    force_system=True,
)


_register_template(
    name="orion",
    format_user=StringFormatter(slots=["Human: {{content}}\n\nAssistant: ", {"eos_token"}]),
    format_system=StringFormatter(slots=[{"bos_token"}, "{{content}}"]),
    force_system=True,
)


_register_template(
    name="phi",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|end|>\n<|assistant|>\n"]),
    format_system=StringFormatter(slots=[{"bos_token"}, "<|system|>\n{{content}}<|end|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="You are a helpful AI assistant.",
    stop_words=["<|end|>"],
    replace_eos=True,
)


_register_template(
    name="qwen",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
)


_register_template(
    name="solar",
    format_user=StringFormatter(slots=["### User:\n{{content}}\n\n### Assistant:\n"]),
    format_system=StringFormatter(slots=["### System:\n{{content}}\n\n"]),
    efficient_eos=True,
)


_register_template(
    name="starchat",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|end|>\n<|assistant|>"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}<|end|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|end|>"],
    replace_eos=True,
    force_system=True,
)


_register_template(
    name="telechat",
    format_user=StringFormatter(slots=["<_user>{{content}}<_bot>"]),
    format_system=StringFormatter(slots=["<_system>{{content}}<_end>"]),
    stop_words=["<_end>"],
    replace_eos=True,
)


_register_template(
    name="vicuna",
    format_user=StringFormatter(slots=["USER: {{content}} ASSISTANT:"]),
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
)


_register_template(
    name="xuanyuan",
    format_user=StringFormatter(slots=["Human: {{content}} Assistant:"]),
    default_system=(
        "以下是用户和人工智能助手之间的对话。用户以Human开头，人工智能助手以Assistant开头，"
        "会对人类提出的问题给出有帮助、高质量、详细和礼貌的回答，并且总是拒绝参与与不道德、"
        "不安全、有争议、政治敏感等相关的话题、问题和指示。\n"
    ),
)


_register_template(
    name="xverse",
    format_user=StringFormatter(slots=["Human: {{content}}\n\nAssistant: "]),
)


_register_template(
    name="yayi",
    format_user=StringFormatter(slots=[{"token": "<|Human|>"}, ":\n{{content}}\n\n", {"token": "<|YaYi|>"}, ":"]),
    format_system=StringFormatter(slots=[{"token": "<|System|>"}, ":\n{{content}}\n\n"]),
    format_separator=EmptyFormatter(slots=["\n\n"]),
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


_register_template(
    name="yi",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
)


_register_template(
    name="yi_vl",
    format_user=StringFormatter(slots=["### Human: {{content}}\n### Assistant:"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system=(
        "This is a chat between an inquisitive human and an AI assistant. "
        "Assume the role of the AI assistant. Read all the images carefully, "
        "and respond to the human's questions with informative, helpful, detailed and polite answers. "
        "这是一个好奇的人类和一个人工智能助手之间的对话。假设你扮演这个AI助手的角色。"
        "仔细阅读所有的图像，并对人类的问题做出信息丰富、有帮助、详细的和礼貌的回答。\n\n"
    ),
    stop_words=["###"],
    efficient_eos=True,
)


_register_template(
    name="yuan",
    format_user=StringFormatter(slots=["{{content}}", {"token": "<sep>"}]),
    format_separator=EmptyFormatter(slots=["\n"]),
    stop_words=["<eod>"],
    replace_eos=True,
)


_register_template(
    name="zephyr",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}", {"eos_token"}, "<|assistant|>"]),
    format_assistant=StringFormatter(slots=["\n{{content}}", {"eos_token"}]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}", {"eos_token"}]),
    default_system="You are Zephyr, a helpful assistant.",
)


_register_template(
    name="ziya",
    format_user=StringFormatter(slots=["<human>:{{content}}\n<bot>:"]),
    format_separator=EmptyFormatter(slots=["\n"]),
)
