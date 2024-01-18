from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

from ..extras.logging import get_logger
from .utils import Role
from .formatter import StringFormatter, FunctionFormatter, ToolFormatter


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


logger = get_logger(__name__)


@dataclass
class Template:

    format_user: Callable
    format_assistant: Callable
    format_system: Callable
    format_tool: Callable
    format_observation: Callable
    format_function: Callable
    system: str
    separator: List[Union[str, Dict[str, str]]]
    stop_words: List[str]
    efficient_eos: bool
    replace_eos: bool

    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        system: str,
        tools: str,
        cutoff_len: Optional[int] = 1_000_000
    ) -> Tuple[List[int], List[int]]:
        r"""
        Returns a single pair of token ids representing prompt and response respectively.
        """
        encoded_pairs = self._encode(tokenizer, messages, system, tools, cutoff_len)
        prompt_ids = []
        for query_ids, resp_ids in encoded_pairs[:-1]:
            prompt_ids = prompt_ids + query_ids + resp_ids
        prompt_ids = prompt_ids + encoded_pairs[-1][0]
        answer_ids = encoded_pairs[-1][1]
        return prompt_ids, answer_ids

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        system: str,
        tools: str,
        cutoff_len: Optional[int] = 1_000_000
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        encoded_pairs = self._encode(tokenizer, messages, system, tools, cutoff_len)
        return encoded_pairs

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        system: str,
        tools: str,
        cutoff_len: int
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: system + query        resp + eos
        Turn t: sep + query           resp + eos
        """
        system = system or self.system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []
            if i == 0 and (system or tools):
                tool_text = self.format_tool(content=tools)[0] if tools else ""
                elements += self.format_system(content=(system + tool_text))
            elif i > 0 and i % 2 == 0:
                elements += self.separator

            if message["role"] == Role.USER:
                elements += self.format_user(content=message["content"], idx=str(i // 2))
            elif message["role"] == Role.ASSISTANT:
                elements += self.format_assistant(content=message["content"])
            elif message["role"] == Role.OBSERVATION:
                elements += self.format_observation(content=message["content"])
            elif message["role"] == Role.FUNCTION:
                elements += self.format_function(content=message["content"])

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        # TODO: need to improve
        encoded_pairs = []
        total_length = 0
        for i in range(0, len(encoded_messages), 2):
            if total_length >= cutoff_len:
                break

            encoded_messages[i] = encoded_messages[i][:cutoff_len-total_length]
            total_length += len(encoded_messages[i])

            encoded_messages[i+1] = encoded_messages[i+1][:max(1, cutoff_len-total_length)]
            total_length += len(encoded_messages[i+1])
            encoded_pairs.append((encoded_messages[i], encoded_messages[i+1]))

        return encoded_pairs

    def _convert_elements_to_ids(
        self,
        tokenizer: "PreTrainedTokenizer",
        elements: List[Union[str, Dict[str, str]]]
    ) -> List[int]:
        r"""
        Converts elements to token ids.
        """
        token_ids = []
        for elem in elements:
            if isinstance(elem, str):
                if len(elem) != 0:
                    token_ids = token_ids + tokenizer.encode(elem, add_special_tokens=False)
            elif isinstance(elem, dict):
                token_ids = token_ids + [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            elif isinstance(elem, set):
                if "bos_token" in elem and tokenizer.bos_token_id:
                    token_ids = token_ids + [tokenizer.bos_token_id]
                elif "eos_token" in elem and tokenizer.eos_token_id:
                    token_ids = token_ids + [tokenizer.eos_token_id]
            else:
                raise ValueError("Input must be string, set[str] or dict[str, str], got {}".format(type(elem)))

        return token_ids


@dataclass
class Llama2Template(Template):

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        system: str,
        tools: str,
        cutoff_len: int
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: system + query        resp + eos
        Turn t: sep + query           resp + eos
        """
        system = system or self.system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []
            system_text = ""
            if i == 0 and (system or tools):
                tool_text = self.format_tool(content=tools)[0] if tools else ""
                system_text = self.format_system(content=(system + tool_text))[0]
            elif i > 0 and i % 2 == 0:
                elements += self.separator

            if message["role"] == Role.USER:
                elements += self.format_user(content=system_text + message["content"], idx=str(i // 2))
            elif message["role"] == Role.ASSISTANT:
                elements += self.format_assistant(content=message["content"])
            elif message["role"] == Role.OBSERVATION:
                elements += self.format_observation(content=message["content"])
            elif message["role"] == Role.FUNCTION:
                elements += self.format_function(content=message["content"])

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        # TODO: need to improve
        encoded_pairs = []
        total_length = 0
        for i in range(0, len(encoded_messages), 2):
            if total_length >= cutoff_len:
                break

            encoded_messages[i] = encoded_messages[i][:cutoff_len-total_length]
            total_length += len(encoded_messages[i])

            encoded_messages[i+1] = encoded_messages[i+1][:max(1, cutoff_len-total_length)]
            total_length += len(encoded_messages[i+1])
            encoded_pairs.append((encoded_messages[i], encoded_messages[i+1]))

        return encoded_pairs


templates: Dict[str, Template] = {}


def register_template(
    name: str,
    format_user: Optional[Callable] = None,
    format_assistant: Optional[Callable] = None,
    format_system: Optional[Callable] = None,
    format_tool: Optional[Callable] = None,
    format_observation: Optional[Callable] = None,
    format_function: Optional[Callable] = None,
    system: Optional[str] = "",
    separator: Optional[List[Union[str, Dict[str, str]]]] = "",
    stop_words: Optional[List[str]] = [],
    efficient_eos: Optional[bool] = False,
    replace_eos: Optional[bool] = False
) -> None:
    template_class = Llama2Template if name.startswith("llama2") else Template
    templates[name] = template_class(
        format_user=format_user or StringFormatter(container=["{{content}}"]),
        format_assistant=format_assistant or StringFormatter(container=[
            "{{content}}", {"eos_token"}
        ]),
        format_system=format_system or StringFormatter(container=["{{content}}"]),
        format_tool=format_tool or ToolFormatter(type="default"),
        format_observation=format_observation or format_user,
        format_function=format_function or FunctionFormatter(container=[
            "Action: {{name}}\nAction Input: {{arguments}}", {"eos_token"}
        ]),
        system=system,
        separator=separator,
        stop_words=stop_words,
        efficient_eos=efficient_eos,
        replace_eos=replace_eos
    )


def get_template_and_fix_tokenizer(
    name: str,
    tokenizer: "PreTrainedTokenizer"
) -> Template:
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = "<|endoftext|>"
        logger.info("Add eos token: {}".format(tokenizer.eos_token))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))

    if name is None: # for pre-training
        return None

    template = templates.get(name, None)
    assert template is not None, "Template {} does not exist.".format(name)

    stop_words = template.stop_words
    if template.replace_eos:
        if not stop_words:
            raise ValueError("Stop words are required to replace the EOS token.")

        tokenizer.eos_token = stop_words[0]
        stop_words = stop_words[1:]
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if stop_words:
        tokenizer.add_special_tokens(
            dict(additional_special_tokens=stop_words),
            replace_additional_special_tokens=False
        )
        logger.info("Add {} to stop words.".format(",".join(stop_words)))

    return template


register_template(
    name="alpaca",
    format_user=StringFormatter(container=[
        "### Instruction:\n{{content}}\n\n### Response:\n"
    ]),
    system=(
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    ),
    separator=[
        "\n\n"
    ]
)


register_template(
    name="aquila",
    format_user=StringFormatter(container=[
        "Human: {{content}}###Assistant:"
    ]),
    format_assistant=StringFormatter(container=[
        "{{content}}"
    ]),
    system=(
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
    ),
    separator=[
        "###"
    ],
    stop_words=[
        "</s>"
    ],
    efficient_eos=True
)


register_template(
    name="baichuan",
    format_user=StringFormatter(container=[
        {"token": "<reserved_102>"},
        "{{content}}",
        {"token": "<reserved_103>"}
    ]),
    format_assistant=StringFormatter(container=[
        "{{content}}"
    ]),
    efficient_eos=True
)


register_template(
    name="baichuan2",
    format_user=StringFormatter(container=[
        {"token": "<reserved_106>"},
        "{{content}}",
        {"token": "<reserved_107>"}
    ]),
    format_assistant=StringFormatter(container=[
        "{{content}}"
    ]),
    efficient_eos=True
)


register_template(
    name="belle",
    format_user=StringFormatter(container=[
        "Human: {{content}}\n\nBelle: "
    ]),
    separator=[
        "\n\n"
    ]
)


register_template(
    name="bluelm",
    format_user=StringFormatter(container=[
        {"token": "[|Human|]:"},
        "{{content}}",
        {"token": "[|AI|]:"}
    ])
)


register_template(
    name="chatglm2",
    format_user=StringFormatter(container=[
        "[Round {{idx}}]\n\n问：{{content}}\n\n答："
    ]),
    format_assistant=StringFormatter(container=[
        "{{content}}"
    ]),
    format_system=StringFormatter(container=[
        {"token": "[gMASK]"},
        {"token": "sop"},
        "{{content}}"
    ]),
    separator=[
        "\n\n"
    ],
    efficient_eos=True
)


register_template(
    name="chatglm3",
    format_user=StringFormatter(container=[
        {"token": "<|user|>"},
        "\n",
        "{{content}}",
        {"token": "<|assistant|>"}
    ]),
    format_assistant=StringFormatter(container=[
        "\n"
        "{{content}}"
    ]),
    format_system=StringFormatter(container=[
        {"token": "[gMASK]"},
        {"token": "sop"},
        {"token": "<|system|>"},
        "\n",
        "{{content}}"
    ]),
    format_observation=StringFormatter(container=[
        {"token": "<|observation|>"},
        "\n",
        "{{content}}"
    ]),
    format_function=FunctionFormatter(container=[
        "{{name}}\n{{arguments}}"
    ]),
    system=(
        "You are ChatGLM3, a large language model trained by Zhipu.AI. "
        "Follow the user's instructions carefully. Respond using markdown."
    ),
    stop_words=[
        "<|user|>",
        "<|observation|>"
    ],
    efficient_eos=True
)


register_template(
    name="codegeex2",
    format_system=StringFormatter(container=[
        {"token": "[gMASK]"},
        {"token": "sop"},
        "{{content}}"
    ])
)


register_template(
    name="deepseek",
    format_user=StringFormatter(container=[
        "User: {{content}}\n\nAssistant:"
    ])
)


register_template(
    name="deepseekcoder",
    format_user=StringFormatter(container=[
        "### Instruction:\n{{content}}\n### Response:\n"
    ]),
    format_assistant=StringFormatter(container=[
        "{{content}}"
    ]),
    system=(
        "You are an AI programming assistant, utilizing the Deepseek Coder model, "
        "developed by Deepseek Company, and you only answer questions related to computer science. "
        "For politically sensitive questions, security and privacy issues, "
        "and other non-computer science questions, you will refuse to answer\n"
    ),
    separator=[
        "\n",
        {"token": "<|EOT|>"},
        "\n"
    ],
    stop_words=[
        "<|EOT|>"
    ],
    efficient_eos=True
)


register_template(
    name="default",
    format_user=StringFormatter(container=[
        "Human: {{content}}\nAssistant: "
    ]),
    system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    ),
    separator=[
        "\n"
    ]
)


register_template(
    name="falcon",
    format_user=StringFormatter(container=[
        "User: {{content}}\nFalcon:"
    ]),
    format_assistant=StringFormatter(container=[
        "{{content}}"
    ]),
    separator=[
        "\n"
    ],
    efficient_eos=True
)


register_template(
    name="intern",
    format_user=StringFormatter(container=[
        "<|User|>:{{content}}",
        {"token": "<eoh>"},
        "\n<|Bot|>:"
    ]),
    format_assistant=StringFormatter(container=[
        "{{content}}"
    ]),
    separator=[
        {"token": "<eoa>"},
        "\n"
    ],
    stop_words=[
        "<eoa>"
    ],
    efficient_eos=True
)


register_template(
    name="intern2",
    format_user=StringFormatter(container=[
        {"token": "[UNUSED_TOKEN_146]"},
        "user\n{{content}}",
        {"token": "[UNUSED_TOKEN_145]"},
        "\n",
        {"token": "[UNUSED_TOKEN_146]"},
        "assistant\n"
    ]),
    format_assistant=StringFormatter(container=[
        "{{content}}"
    ]),
    format_system=StringFormatter(container=[
        {"token": "[UNUSED_TOKEN_146]"},
        "system\n{{content}}",
        {"token": "[UNUSED_TOKEN_145]"},
        "\n"
    ]),
    system=(
        "You are an AI assistant whose name is InternLM (书生·浦语).\n"
        "- InternLM (书生·浦语) is a conversational language model that is developed "
        "by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
        "- InternLM (书生·浦语) can understand and communicate fluently in the language chosen "
        "by the user such as English and 中文."
    ),
    separator=[
        {"token": "[UNUSED_TOKEN_145]"},
        "\n"
    ],
    stop_words=[
        "[UNUSED_TOKEN_145]"
    ],
    efficient_eos=True
)


register_template(
    name="llama2",
    format_user=StringFormatter(container=["[INST] {{content}} [/INST]"]),
    format_system=StringFormatter(container=["<<SYS>>\n{{content}}\n<</SYS>>\n\n"]),
    system=(
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe. "
        "Your answers should not include any harmful, unethical, "
        "racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, "
        "explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information."
    )
)


register_template(
    name="llama2_zh",
    format_user=StringFormatter(container=["[INST] {{content}} [/INST]"]),
    format_system=StringFormatter(container=["<<SYS>>\n{{content}}\n<</SYS>>\n\n"]),
    system="You are a helpful assistant. 你是一个乐于助人的助手。"
)


register_template(
    name="mistral",
    format_user=StringFormatter(container=["[INST] {{content}} [/INST]"])
)


register_template(
    name="openchat",
    format_user=StringFormatter(container=[
        "GPT4 Correct User: {{content}}",
        {"token": "<|end_of_turn|>"},
        "GPT4 Correct Assistant:"
    ]),
    format_assistant=StringFormatter(container=[
        "{{content}}"
    ]),
    separator=[
        {"token": "<|end_of_turn|>"}
    ],
    stop_words=[
        "<|end_of_turn|>"
    ],
    efficient_eos=True
)


register_template(
    name="qwen",
    format_user=StringFormatter(container=[
        "<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"
    ]),
    format_system=StringFormatter(container=[
        "<|im_start|>system\n{{content}}<|im_end|>\n"
    ]),
    system="You are a helpful assistant.",
    separator=[
        "\n"
    ],
    stop_words=[
        "<|im_end|>"
    ],
    replace_eos=True
)


register_template(
    name="solar",
    format_user=StringFormatter(container=[
        "### User:\n{{content}}\n\n### Assistant:\n"
    ])
)


register_template(
    name="starchat",
    format_user=StringFormatter(container=[
        {"token": "<|user|>"},
        "\n{{content}}",
        {"token": "<|end|>"},
        "\n",
        {"token": "<|assistant|>"}
    ]),
    format_assistant=StringFormatter(container=[
        "{{content}}"
    ]),
    format_system=StringFormatter(container=[
        {"token": "<|system|>"},
        "\n{{content}}",
        {"token": "<|end|>"},
        "\n"
    ]),
    separator=[
        {"token": "<|end|>"},
        "\n"
    ],
    stop_words=[
        "<|end|>"
    ],
    efficient_eos=True
)


register_template(
    name="vanilla"
)


register_template(
    name="vicuna",
    format_user=StringFormatter(container=[
        "USER: {{content}} ASSISTANT:"
    ]),
    system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )
)


register_template(
    name="xuanyuan",
    format_user=StringFormatter(container=[
        "Human: {{content}} Assistant:"
    ]),
    system=(
        "以下是用户和人工智能助手之间的对话。用户以Human开头，人工智能助手以Assistant开头，"
        "会对人类提出的问题给出有帮助、高质量、详细和礼貌的回答，并且总是拒绝参与与不道德、"
        "不安全、有争议、政治敏感等相关的话题、问题和指示。\n"
    )
)


register_template(
    name="xverse",
    format_user=StringFormatter(container=[
        "Human: {{content}}\n\nAssistant: "
    ])
)


register_template(
    name="yayi",
    format_user=StringFormatter(container=[
        {"token": "<|Human|>"},
        ":\n{{content}}\n\n",
        {"token": "<|YaYi|>"},
        ":"
    ]),
    format_system=StringFormatter(container=[
        {"token": "<|System|>"},
        ":\n{{content}}\n\n"
    ]),
    system=(
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
    separator=[
        "\n\n"
    ],
    stop_words=[
        "<|End|>"
    ]
)


register_template(
    name="yi",
    format_user=StringFormatter(container=[
        "<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"
    ]),
    separator=[
        "\n"
    ],
    stop_words=[
        "<|im_end|>"
    ],
    replace_eos=True
)


register_template(
    name="yuan",
    format_user=StringFormatter(container=[
        "{{content}}",
        {"token": "<sep>"}
    ]),
    separator=[
        "\n"
    ],
    stop_words=[
        "<eod>"
    ],
    replace_eos=True
)


register_template(
    name="zephyr",
    format_user=StringFormatter(container=[
        "<|user|>\n{{content}}</s><|assistant|>"
    ]),
    format_system=StringFormatter(container=[
        "<|system|>\n{{content}}</s>",
    ]),
    system="You are a friendly chatbot who always responds in the style of a pirate"
)


register_template(
    name="ziya",
    format_user=StringFormatter(container=[
        {"token": "<human>"},
        ":{{content}}\n",
        {"token": "<bot>"},
        ":"
    ]),
    separator=[
        "\n"
    ]
)
