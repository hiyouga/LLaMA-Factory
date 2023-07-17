from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Format:
    prefix: str
    prompt: str
    sep: str
    use_history: bool


templates: Dict[str, Format] = {}


@dataclass
class Template:

    name: str

    def __post_init__(self):
        if self.name in templates:
            self.prefix = templates[self.name].prefix
            self.prompt = templates[self.name].prompt
            self.sep = templates[self.name].sep
            self.use_history = templates[self.name].use_history
        else:
            raise ValueError("Template {} does not exist.".format(self.name))

    def get_prompt(
        self, query: str, history: Optional[List[Tuple[str, str]]] = None, prefix: Optional[str] = ""
    ) -> str:
        r"""
        Returns a string containing prompt without response.
        """
        return "".join(self._format_example(query, history, prefix))

    def get_dialog(
        self, query: str, resp: str, history: Optional[List[Tuple[str, str]]] = None, prefix: Optional[str] = ""
    ) -> List[str]:
        r"""
        Returns a list containing 2 * n elements where the 2k-th is a query and the (2k+1)-th is a response.
        """
        return self._format_example(query, history, prefix) + [resp]

    def _format_example(
        self, query: str, history: Optional[List[Tuple[str, str]]] = None, prefix: Optional[str] = ""
    ) -> List[str]:
        prefix = prefix if prefix else self.prefix # use prefix if provided
        prefix = prefix + self.sep if prefix else "" # add separator for non-empty prefix
        history = history if (history and self.use_history) else []
        history = history + [(query, "<dummy>")]
        convs = []
        for turn_idx, (user_query, bot_resp) in enumerate(history):
            if turn_idx == 0:
                convs.append(prefix + self.prompt.format(query=user_query))
                convs.append(bot_resp)
            else:
                convs.append(self.sep + self.prompt.format(query=user_query))
                convs.append(bot_resp)
        return convs[:-1] # drop last


def register_template(name: str, prefix: str, prompt: str, sep: str, use_history: bool) -> None:
    templates[name] = Format(
        prefix=prefix,
        prompt=prompt,
        sep=sep,
        use_history=use_history
    )


r"""
Supports language model inference without histories.
"""
register_template(
    name="vanilla",
    prefix="",
    prompt="{query}",
    sep="",
    use_history=False
)


r"""
Default template.
"""
register_template(
    name="default",
    prefix="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    prompt="Human: {query}\nAssistant: ",
    sep="\n",
    use_history=True
)


r"""
Supports: https://huggingface.co/tatsu-lab/alpaca-7b-wdiff
          https://github.com/ymcui/Chinese-LLaMA-Alpaca
"""
register_template(
    name="alpaca",
    prefix="Below is an instruction that describes a task. "
           "Write a response that appropriately completes the request.",
    prompt="### Instruction:\n{query}\n\n### Response:\n",
    sep="\n\n",
    use_history=True
)


r"""
Supports: https://huggingface.co/lmsys/vicuna-7b-delta-v1.1
          https://huggingface.co/lmsys/vicuna-13b-delta-v1.1
"""
register_template(
    name="vicuna",
    prefix="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    prompt="USER: {query} ASSISTANT: ",
    sep="</s>",
    use_history=True
)


r"""
Supports: https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-13B
"""
register_template(
    name="belle",
    prefix="",
    prompt="Human: {query}\n\nBelle: ",
    sep="\n\n",
    use_history=True
)


r"""
Supports: https://github.com/CVI-SZU/Linly
"""
register_template(
    name="linly",
    prefix="",
    prompt="User: {query}\nBot: ",
    sep="\n",
    use_history=True
)


r"""
Supports: https://github.com/Neutralzz/BiLLa
"""
register_template(
    name="billa",
    prefix="",
    prompt="Human: {query}\nAssistant: ",
    sep="\n",
    use_history=True
)


r"""
Supports: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1
"""
register_template(
    name="ziya",
    prefix="",
    prompt="<human>:{query}\n<bot>:",
    sep="\n",
    use_history=True
)


r"""
Supports: https://huggingface.co/qhduan/aquilachat-7b
"""
register_template(
    name="aquila",
    prefix="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    prompt="Human: {query}###Assistant: ",
    sep="###",
    use_history=True
)


r"""
Supports: https://huggingface.co/internlm/internlm-chat-7b
"""
register_template(
    name="intern",
    prefix="",
    prompt="<|User|>:{query}<eoh>\n<|Bot|>:",
    sep="<eoa>\n",
    use_history=True
)


r"""
Supports: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat
"""
register_template(
    name="baichuan",
    prefix="",
    prompt="<reserved_102>{query}<reserved_103>",
    sep="",
    use_history=True
)
