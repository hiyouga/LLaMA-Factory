from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Template:

    prefix: str
    prompt: str
    sep: str
    use_history: bool
    stop_words: List[str]

    def get_prompt(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        prefix: Optional[str] = "",
        eos_token: Optional[str] = "</s>"
    ) -> str:
        r"""
        Returns a string containing prompt without response.
        """
        return eos_token.join(map(lambda x: x[0] + x[1], self._format_example(query, history, prefix)))

    def get_dialog(
        self,
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        prefix: Optional[str] = ""
    ) -> List[Tuple[str, str]]:
        r"""
        Returns a list containing prompt-response pairs.
        """
        result = self._format_example(query, history, prefix)
        result[-1][-1] = resp
        return result

    def _format_example(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        prefix: Optional[str] = ""
    ) -> List[Tuple[str, str]]:
        prefix = prefix or self.prefix # use prefix if provided
        prefix = prefix + self.sep if prefix else "" # add separator for non-empty prefix
        history = history if (history and self.use_history) else []
        history = history + [(query, "")]
        return [
            [(self.sep if i else prefix) + self.prompt.format(query=q), r]
            for i, (q, r) in enumerate(history)
        ]


@dataclass
class Llama2Template(Template):

    def _format_example(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        prefix: Optional[str] = ""
    ) -> List[Tuple[str, str]]:
        prefix = prefix or self.prefix # use prefix if provided
        prefix = prefix if prefix.startswith("<<SYS>>") else "<<SYS>>\n{}\n<</SYS>>\n\n".format(prefix)
        history = history if (history and self.use_history) else []
        history = history + [(query, "")]
        return [
            [(self.sep if i else "") + self.prompt.format(query=(q if i else prefix + q)), r]
            for i, (q, r) in enumerate(history)
        ]


templates: Dict[str, Template] = {}


def register_template(
    name: str, prefix: str, prompt: str, sep: str, use_history: bool, stop_words: List[str]
) -> None:
    template_class = Llama2Template if name == "llama2" else Template
    templates[name] = template_class(
        prefix=prefix,
        prompt=prompt,
        sep=sep,
        use_history=use_history,
        stop_words=stop_words
    )


def get_template(name: str) -> Template:
    template = templates.get(name, None)
    assert template is not None, "Template {} does not exist.".format(name)
    return template


r"""
Supports language model inference without histories.
"""
register_template(
    name="vanilla",
    prefix="",
    prompt="{query}",
    sep="",
    use_history=False,
    stop_words=[]
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
    use_history=True,
    stop_words=[]
)


r"""
Supports: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
          https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
          https://huggingface.co/meta-llama/Llama-2-70b-chat-hf
"""
register_template(
    name="llama2",
    prefix="<<SYS>>\nYou are a helpful, respectful and honest assistant. "
           "Always answer as helpfully as possible, while being safe.  "
           "Your answers should not include any harmful, unethical, "
           "racist, sexist, toxic, dangerous, or illegal content. "
           "Please ensure that your responses are socially unbiased and positive in nature.\n"
           "If a question does not make any sense, or is not factually coherent, "
           "explain why instead of answering something not correct. "
           "If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n",
    prompt="[INST] {query} [/INST] ",
    sep="<s>",
    use_history=True,
    stop_words=[]
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
    use_history=True,
    stop_words=[]
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
    sep="",
    use_history=True,
    stop_words=[]
)


r"""
Supports: https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-13B
"""
register_template(
    name="belle",
    prefix="",
    prompt="Human: {query}\n\nBelle: ",
    sep="\n\n",
    use_history=True,
    stop_words=[]
)


r"""
Supports: https://github.com/CVI-SZU/Linly
"""
register_template(
    name="linly",
    prefix="",
    prompt="User: {query}\nBot: ",
    sep="\n",
    use_history=True,
    stop_words=[]
)


r"""
Supports: https://github.com/Neutralzz/BiLLa
"""
register_template(
    name="billa",
    prefix="",
    prompt="Human: {query}\nAssistant: ",
    sep="\n",
    use_history=True,
    stop_words=[]
)


r"""
Supports: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1
"""
register_template(
    name="ziya",
    prefix="",
    prompt="<human>:{query}\n<bot>:",
    sep="\n",
    use_history=True,
    stop_words=[]
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
    use_history=True,
    stop_words=[]
)


r"""
Supports: https://huggingface.co/internlm/internlm-chat-7b
"""
register_template(
    name="intern",
    prefix="",
    prompt="<|User|>:{query}<eoh>\n<|Bot|>:",
    sep="<eoa>\n",
    use_history=True,
    stop_words=["<eoa>"]
)


r"""
Supports: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat
"""
register_template(
    name="baichuan",
    prefix="",
    prompt="<reserved_102>{query}<reserved_103>",
    sep="",
    use_history=True,
    stop_words=[]
)


r"""
Supports: https://huggingface.co/HuggingFaceH4/starchat-alpha
          https://huggingface.co/HuggingFaceH4/starchat-beta
"""
register_template(
    name="starchat",
    prefix="<|system|>\n",
    prompt="<|user|>\n{query}<|end|>\n<|assistant|>\n",
    sep="<|end|>\n",
    use_history=True,
    stop_words=["<|end|>"]
)


r"""
Supports: https://huggingface.co/Qwen/Qwen-7B-Chat
"""
register_template(
    name="chatml",
    prefix="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
    prompt="<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
    sep="<|im_end|>\n",
    use_history=True,
    stop_words=["<|im_end|>"]
)
