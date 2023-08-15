import tiktoken
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from llmtuner.extras.logging import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


logger = get_logger(__name__)


@dataclass
class Template:

    prefix: List[Union[str, Dict[str, str]]]
    prompt: List[Union[str, Dict[str, str]]]
    sep: List[Union[str, Dict[str, str]]]
    stop_words: List[str]
    use_history: bool

    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        prefix: Optional[str] = None
    ) -> Tuple[List[int], List[int]]:
        r"""
        Returns a single pair of token ids representing prompt and response respectively.
        """
        prefix, history = self._format(query, resp, history, prefix)
        encoded_pairs = self._encode(tokenizer, prefix, history)
        prompt_ids = []
        for query_ids, resp_ids in encoded_pairs[:-1]:
            prompt_ids = prompt_ids + query_ids + resp_ids
        prompt_ids = prompt_ids + encoded_pairs[-1][0]
        return prompt_ids, encoded_pairs[-1][1]

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        prefix: Optional[str] = None
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        prefix, history = self._format(query, resp, history, prefix)
        encoded_pairs = self._encode(tokenizer, prefix, history)
        return encoded_pairs

    def _format(
        self,
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        prefix: Optional[str] = None
    ) -> Tuple[List[Union[str, Dict[str, str]]], List[Tuple[str, str]]]:
        r"""
        Aligns inputs to the standard format.
        """
        prefix = [prefix] if prefix else self.prefix # use prefix if provided
        history = history if (history and self.use_history) else []
        history = history + [(query, resp)]
        return prefix, history

    def _get_special_ids(
        self,
        tokenizer: "PreTrainedTokenizer"
    ) -> Tuple[List[int], List[int]]:
        if tokenizer.bos_token_id:
            bos_ids = [tokenizer.bos_token_id]
        else:
            bos_ids = [] # bos token is optional

        if tokenizer.eos_token_id:
            eos_ids = [tokenizer.eos_token_id]
        else:
            raise ValueError("EOS token is required.")

        return bos_ids, eos_ids

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        prefix: List[Union[str, Dict[str, str]]],
        history: List[Tuple[str, str]]
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: bos + prefix + sep + query    resp + eos
        Turn t: sep + bos + query             resp + eos
        """
        bos_ids, eos_ids = self._get_special_ids(tokenizer)
        sep_ids = self._convert_inputs_to_ids(tokenizer, context=self.sep)
        encoded_pairs = []
        for turn_idx, (query, resp) in enumerate(history):
            if turn_idx == 0:
                if prefix: # has prefix
                    prefix_ids = bos_ids + self._convert_inputs_to_ids(tokenizer, context=prefix) + sep_ids
                else:
                    prefix_ids = bos_ids
            else:
                prefix_ids = sep_ids + bos_ids

            query_ids = self._convert_inputs_to_ids(tokenizer, context=self.prompt, query=query, idx=str(turn_idx))
            resp_ids = self._convert_inputs_to_ids(tokenizer, context=[resp])
            encoded_pairs.append((prefix_ids + query_ids, resp_ids + eos_ids))
        return encoded_pairs

    def _convert_inputs_to_ids(
        self,
        tokenizer: "PreTrainedTokenizer",
        context: List[Union[str, Dict[str, str]]],
        query: Optional[str] = "",
        idx: Optional[str] = ""
    ) -> List[int]:
        r"""
        Converts context to token ids.
        """
        if isinstance(getattr(tokenizer, "tokenizer", None), tiktoken.Encoding): # for tiktoken tokenizer (Qwen)
            kwargs = dict(allowed_special="all")
        else:
            kwargs = dict(add_special_tokens=False)

        token_ids = []
        for elem in context:
            if isinstance(elem, str):
                elem = elem.replace("{{query}}", query, 1)
                elem = elem.replace("{{idx}}", idx, 1)
                token_ids = token_ids + tokenizer.encode(elem, **kwargs)
            elif isinstance(elem, dict):
                token_ids = token_ids + [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            else:
                raise NotImplementedError
        return token_ids


@dataclass
class Llama2Template(Template):

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        prefix: List[Union[str, Dict[str, str]]],
        history: List[Tuple[str, str]]
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: bos + prefix + query    resp + eos
        Turn t: bos + query             resp + eos
        """
        bos_ids, eos_ids = self._get_special_ids(tokenizer)
        encoded_pairs = []
        assert isinstance(prefix[0], str), "LLaMA-2 template only accepts list containing a single string."
        for turn_idx, (query, resp) in enumerate(history):
            if turn_idx == 0: # llama2 template has not sep_ids
                query = prefix[0] + query
            query_ids = self._convert_inputs_to_ids(tokenizer, context=self.prompt, query=query)
            resp_ids = self._convert_inputs_to_ids(tokenizer, context=[resp])
            encoded_pairs.append((bos_ids + query_ids, resp_ids + eos_ids))
        return encoded_pairs


templates: Dict[str, Template] = {}


def register_template(
    name: str,
    prefix: List[Union[str, Dict[str, str]]],
    prompt: List[Union[str, Dict[str, str]]],
    sep: List[Union[str, Dict[str, str]]],
    stop_words: List[str],
    use_history: bool
) -> None:
    template_class = Llama2Template if "llama2" in name else Template
    templates[name] = template_class(
        prefix=prefix,
        prompt=prompt,
        sep=sep,
        stop_words=stop_words,
        use_history=use_history
    )


def get_template_and_fix_tokenizer(
    name: str,
    tokenizer: "PreTrainedTokenizer"
) -> Template:
    template = templates.get(name, None)
    assert template is not None, "Template {} does not exist.".format(name)

    additional_special_tokens = template.stop_words

    if len(template.stop_words): # inplace method
        if tokenizer.eos_token_id is not None:
            additional_special_tokens.append(tokenizer.eos_token)

        tokenizer.eos_token = template.stop_words[0]
        additional_special_tokens.pop(0)
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = "<|endoftext|>"
        logger.info("Add eos token: {}".format(tokenizer.eos_token))

    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))

    tokenizer.add_special_tokens(dict(additional_special_tokens=additional_special_tokens))
    return template


r"""
Supports language model inference without histories.
"""
register_template(
    name="vanilla",
    prefix=[],
    prompt=[
        "{{query}}"
    ],
    sep=[],
    stop_words=[],
    use_history=False
)


r"""
Default template.
"""
register_template(
    name="default",
    prefix=[
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ],
    prompt=[
        "Human: {{query}}\nAssistant: "
    ],
    sep=[
        "\n"
    ],
    stop_words=[],
    use_history=True
)


r"""
Supports: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
          https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
          https://huggingface.co/meta-llama/Llama-2-70b-chat-hf
"""
register_template(
    name="llama2",
    prefix=[
        "<<SYS>>\nYou are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe.  "
        "Your answers should not include any harmful, unethical, "
        "racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n"
        "If a question does not make any sense, or is not factually coherent, "
        "explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n"
    ],
    prompt=[
        "[INST] {{query}} [/INST] "
    ],
    sep=[],
    stop_words=[],
    use_history=True
)


r"""
Supports: https://github.com/ymcui/Chinese-LLaMA-Alpaca-2
          https://huggingface.co/ziqingyang/chinese-alpaca-2-7b
"""
register_template(
    name="llama2_zh",
    prefix=[
        "<<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n"
    ],
    prompt=[
        "[INST] {{query}} [/INST] "
    ],
    sep=[],
    stop_words=[],
    use_history=True
)


r"""
Supports: https://huggingface.co/tatsu-lab/alpaca-7b-wdiff
          https://github.com/ymcui/Chinese-LLaMA-Alpaca
"""
register_template(
    name="alpaca",
    prefix=[
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    ],
    prompt=[
        "### Instruction:\n{{query}}\n\n### Response:\n"
    ],
    sep=[
        "\n\n"
    ],
    stop_words=[],
    use_history=True
)


r"""
Supports: https://huggingface.co/lmsys/vicuna-7b-delta-v1.1
          https://huggingface.co/lmsys/vicuna-13b-delta-v1.1
"""
register_template(
    name="vicuna",
    prefix=[
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ],
    prompt=[
        "USER: {{query}} ASSISTANT: "
    ],
    sep=[],
    stop_words=[],
    use_history=True
)


r"""
Supports: https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-13B
"""
register_template(
    name="belle",
    prefix=[],
    prompt=[
        "Human: {{query}}\n\nBelle: "
    ],
    sep=[
        "\n\n"
    ],
    stop_words=[],
    use_history=True
)


r"""
Supports: https://github.com/CVI-SZU/Linly
"""
register_template(
    name="linly",
    prefix=[],
    prompt=[
        "User: {{query}}\nBot: "
    ],
    sep=[
        "\n"
    ],
    stop_words=[],
    use_history=True
)


r"""
Supports: https://github.com/Neutralzz/BiLLa
"""
register_template(
    name="billa",
    prefix=[],
    prompt=[
        "Human: {{query}}\nAssistant: "
    ],
    sep=[
        "\n"
    ],
    stop_words=[],
    use_history=True
)


r"""
Supports: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1
"""
register_template(
    name="ziya",
    prefix=[],
    prompt=[
        {"token": "<human>"},
        ":{{query}}\n",
        {"token": "<bot>"},
        ":"
    ],
    sep=[
        "\n"
    ],
    stop_words=[],
    use_history=True
)


r"""
Supports: https://huggingface.co/qhduan/aquilachat-7b
"""
register_template(
    name="aquila",
    prefix=[
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
    ],
    prompt=[
        "Human: {{query}}###Assistant: "
    ],
    sep=[
        "###"
    ],
    stop_words=[],
    use_history=True
)


r"""
Supports: https://huggingface.co/internlm/internlm-chat-7b
"""
register_template(
    name="intern",
    prefix=[],
    prompt=[
        "<|User|>:{{query}}",
        {"token": "<eoh>"},
        "\n<|Bot|>:"
    ],
    sep=[
        "\n"
    ],
    stop_words=[
        "<eoa>"
    ],
    use_history=True
)


r"""
Supports: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat
"""
register_template(
    name="baichuan",
    prefix=[
        {"token": "<reserved_102>"} # user token (a little difference in the first turn)
    ],
    prompt=[
        "{{query}}",
        {"token": "<reserved_103>"} # assistant token
    ],
    sep=[],
    stop_words=[
        "<reserved_102>" # user token
    ],
    use_history=True
)


r"""
Supports: https://huggingface.co/HuggingFaceH4/starchat-alpha
          https://huggingface.co/HuggingFaceH4/starchat-beta
"""
register_template(
    name="starchat",
    prefix=[
        {"token": "<|system|>"},
        "\n",
        {"token": "<|end|>"}
    ],
    prompt=[
        {"token": "<|user|>"},
        "\n{{query}}",
        {"token": "<|end|>"},
        "\n",
        {"token": "<|assistant|>"}
    ],
    sep=[
        "\n"
    ],
    stop_words=[
        "<|end|>"
    ],
    use_history=True
)


r"""
Supports: https://huggingface.co/Qwen/Qwen-7B-Chat
"""
register_template(
    name="chatml",
    prefix=[
        {"token": "<|im_start|>"},
        "system\nYou are a helpful assistant.",
        {"token": "<|im_end|>"}
    ],
    prompt=[
        {"token": "<|im_start|>"},
        "user\n{{query}}",
        {"token": "<|im_end|>"},
        "\n",
        {"token": "<|im_start|>"},
        "assistant\n"
    ],
    sep=[
        "\n"
    ],
    stop_words=[
        "<|im_end|>"
    ],
    use_history=True
)


r"""
Supports: https://huggingface.co/THUDM/chatglm2-6b
"""
register_template(
    name="chatglm2",
    prefix=[
        {"token": "[gMASK]"},
        {"token": "sop"}
    ],
    prompt=[
        "[Round {{idx}}]\n\n问：{{query}}\n\n答："
    ],
    sep=[
        "\n\n"
    ],
    stop_words=[],
    use_history=True
)
