from typing import Optional
from dataclasses import dataclass


@dataclass
class Template:

    name: str

    def __post_init__(self):
        assert hasattr(self, "_format_{}".format(self.name)), "Template {} does not exist.".format(self.name)

    def get_prompt(self, query: str, history: Optional[list] = None, prefix: Optional[str] = "") -> str:
        return getattr(self, "_format_{}".format(self.name))(query, history, prefix)

    def _format_vanilla(self, query: str, history: Optional[list], prefix: Optional[str] = "") -> str:
        prompt = prefix
        if history:
            for old_query, response in history:
                prompt += old_query + "\n" + response + "\n"
        prompt += query
        return prompt

    def _format_alpaca(self, query: str, history: Optional[list], prefix: Optional[str] = "") -> str:
        r"""
        Supports: https://huggingface.co/tatsu-lab/alpaca-7b-wdiff
        """
        if prefix:
            prompt = prefix
        else:
            prompt = "Below is an instruction that describes a task. "
            prompt += "Write a response that appropriately completes the request.\n\n"
            prompt += "Instruction:\n"
        if history:
            for old_query, response in history:
                prompt += "Human:\n{}\n\nAssistant:\n{}\n\n".format(old_query, response)
        prompt += "Human:\n{}\n\nAssistant:".format(query)
        return prompt

    def _format_vicuna(self, query: str, history: Optional[list], prefix: Optional[str] = "") -> str:
        r"""
        Supports: https://huggingface.co/lmsys/vicuna-7b-delta-v1.1
                  https://huggingface.co/lmsys/vicuna-13b-delta-v1.1
        """
        if prefix:
            prompt = prefix
        else:
            prompt = "A chat between a curious user and an artificial intelligence assistant. "
            prompt += "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        if history:
            for old_query, response in history:
                prompt += "USER: {} ASSISTANT: {}</s>".format(old_query, response)
        prompt += "USER: {} ASSISTANT: ".format(query)
        return prompt

    def _format_belle(self, query: str, history: Optional[list], prefix: Optional[str] = "") -> str:
        r"""
        Supports: https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-13B
        """
        prompt = prefix
        if history:
            for old_query, response in history:
                prompt += "Human: {}\n\nBelle: {}\n\n".format(old_query, response)
        prompt += "Human: {}\n\nBelle: ".format(query)
        return prompt

    def _format_linly(self, query: str, history: Optional[list], prefix: Optional[str] = "") -> str:
        r"""
        Supports: https://github.com/CVI-SZU/Linly
        """
        prompt = prefix
        if history:
            for old_query, response in history:
                prompt += "User: {}\nBot: {}\n".format(old_query, response)
        prompt += "User: {}\nBot: ".format(query)
        return prompt

    def _format_billa(self, query: str, history: Optional[list], prefix: Optional[str] = "") -> str:
        r"""
        Supports: https://github.com/Neutralzz/BiLLa
        """
        prompt = prefix
        if history:
            for old_query, response in history:
                prompt += "Human: {}\nAssistant: {}\n".format(old_query, response)
        prompt += "Human: {}\nAssistant: ".format(query)
        return prompt

    def _format_ziya(self, query: str, history: Optional[list], prefix: Optional[str] = "") -> str:
        r"""
        Supports: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1
        """
        prompt = prefix
        if history:
            for old_query, response in history:
                prompt += "<human>:{}\n<bot>:{}\n".format(old_query, response)
        prompt += "<human>:{}\n<bot>:".format(query)
        return prompt
