from typing import Optional
from dataclasses import dataclass


@dataclass
class Template:

    name: str

    def get_prompt(self, query: str, history: Optional[list] = None, prefix: Optional[str] = "") -> str:
        return getattr(self, "_format_{}".format(self.name))(query, history, prefix)

    def _format_alpaca(self, query: str, history: Optional[list], prefix: Optional[str] = "") -> str:
        if prefix:
            prompt = prefix
        else:
            prompt = "Below is an instruction that describes a task. "
            prompt += "Write a response that appropriately completes the request.\n"
            prompt += "Instruction:\n"
        if history:
            for old_query, response in history:
                prompt += "Human:{}\nAssistant:{}\n".format(old_query, response)
        prompt += "Human:{}\nAssistant:".format(query)
        return prompt

    def _format_vicuna(self, query: str, history: Optional[list], prefix: Optional[str] = "") -> str:
        if prefix:
            prompt = prefix
        else:
            prompt = "A chat between a curious user and an artificial intelligence assistant. "
            prompt += "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        if history:
            for old_query, response in history:
                prompt += "USER: {} ASSISTANT: {}</s>".format(old_query, response)
        prompt += "USER: {} ASSISTANT:".format(query)
        return prompt


    def _format_ziya(self, query: str, history: Optional[list], prefix: Optional[str] = "") -> str:
        prompt = prefix
        if history:
            for old_query, response in history:
                prompt += "<human>:{}\n<bot>:{}\n".format(old_query, response)
        prompt += "<human>:{}\n<bot>:".format(query)
        return prompt
