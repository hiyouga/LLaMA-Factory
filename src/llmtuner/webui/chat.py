import os
from typing import List, Tuple

from llmtuner.chat.stream_chat import ChatModel
from llmtuner.extras.misc import torch_gc
from llmtuner.hparams import GeneratingArguments
from llmtuner.tuner import get_infer_args
from llmtuner.webui.common import get_model_path, get_save_dir
from llmtuner.webui.locales import ALERTS


class WebChatModel(ChatModel):

    def __init__(self, *args):
        self.model = None
        self.tokenizer = None
        self.generating_args = GeneratingArguments()
        if len(args) != 0:
            super().__init__(*args)

    def load_model(
        self,
        lang: str,
        model_name: str,
        checkpoints: List[str],
        finetuning_type: str,
        quantization_bit: str,
        template: str,
        source_prefix: str
    ):
        if self.model is not None:
            yield ALERTS["err_exists"][lang]
            return

        if not model_name:
            yield ALERTS["err_no_model"][lang]
            return

        model_name_or_path = get_model_path(model_name)
        if not model_name_or_path:
            yield ALERTS["err_no_path"][lang]
            return

        if checkpoints:
            checkpoint_dir = ",".join(
                [os.path.join(get_save_dir(model_name), finetuning_type, checkpoint) for checkpoint in checkpoints]
            )
        else:
            checkpoint_dir = None

        yield ALERTS["info_loading"][lang]
        args = dict(
            model_name_or_path=model_name_or_path,
            checkpoint_dir=checkpoint_dir,
            finetuning_type=finetuning_type,
            quantization_bit=int(quantization_bit) if quantization_bit else None,
            prompt_template=template,
            source_prefix=source_prefix
        )
        super().__init__(*get_infer_args(args))

        yield ALERTS["info_loaded"][lang]

    def unload_model(self, lang: str):
        yield ALERTS["info_unloading"][lang]
        self.model = None
        self.tokenizer = None
        torch_gc()
        yield ALERTS["info_unloaded"][lang]

    def predict(
        self,
        chatbot: List[Tuple[str, str]],
        query: str,
        history: List[Tuple[str, str]],
        max_new_tokens: int,
        top_p: float,
        temperature: float
    ):
        chatbot.append([query, ""])
        response = ""
        for new_text in self.stream_chat(
            query, history, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature
        ):
            response += new_text
            new_history = history + [(query, response)]
            chatbot[-1] = [query, response]
            yield chatbot, new_history
