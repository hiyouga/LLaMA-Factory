import torch
import tiktoken
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple
from threading import Thread
from transformers import GenerationConfig, TextIteratorStreamer

from llmtuner.data.template import get_template_and_fix_tokenizer
from llmtuner.extras.misc import get_logits_processor
from llmtuner.model import dispatch_model, get_infer_args, load_model_and_tokenizer


@dataclass
class Response:

    response_text: str
    response_length: int
    prompt_length: int
    finish_reason: Literal["stop", "length"]


class ChatModel:

    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        model_args, data_args, finetuning_args, self.generating_args = get_infer_args(args)
        self.can_generate = (finetuning_args.stage == "sft")
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_args, finetuning_args, is_trainable=False, add_valuehead=(not self.can_generate)
        )
        self.tokenizer.padding_side = "left" if self.can_generate else "right"
        self.model = dispatch_model(self.model)
        self.template = get_template_and_fix_tokenizer(data_args.template, self.tokenizer)

    def _process_args(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None,
        **input_kwargs
    ) -> Tuple[Dict[str, Any], int]:
        prompt, _ = self.template.encode_oneturn(
            tokenizer=self.tokenizer, query=query, resp="", history=history, system=system
        )
        prompt_length = len(prompt)
        input_ids = torch.tensor([prompt], device=self.model.device)

        do_sample = input_kwargs.pop("do_sample", None)
        temperature = input_kwargs.pop("temperature", None)
        top_p = input_kwargs.pop("top_p", None)
        top_k = input_kwargs.pop("top_k", None)
        num_return_sequences = input_kwargs.pop("num_return_sequences", None)
        repetition_penalty = input_kwargs.pop("repetition_penalty", None)
        max_length = input_kwargs.pop("max_length", None)
        max_new_tokens = input_kwargs.pop("max_new_tokens", None)

        generating_args = self.generating_args.to_dict()
        generating_args.update(dict(
            do_sample=do_sample if do_sample is not None else generating_args["do_sample"],
            temperature=temperature or generating_args["temperature"],
            top_p=top_p or generating_args["top_p"],
            top_k=top_k or generating_args["top_k"],
            num_return_sequences=num_return_sequences or 1,
            repetition_penalty=repetition_penalty or generating_args["repetition_penalty"],
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            pad_token_id=self.tokenizer.pad_token_id
        ))

        if isinstance(num_return_sequences, int) and num_return_sequences > 1:
            generating_args["do_sample"] = True

        if max_length:
            generating_args.pop("max_new_tokens", None)
            generating_args["max_length"] = max_length

        if max_new_tokens:
            generating_args.pop("max_length", None)
            generating_args["max_new_tokens"] = max_new_tokens

        gen_kwargs = dict(
            inputs=input_ids,
            generation_config=GenerationConfig(**generating_args),
            logits_processor=get_logits_processor()
        )

        return gen_kwargs, prompt_length

    @torch.inference_mode()
    def chat(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None,
        **input_kwargs
    ) -> List[Response]:
        r"""
        Args: query, history, system, **input_kwargs

        Returns: [(response_text, prompt_length, response_length)] * n (default n=1)
        """
        gen_kwargs, prompt_length = self._process_args(query, history, system, **input_kwargs)
        generate_output = self.model.generate(**gen_kwargs)
        response_ids = generate_output[:, prompt_length:]
        response = self.tokenizer.batch_decode(
            response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        results = []
        for i in range(len(response)):
            eos_index = (response_ids[i] == self.tokenizer.eos_token_id).nonzero()
            response_length = (eos_index[0].item() + 1) if len(eos_index) else len(response_ids[i])
            results.append(Response(
                response_text=response[i],
                response_length=response_length,
                prompt_length=prompt_length,
                finish_reason="stop" if len(eos_index) else "length"
            ))

        return results

    @torch.inference_mode()
    def stream_chat(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None,
        **input_kwargs
    ) -> Generator[str, None, None]:
        gen_kwargs, _ = self._process_args(query, history, system, **input_kwargs)
        streamer = TextIteratorStreamer(self.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        yield from streamer

    @torch.inference_mode()
    def get_scores(
        self,
        batch_input: List[str],
        **input_kwargs
    ) -> List[float]:
        if isinstance(getattr(self.tokenizer, "tokenizer", None), tiktoken.Encoding): # for tiktoken tokenizer (Qwen)
            kwargs = dict(allowed_special="all")
        else:
            kwargs = dict(add_special_tokens=True)

        max_length = input_kwargs.pop("max_length", None)
        device = getattr(self.model.pretrained_model, "device", "cuda")

        inputs = self.tokenizer(
            batch_input,
            padding=True,
            truncation=True,
            max_length=max_length or getattr(self.model.config, "max_position_embeddings", 1024),
            pad_to_multiple_of=8,
            return_tensors="pt",
            **kwargs
        ).to(device)

        input_ids: torch.Tensor = inputs["input_ids"]
        _, _, values = self.model(**inputs, output_hidden_states=True, return_dict=True)

        if getattr(self.model.config, "model_type", None) == "chatglm":
            values = torch.transpose(values, 0, 1)

        scores = []
        for i in range(input_ids.size(0)):
            end_indexes = (input_ids[i] != self.tokenizer.pad_token_id).nonzero()
            end_index = end_indexes[-1].item() if len(end_indexes) else 0
            scores.append(values[i, end_index].nan_to_num().item())

        return scores
