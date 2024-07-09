import os
import random
from enum import Enum, unique
from typing import TYPE_CHECKING, Any, Dict, List

import torch
from datasets import load_dataset
from transformers import BitsAndBytesConfig, GPTQConfig
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled
from transformers.utils.versions import require_version

from ...extras.constants import FILEEXT2TYPE
from ...extras.logging import get_logger
from ...extras.misc import get_current_device


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer

    from ...hparams import ModelArguments


logger = get_logger(__name__)


@unique
class QuantizationMethod(str, Enum):
    r"""
    Borrowed from `transformers.utils.quantization_config.QuantizationMethod`.
    """

    BITS_AND_BYTES = "bitsandbytes"
    GPTQ = "gptq"
    AWQ = "awq"
    AQLM = "aqlm"
    QUANTO = "quanto"
    EETQ = "eetq"
    HQQ = "hqq"


def _get_quantization_dataset(tokenizer: "PreTrainedTokenizer", model_args: "ModelArguments") -> List[str]:
    r"""
    Inspired by: https://github.com/huggingface/optimum/blob/v1.16.0/optimum/gptq/data.py#L133
    TODO: remove tokenizer.decode() https://github.com/huggingface/optimum/pull/1600
    """
    if os.path.isfile(model_args.export_quantization_dataset):
        data_path = FILEEXT2TYPE.get(model_args.export_quantization_dataset.split(".")[-1], None)
        data_files = model_args.export_quantization_dataset
    else:
        data_path = model_args.export_quantization_dataset
        data_files = None

    dataset = load_dataset(path=data_path, data_files=data_files, split="train", cache_dir=model_args.cache_dir)
    maxlen = model_args.export_quantization_maxlen

    samples = []
    for _ in range(model_args.export_quantization_nsamples):
        while True:
            sample_idx = random.randint(0, len(dataset) - 1)
            sample: Dict[str, torch.Tensor] = tokenizer(dataset[sample_idx]["text"], return_tensors="pt")
            if sample["input_ids"].size(1) >= maxlen:
                break  # TODO: fix large maxlen

        word_idx = random.randint(0, sample["input_ids"].size(1) - maxlen - 1)
        input_ids = sample["input_ids"][:, word_idx : word_idx + maxlen]
        samples.append(tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True))

    return samples


def configure_quantization(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    init_kwargs: Dict[str, Any],
) -> None:
    r"""
    Priority: PTQ-quantized (training) > AutoGPTQ (export) > Bitsandbytes (training)
    """
    if getattr(config, "quantization_config", None):  # ptq
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantized models.")

        if model_args.quantization_device_map != "auto":
            init_kwargs["device_map"] = {"": get_current_device()}

        quantization_config: Dict[str, Any] = getattr(config, "quantization_config", None)
        quant_method = quantization_config.get("quant_method", "")

        if quant_method == QuantizationMethod.GPTQ:
            require_version("auto_gptq>=0.5.0", "To fix: pip install auto_gptq>=0.5.0")
            quantization_config.pop("disable_exllama", None)  # remove deprecated args
            quantization_config["use_exllama"] = False  # disable exllama

        if quant_method == QuantizationMethod.AWQ:
            require_version("autoawq", "To fix: pip install autoawq")

        if quant_method == QuantizationMethod.AQLM:
            require_version("transformers>=4.39.0", "To fix: pip install transformers>=4.39.0")
            require_version("aqlm>=1.1.0", "To fix: pip install aqlm[gpu]>=1.1.0")
            quantization_config["bits"] = 2

        quant_bits = quantization_config.get("bits", "?")
        logger.info("Loading {}-bit {}-quantized model.".format(quant_bits, quant_method.upper()))

    elif model_args.export_quantization_bit is not None:  # auto-gptq
        require_version("optimum>=1.16.0", "To fix: pip install optimum>=1.16.0")
        require_version("auto_gptq>=0.5.0", "To fix: pip install auto_gptq>=0.5.0")
        from accelerate.utils import get_max_memory

        if getattr(config, "model_type", None) == "chatglm":
            raise ValueError("ChatGLM model is not supported.")

        init_kwargs["quantization_config"] = GPTQConfig(
            bits=model_args.export_quantization_bit,
            tokenizer=tokenizer,
            dataset=_get_quantization_dataset(tokenizer, model_args),
        )
        init_kwargs["device_map"] = "auto"
        init_kwargs["max_memory"] = get_max_memory()
        logger.info("Quantizing model to {} bit.".format(model_args.export_quantization_bit))

    elif model_args.quantization_bit is not None:  # bnb
        if model_args.quantization_bit == 8:
            require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")
            init_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        elif model_args.quantization_bit == 4:
            require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
            init_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_args.compute_dtype,
                bnb_4bit_use_double_quant=model_args.double_quantization,
                bnb_4bit_quant_type=model_args.quantization_type,
                bnb_4bit_quant_storage=model_args.compute_dtype,  # crucial for fsdp+qlora
            )

        if is_deepspeed_zero3_enabled() or is_fsdp_enabled() or model_args.quantization_device_map == "auto":
            if model_args.quantization_bit != 4:
                raise ValueError("Only 4-bit quantized model can use auto device map.")

            require_version("transformers>=4.39.0", "To fix: pip install transformers>=4.39.0")
            require_version("accelerate>=0.28.0", "To fix: pip install accelerate>=0.28.0")
            require_version("bitsandbytes>=0.43.0", "To fix: pip install bitsandbytes>=0.43.0")
            init_kwargs["torch_dtype"] = model_args.compute_dtype  # fsdp+qlora requires same dtype
        else:
            init_kwargs["device_map"] = {"": get_current_device()}

        logger.info("Quantizing model to {} bit.".format(model_args.quantization_bit))
