import math
import os
import random
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import BitsAndBytesConfig, GPTQConfig, PreTrainedModel, PreTrainedTokenizerBase
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils.versions import require_version

from ..extras.constants import FILEEXT2TYPE, LAYERNORM_NAMES
from ..extras.logging import get_logger
from ..extras.misc import get_current_device, infer_optim_dtype
from ..extras.packages import is_flash_attn2_available
from ..extras.patches.llama_patch import apply_llama_patch
from .utils import QuantizationMethod, add_z3_leaf_module


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import ModelArguments


logger = get_logger(__name__)
SUPPORTED_CLASS_FOR_S2ATTN = ["llama"]


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


def _configure_attn_implementation(
    config: "PretrainedConfig", model_args: "ModelArguments", init_kwargs: Dict[str, Any]
) -> None:
    if model_args.flash_attn:
        if not is_flash_attn2_available():
            logger.warning("FlashAttention2 is not installed.")
            return

        logger.info("Using FlashAttention-2 for faster training and inference.")
        if getattr(config, "model_type", None) == "internlm2":  # special case for custom models
            setattr(config, "attn_implementation", "flash_attention_2")
        else:
            init_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        init_kwargs["attn_implementation"] = "eager"


def _configure_rope(config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool) -> None:
    if model_args.rope_scaling is None:
        return

    if not hasattr(config, "rope_scaling"):
        logger.warning("Current model does not support RoPE scaling.")
        return

    if is_trainable:
        if model_args.rope_scaling == "dynamic":
            logger.warning(
                "Dynamic NTK scaling may not work well with fine-tuning. "
                "See: https://github.com/huggingface/transformers/pull/24653"
            )

        current_max_length = getattr(config, "max_position_embeddings", None)
        if current_max_length and model_args.model_max_length > current_max_length:
            scaling_factor = float(math.ceil(model_args.model_max_length / current_max_length))
        else:
            logger.warning("Input length is smaller than max length. Consider increase input length.")
            scaling_factor = 1.0
    else:
        scaling_factor = 2.0

    setattr(config, "rope_scaling", {"type": model_args.rope_scaling, "factor": scaling_factor})
    logger.info(
        "Using {} scaling strategy and setting scaling factor to {}".format(model_args.rope_scaling, scaling_factor)
    )


def _configure_longlora(config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool) -> None:
    if not is_trainable or not model_args.shift_attn:
        return

    if getattr(config, "model_type", None) in SUPPORTED_CLASS_FOR_S2ATTN:
        setattr(config, "group_size_ratio", 0.25)
        apply_llama_patch()
        logger.info("Using shift short attention with group_size_ratio=1/4.")
    else:
        logger.warning("Current model does not support shift short attention.")


def _configure_quantization(
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

        init_kwargs["device_map"] = {"": get_current_device()}
        quantization_config: Dict[str, Any] = getattr(config, "quantization_config", None)
        quant_method = quantization_config.get("quant_method", "")

        if quant_method == QuantizationMethod.GPTQ:
            require_version("auto_gptq>=0.5.0", "To fix: pip install auto_gptq>=0.5.0")
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
                bnb_4bit_quant_storage=model_args.compute_dtype,  # crucial for fsdp qlora
            )

        if is_deepspeed_zero3_enabled() or model_args.quantization_device_map == "auto":
            if model_args.quantization_bit != 4:
                raise ValueError("Only 4-bit quantized model can use auto device map.")

            require_version("transformers>=4.39.0", "To fix: pip install transformers>=4.39.0")
            require_version("accelerate>=0.28.0", "To fix: pip install accelerate>=0.28.0")
            require_version("bitsandbytes>=0.43.0", "To fix: pip install bitsandbytes>=0.43.0")
        else:
            init_kwargs["device_map"] = {"": get_current_device()}

        logger.info("Quantizing model to {} bit.".format(model_args.quantization_bit))


def _noisy_mean_initialization(embed_weight: torch.Tensor, num_new_tokens: int):
    embedding_dim = embed_weight.size(1)
    avg_weight = embed_weight[:-num_new_tokens].mean(dim=0, keepdim=True)
    noise_weight = torch.empty_like(embed_weight[-num_new_tokens:])
    noise_weight.normal_(mean=0, std=(1.0 / math.sqrt(embedding_dim)))
    embed_weight[-num_new_tokens:] = avg_weight + noise_weight


def _resize_embedding_layer(model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer") -> None:
    r"""
    Resize token embeddings.
    """
    if is_deepspeed_zero3_enabled():
        import deepspeed  # type: ignore

        params = [model.get_input_embeddings().weight]
        if model.get_output_embeddings() is not None and not model.config.tie_word_embeddings:
            params.append(model.get_output_embeddings().weight)

        context_maybe_zero3 = deepspeed.zero.GatheredParameters(params, modifier_rank=0)
    else:
        context_maybe_zero3 = nullcontext()

    with context_maybe_zero3:
        current_embedding_size = model.get_input_embeddings().weight.size(0)

    if len(tokenizer) > current_embedding_size:
        if not isinstance(model.get_output_embeddings(), torch.nn.Linear):
            logger.warning("Current model does not support resizing token embeddings.")
            return

        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
        with context_maybe_zero3:
            new_embedding_size = model.get_input_embeddings().weight.size(0)
            num_new_tokens = new_embedding_size - current_embedding_size
            _noisy_mean_initialization(model.get_input_embeddings().weight.data, num_new_tokens)
            _noisy_mean_initialization(model.get_output_embeddings().weight.data, num_new_tokens)

        logger.info("Resized token embeddings from {} to {}.".format(current_embedding_size, new_embedding_size))


def _fp32_forward_post_hook(
    module: "torch.nn.Module", args: Tuple["torch.Tensor"], output: "torch.Tensor"
) -> "torch.Tensor":
    return output.to(torch.float32)


def _prepare_model_for_training(
    model: "PreTrainedModel", model_args: "ModelArguments", output_layer_name: str = "lm_head"
) -> None:
    r"""
    Includes:
        (1) cast the layernorm in fp32
        (2) make output embedding layer require grads
        (3) add the upcasting of the lm_head in fp32
    Inspired by: https://github.com/huggingface/peft/blob/v0.7.1/src/peft/utils/other.py#L72
    """
    if model_args.upcast_layernorm:
        logger.info("Upcasting layernorm weights in float32.")
        for name, param in model.named_parameters():
            if param.ndim == 1 and any(ln_name in name for ln_name in LAYERNORM_NAMES):
                param.data = param.data.to(torch.float32)

    if not model_args.disable_gradient_checkpointing:
        if not getattr(model, "supports_gradient_checkpointing", False):
            logger.warning("Current model does not support gradient checkpointing.")
        else:
            # use_reentrant=False might increase VRAM usage (have not been empirically verified yet)
            # According to: https://github.com/huggingface/transformers/issues/28339
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
            model.enable_input_require_grads()
            setattr(model.config, "use_cache", False)  # turn off when gradient checkpointing is enabled
            logger.info("Gradient checkpointing enabled.")

    if hasattr(model, output_layer_name) and model_args.upcast_lmhead_output:
        logger.info("Upcasting lm_head outputs in float32.")
        output_layer = getattr(model, output_layer_name)
        if isinstance(output_layer, torch.nn.Linear) and output_layer.weight.dtype != torch.float32:
            output_layer.register_forward_hook(_fp32_forward_post_hook)


def patch_tokenizer(tokenizer: "PreTrainedTokenizer") -> None:
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)


def patch_config(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    init_kwargs: Dict[str, Any],
    is_trainable: bool,
) -> None:
    if model_args.compute_dtype is None:  # priority: bf16 > fp16 > fp32
        model_args.compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))

    _configure_attn_implementation(config, model_args, init_kwargs)
    _configure_rope(config, model_args, is_trainable)
    _configure_longlora(config, model_args, is_trainable)
    _configure_quantization(config, tokenizer, model_args, init_kwargs)

    if model_args.use_cache and not is_trainable:
        setattr(config, "use_cache", True)
        logger.info("Using KV cache for faster generation.")

    if model_args.moe_aux_loss_coef is not None:
        if getattr(config, "model_type", None) in ["mixtral", "qwen2_moe"]:
            setattr(config, "router_aux_loss_coef", model_args.moe_aux_loss_coef)
        elif getattr(config, "model_type", None) == "deepseek":
            setattr(config, "aux_loss_alpha", model_args.moe_aux_loss_coef)

    if getattr(config, "model_type", None) == "qwen":
        setattr(config, "use_flash_attn", model_args.flash_attn)
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, model_args.compute_dtype == dtype)

    if getattr(config, "model_type", None) == "qwen2" and is_trainable and model_args.flash_attn:
        setattr(config, "use_cache", False)  # qwen2 does not support use_cache when using flashattn

    if getattr(config, "model_type", None) == "qwen2_moe" and is_trainable:
        setattr(config, "output_router_logits", True)

    init_kwargs["torch_dtype"] = model_args.compute_dtype
    if not is_deepspeed_zero3_enabled():
        init_kwargs["low_cpu_mem_usage"] = model_args.low_cpu_mem_usage
        if init_kwargs["low_cpu_mem_usage"]:
            if "device_map" not in init_kwargs:
                init_kwargs["device_map"] = model_args.device_map or {"": get_current_device()}

            if init_kwargs["device_map"] == "auto":
                init_kwargs["offload_folder"] = model_args.offload_folder


def patch_model(
    model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer", model_args: "ModelArguments", is_trainable: bool
) -> None:
    gen_config = model.generation_config  # check and fix generation config
    if not gen_config.do_sample and (
        (gen_config.temperature is not None and gen_config.temperature != 1.0)
        or (gen_config.top_p is not None and gen_config.top_p != 1.0)
        or (gen_config.typical_p is not None and gen_config.typical_p != 1.0)
    ):
        gen_config.do_sample = True

    if "GenerationMixin" not in str(model.generate.__func__):
        model.generate = MethodType(PreTrainedModel.generate, model)

    if is_trainable and getattr(model.config, "model_type", None) == "chatglm":
        setattr(model, "lm_head", model.transformer.output_layer)
        setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])

    if model_args.resize_vocab:
        _resize_embedding_layer(model, tokenizer)

    if is_trainable:
        _prepare_model_for_training(model, model_args)

    if getattr(model.config, "model_type", None) == "mixtral":
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

        add_z3_leaf_module(model, MixtralSparseMoeBlock)

    if getattr(model.config, "model_type", None) == "qwen2moe":
        from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

        add_z3_leaf_module(model, Qwen2MoeSparseMoeBlock)

    try:
        model.add_model_tags(["llama-factory"])
    except Exception:
        logger.warning("Cannot properly tag the model.")


def patch_valuehead_model(model: "AutoModelForCausalLMWithValueHead") -> None:
    def tie_weights(self: "AutoModelForCausalLMWithValueHead") -> None:
        if isinstance(self.pretrained_model, PreTrainedModel):
            self.pretrained_model.tie_weights()

    def get_input_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_input_embeddings()

    def create_or_update_model_card(self: "AutoModelForCausalLMWithValueHead", output_dir: str) -> None:
        if isinstance(self.pretrained_model, PeftModel):
            self.pretrained_model.create_or_update_model_card(output_dir)

    ignore_modules = [name for name, _ in model.named_parameters() if "pretrained_model" in name]
    setattr(model, "_keys_to_ignore_on_save", ignore_modules)
    setattr(model, "tie_weights", MethodType(tie_weights, model))
    setattr(model, "get_input_embeddings", MethodType(get_input_embeddings, model))
    setattr(model, "create_or_update_model_card", MethodType(create_or_update_model_card, model))
