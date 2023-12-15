import math
import torch
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict

from transformers import BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizerBase
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils.versions import require_version

from llmtuner.extras.logging import get_logger
from llmtuner.extras.misc import get_current_device, infer_optim_dtype
from llmtuner.extras.packages import is_flash_attn2_available

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer
    from trl import AutoModelForCausalLMWithValueHead
    from llmtuner.hparams import ModelArguments


logger = get_logger(__name__)
SUPPORTED_CLASS_FOR_S2ATTN = [] # TODO: add llama


def configure_flashattn(config_kwargs: Dict[str, Any], model_args: "ModelArguments"):
    if model_args.flash_attn and is_flash_attn2_available():
        config_kwargs["use_flash_attention_2"] = True
        logger.info("Using FlashAttention-2 for faster training and inference.")


def configure_longlora(config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool):
    if is_trainable and model_args.shift_attn:
        if getattr(config, "model_type", None) in SUPPORTED_CLASS_FOR_S2ATTN:
            setattr(config, "group_size_ratio", 0.25)
            logger.info("Using shift short attention with group_size_ratio=1/4.")
        else:
            logger.warning("Current model does not support shift short attention.")


def configure_quantization(config: "PretrainedConfig", config_kwargs: Dict[str, Any], model_args: "ModelArguments"):
    if getattr(config, "quantization_config", None): # gptq or awq
        model_args.quantization_bit = None # remove bnb quantization
        config_kwargs["device_map"] = {"": get_current_device()}
        quantization_config = getattr(config, "quantization_config", None)
        logger.info("Loading {}-bit pre-quantized model.".format(quantization_config.get("bits", -1)))

    if model_args.quantization_bit is not None: # bnb
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        if model_args.quantization_bit == 8:
            require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")
            config_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        if model_args.quantization_bit == 4:
            require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_args.compute_dtype,
                bnb_4bit_use_double_quant=model_args.double_quantization,
                bnb_4bit_quant_type=model_args.quantization_type
            )

        config_kwargs["device_map"] = {"": get_current_device()}
        logger.info("Quantizing model to {} bit.".format(model_args.quantization_bit))


def configure_rope(config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool):
    if model_args.rope_scaling is not None:
        if not hasattr(config, "rope_scaling"):
            logger.warning("Current model does not support RoPE scaling.")
        else:
            if is_trainable:
                if model_args.rope_scaling == "dynamic":
                    logger.warning(
                        "Dynamic NTK may not work well with fine-tuning. "
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
            logger.info("Using {} scaling strategy and setting scaling factor to {}".format(
                model_args.rope_scaling, scaling_factor
            ))


def patch_config(config: "PretrainedConfig", model_args: "ModelArguments"):
    if model_args.compute_dtype is None: # priority: bf16 > fp16 > fp32
        model_args.compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))
    setattr(config, "torch_dtype", model_args.compute_dtype)

    if getattr(config, "model_type", None) == "qwen":
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, getattr(config, "torch_dtype", None) == dtype)


def patch_model(model: "PreTrainedModel"):
    if "GenerationMixin" not in str(model.generate.__func__):
        model.generate = MethodType(PreTrainedModel.generate, model)

    if getattr(model.config, "model_type", None) == "chatglm":
        setattr(model, "lm_head", model.transformer.output_layer)
        setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])


def patch_valuehead_model(model: "AutoModelForCausalLMWithValueHead"):
    def get_input_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        return self.pretrained_model.get_input_embeddings()

    setattr(model, "get_input_embeddings", MethodType(get_input_embeddings, model))
    ignore_modules = [name for name, _ in model.named_parameters() if "pretrained_model" in name]
    setattr(model, "_keys_to_ignore_on_save", ignore_modules)
    setattr(model, "tie_weights", MethodType(lambda _: None, model)) # use empty method


def patch_tokenizer(tokenizer: "PreTrainedTokenizer"):
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)
