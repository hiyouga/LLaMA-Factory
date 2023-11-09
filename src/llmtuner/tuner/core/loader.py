import os
import math
import torch
from types import MethodType
from typing import TYPE_CHECKING, Literal, Optional, Tuple

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase
)
from transformers.models.llama import modeling_llama as LlamaModule
from transformers.utils.versions import require_version
from peft import PeftModel
from trl import AutoModelForCausalLMWithValueHead

try:
    from transformers.integrations import is_deepspeed_zero3_enabled
except ImportError: # https://github.com/huggingface/transformers/releases/tag/v4.33.1
    from transformers.deepspeed import is_deepspeed_zero3_enabled

from llmtuner.extras.logging import reset_logging, get_logger
from llmtuner.extras.misc import count_parameters, infer_optim_dtype
from llmtuner.extras.patches import llama_patch as LlamaPatches
from llmtuner.hparams import FinetuningArguments
from llmtuner.tuner.core.adapter import init_adapter, load_valuehead_params
from llmtuner.tuner.core.utils import prepare_model_for_training

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from llmtuner.hparams import ModelArguments


logger = get_logger(__name__)


require_version("transformers>=4.31.0,<4.35.0", "To fix: pip install \"transformers>=4.31.0,<4.35.0\"")
require_version("datasets>=2.14.0", "To fix: pip install datasets>=2.14.0")
require_version("accelerate>=0.21.0", "To fix: pip install accelerate>=0.21.0")
require_version("peft>=0.6.0", "To fix: pip install peft>=0.6.0")
require_version("trl==0.7.2", "To fix: pip install trl==0.7.2")


def load_model_and_tokenizer(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: Optional[bool] = False,
    stage: Optional[Literal["pt", "sft", "rm", "ppo"]] = "sft"
) -> Tuple[PreTrainedModel, "PreTrainedTokenizer"]:
    r"""
    Loads pretrained model and tokenizer.

    Support both training and inference.
    """

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        split_special_tokens=model_args.split_special_tokens,
        padding_side="right", # training with left-padded tensors in fp16 precision may cause overflow
        **config_kwargs
    )

    if finetuning_args.finetuning_type != "lora" and model_args.checkpoint_dir is not None:
        model_to_load = model_args.checkpoint_dir[0]
    else:
        model_to_load = model_args.model_name_or_path

    config = AutoConfig.from_pretrained(model_to_load, **config_kwargs)

    # Fix tokenizer (for ChatGLM2 and ChatGLM3)
    if getattr(config, "model_type", None) == "chatglm":
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    # Set model dtype
    if model_args.compute_dtype is not None: # for training
        setattr(config, "torch_dtype", model_args.compute_dtype)
    else: # for evaluation, priority: bf16 > fp16 > fp32
        model_args.compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))

    # Fix config (for Qwen)
    if getattr(config, "model_type", None) == "qwen":
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, getattr(config, "torch_dtype", None) == dtype)

    # Set RoPE scaling
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

    # Set FlashAttention-2
    if model_args.flash_attn:
        if getattr(config, "model_type", None) == "llama":
            LlamaModule.LlamaAttention = LlamaPatches.LlamaFlashAttention2
            LlamaModule.LlamaModel._prepare_decoder_attention_mask = LlamaPatches._prepare_decoder_attention_mask
            logger.info("Using FlashAttention-2 for faster training and inference.")
        elif getattr(config, "model_type", None) in ["qwen", "Yi"]:
            logger.info("Current model automatically enables FlashAttention if installed.")
        else:
            logger.warning("Current model does not support FlashAttention-2.")
    elif is_trainable and model_args.shift_attn and getattr(config, "model_type", None) == "llama":
        LlamaModule.LlamaAttention = LlamaPatches.LlamaShiftShortAttention
        logger.warning("Using `--flash_attn` for faster training in large context length.")

    # Set shift short attention (S^2-Attn)
    if is_trainable and model_args.shift_attn:
        if getattr(config, "model_type", None) == "llama":
            setattr(config, "group_size_ratio", 0.25)
            logger.info("Using shift short attention with group_size_ratio=1/4.")
        else:
            logger.warning("Current model does not support shift short attention.")

    # Quantization configurations (using bitsandbytes library).
    if model_args.quantization_bit is not None:
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        if model_args.quantization_bit == 8:
            require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")
            config_kwargs["load_in_8bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        if model_args.quantization_bit == 4:
            require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
            config_kwargs["load_in_4bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_args.compute_dtype,
                bnb_4bit_use_double_quant=model_args.double_quantization,
                bnb_4bit_quant_type=model_args.quantization_type
            )

        config_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))} if is_trainable else "auto"
        logger.info("Quantizing model to {} bit.".format(model_args.quantization_bit))

    # Load and prepare pre-trained models (without valuehead).
    model = AutoModelForCausalLM.from_pretrained(
        model_to_load,
        config=config,
        torch_dtype=model_args.compute_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        **config_kwargs
    )

    # Disable custom generate method (for Qwen and Baichuan2)
    if isinstance(model, PreTrainedModel) and "GenerationMixin" not in str(model.generate.__func__):
        model.generate = MethodType(PreTrainedModel.generate, model)

    # Fix LM head (for ChatGLM2 and ChatGLM3)
    if getattr(config, "model_type", None) == "chatglm":
        setattr(model, "lm_head", model.transformer.output_layer)
        setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])

    # Register auto class to save the custom code files.
    if isinstance(config, PretrainedConfig) and "AutoConfig" in getattr(config, "auto_map", {}):
        config.__class__.register_for_auto_class()
    if isinstance(model, PreTrainedModel) and "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
        model.__class__.register_for_auto_class()
    if isinstance(tokenizer, PreTrainedTokenizerBase) and "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()

    # Initialize adapters
    model = prepare_model_for_training(model=model, finetuning_args=finetuning_args) if is_trainable else model
    model = init_adapter(model, model_args, finetuning_args, is_trainable)
    model = model.train() if is_trainable else model.eval()

    # Prepare model with valuehead for RLHF
    if stage == "rm" or stage == "ppo":
        model: "AutoModelForCausalLMWithValueHead" = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        reset_logging()
        if stage == "rm" and model_args.checkpoint_dir is not None: # load valuehead weights to evaluate reward model
            logger.warning("Only the last checkpoint containing valuehead will be loaded.")
            if load_valuehead_params(model, model_args):
                model.v_head.load_state_dict({
                    "summary.weight": getattr(model, "reward_head_weight"),
                    "summary.bias": getattr(model, "reward_head_bias")
                })

        if stage == "ppo": # load reward model
            logger.info("Load reward model from {}".format(model_args.reward_model))
            if isinstance(model.pretrained_model, PeftModel):
                model.pretrained_model.load_adapter(model_args.reward_model, "reward")
            for name, param in model.named_parameters(): # https://github.com/huggingface/peft/issues/1090
                if "default" in name:
                    param.data = param.data.to(torch.float32) # trainable params should in fp32
            assert load_valuehead_params(model, model_args), "Reward model is not correctly loaded."

    # Prepare model for inference
    if not is_trainable:
        model.requires_grad_(False) # fix all model params
        model = model.to(model_args.compute_dtype) if model_args.quantization_bit is None else model

    trainable_params, all_param = count_parameters(model)
    logger.info("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    ))

    if not is_trainable:
        logger.info("This IS expected that the trainable params is 0 if you are using model for inference only.")

    return model, tokenizer
