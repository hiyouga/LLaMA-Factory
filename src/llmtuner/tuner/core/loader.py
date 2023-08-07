import os
import torch
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
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.deepspeed import is_deepspeed_zero3_enabled
from trl import AutoModelForCausalLMWithValueHead

from llmtuner.extras.logging import reset_logging, get_logger
from llmtuner.extras.misc import count_parameters, prepare_model_for_training
from llmtuner.extras.save_and_load import load_valuehead_params
from llmtuner.hparams import FinetuningArguments
from llmtuner.tuner.core.adapter import init_adapter

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from llmtuner.hparams import ModelArguments


logger = get_logger(__name__)


check_min_version("4.29.1")
require_version("datasets>=2.12.0", "To fix: pip install datasets>=2.12.0")
require_version("accelerate>=0.21.0", "To fix: pip install accelerate>=0.21.0")
require_version("peft>=0.4.0", "To fix: pip install peft>=0.4.0")
require_version("trl>=0.4.7", "To fix: pip install trl>=0.4.7")


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
    if (not is_trainable) and model_args.checkpoint_dir is None:
        logger.warning("Checkpoint is not found at evaluation, load the original model.")
        finetuning_args = FinetuningArguments(finetuning_type="none")

    assert stage in ["pt", "sft"] or finetuning_args.finetuning_type == "lora", \
        "RM and PPO training can only be performed with the LoRA method."

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        padding_side=model_args.padding_side,
        **config_kwargs
    )
    if tokenizer.eos_token_id is None: # fix qwen tokenizer
        tokenizer.eos_token = "<|endoftext|>"
    if tokenizer.pad_token_id is None: # add pad token
        tokenizer.pad_token = tokenizer.eos_token

    if model_args.checkpoint_dir is not None and finetuning_args.finetuning_type == "full":
        model_to_load = model_args.checkpoint_dir[0]
    else:
        model_to_load = model_args.model_name_or_path

    config = AutoConfig.from_pretrained(model_to_load, **config_kwargs)
    is_mergeable = True

    # Quantization configurations (using bitsandbytes library).
    if model_args.quantization_bit is not None:
        if model_args.quantization_bit == 8:
            require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")
            config_kwargs["load_in_8bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )

        elif model_args.quantization_bit == 4:
            require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
            config_kwargs["load_in_4bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_args.compute_dtype,
                bnb_4bit_use_double_quant=model_args.double_quantization,
                bnb_4bit_quant_type=model_args.quantization_type
            )

        is_mergeable = False
        config_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
        logger.info("Quantizing model to {} bit.".format(model_args.quantization_bit))

    # Load and prepare pretrained models (without valuehead).
    model = AutoModelForCausalLM.from_pretrained(
        model_to_load,
        config=config,
        torch_dtype=torch.bfloat16 if model_args.compute_dtype == torch.bfloat16 else torch.float16,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        **config_kwargs
    )

    # Register auto class to save the custom code files.
    if isinstance(config, PretrainedConfig) and "AutoConfig" in getattr(config, "auto_map", {}):
        config.__class__.register_for_auto_class()
    if isinstance(model, PreTrainedModel) and "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
        model.__class__.register_for_auto_class()
    if isinstance(tokenizer, PreTrainedTokenizerBase) and "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()

    # Initialize adapters
    model = prepare_model_for_training(model, finetuning_args.finetuning_type) if is_trainable else model
    model = init_adapter(model, model_args, finetuning_args, is_trainable, is_mergeable)

    if stage == "rm" or stage == "ppo": # add value head
        model: "AutoModelForCausalLMWithValueHead" = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        reset_logging()

        if stage == "rm" and model_args.checkpoint_dir is not None: # load valuehead weights to evaluate reward model
            logger.warning("Only the last checkpoint containing valuehead will be loaded as the valuehead.")
            if load_valuehead_params(model, model_args.checkpoint_dir[-1]):
                model.v_head.load_state_dict({
                    "summary.weight": getattr(model, "reward_head_weight"),
                    "summary.bias": getattr(model, "reward_head_bias")
                })

        if stage == "ppo": # load reward model
            assert is_trainable, "PPO stage cannot be performed at evaluation."
            assert model_args.reward_model is not None, "Reward model is necessary for PPO training."
            logger.info("Load reward model from {}".format(model_args.reward_model))
            model.pretrained_model.load_adapter(model_args.reward_model, "reward", is_trainable=False)
            assert load_valuehead_params(model, model_args.reward_model), "Reward model is not correctly loaded."

    if not is_trainable:
        model.requires_grad_(False) # fix all model params
        model = model.half() if model_args.quantization_bit is None else model # cast from fp32 to fp16

    trainable_params, all_param = count_parameters(model)
    logger.info("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    ))

    return model, tokenizer
