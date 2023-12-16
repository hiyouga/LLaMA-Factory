from typing import TYPE_CHECKING, Optional, Tuple
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils.versions import require_version
from trl import AutoModelForCausalLMWithValueHead

import llmtuner.model.patcher as patcher
from llmtuner.extras.logging import get_logger
from llmtuner.extras.misc import count_parameters, try_download_model_from_ms
from llmtuner.model.adapter import init_adapter
from llmtuner.model.utils import (
    load_valuehead_params, prepare_model_for_training, resize_embedding_layer, register_autoclass
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from llmtuner.hparams import ModelArguments, FinetuningArguments


logger = get_logger(__name__)


require_version("transformers>=4.36.1", "To fix: pip install transformers>=4.36.1")
require_version("datasets>=2.14.3", "To fix: pip install datasets>=2.14.3")
require_version("accelerate>=0.21.0", "To fix: pip install accelerate>=0.21.0")
require_version("peft>=0.7.0", "To fix: pip install peft>=0.7.0")
require_version("trl==0.7.4", "To fix: pip install trl==0.7.4")


def load_model_and_tokenizer(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: Optional[bool] = False,
    add_valuehead: Optional[bool] = False
) -> Tuple["PreTrainedModel", "PreTrainedTokenizer"]:
    r"""
    Loads pretrained model and tokenizer.

    Support both training and inference.
    """

    try_download_model_from_ms(model_args)

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
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    patcher.patch_tokenizer(tokenizer)
    patcher.patch_config(config, model_args)
    patcher.configure_rope(config, model_args, is_trainable)
    patcher.configure_flashattn(config_kwargs, model_args)
    patcher.configure_longlora(config, model_args, is_trainable)
    patcher.configure_quantization(config, config_kwargs, tokenizer, model_args, finetuning_args)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=model_args.compute_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        **config_kwargs
    )
    patcher.patch_model(model)
    register_autoclass(config, model, tokenizer)
    resize_embedding_layer(model, tokenizer)

    model = prepare_model_for_training(model=model, finetuning_args=finetuning_args) if is_trainable else model
    model = init_adapter(model, model_args, finetuning_args, is_trainable)

    if add_valuehead:
        model: "AutoModelForCausalLMWithValueHead" = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patcher.patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info("Loaded valuehead from checkpoint: {}".format(vhead_path))

    if not is_trainable:
        model.requires_grad_(False) # fix all model params
        model = model.to(model_args.compute_dtype) if not getattr(model, "quantization_method", None) else model
        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    logger.info("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    ))

    if not is_trainable:
        logger.info("This IS expected that the trainable params is 0 if you are using model for inference only.")

    return model, tokenizer
