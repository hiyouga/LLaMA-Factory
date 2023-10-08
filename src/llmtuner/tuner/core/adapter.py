import torch
from typing import TYPE_CHECKING

from peft import (
    PeftModel,
    TaskType,
    LoraConfig,
    get_peft_model
)

from llmtuner.extras.logging import get_logger
from llmtuner.tuner.core.utils import find_all_linear_modules

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from llmtuner.hparams import ModelArguments, FinetuningArguments


logger = get_logger(__name__)


def init_adapter(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    is_mergeable: bool
) -> "PreTrainedModel":
    r"""
    Initializes the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """

    if finetuning_args.finetuning_type == "none" and is_trainable:
        raise ValueError("You cannot use finetuning_type=none while training.")

    if finetuning_args.finetuning_type == "full" and is_trainable:
        logger.info("Fine-tuning method: Full")
        model = model.float()

    if finetuning_args.finetuning_type == "freeze":
        logger.info("Fine-tuning method: Freeze")
        num_layers = getattr(model.config, "num_layers")
        if finetuning_args.num_layer_trainable > 0: # fine-tuning the last n layers if num_layer_trainable > 0
            trainable_layer_ids = [num_layers - k - 1 for k in range(finetuning_args.num_layer_trainable)]
        else: # fine-tuning the first n layers if num_layer_trainable < 0
            trainable_layer_ids = [k for k in range(-finetuning_args.num_layer_trainable)]

        trainable_layers = ["{:d}.{}".format(idx, finetuning_args.name_module_trainable) for idx in trainable_layer_ids]
        for name, param in model.named_parameters():
            if not any(trainable_layer in name for trainable_layer in trainable_layers):
                param.requires_grad_(False)
            else:
                param.data = param.data.to(torch.float32)

    if finetuning_args.finetuning_type == "lora":
        logger.info("Fine-tuning method: LoRA")
        latest_checkpoint = None

        if model_args.checkpoint_dir is not None:
            if (is_trainable and finetuning_args.resume_lora_training) or (not is_mergeable): # continually fine-tuning
                checkpoints_to_merge, latest_checkpoint = model_args.checkpoint_dir[:-1], model_args.checkpoint_dir[-1]
            else:
                checkpoints_to_merge = model_args.checkpoint_dir

            for checkpoint in checkpoints_to_merge:
                model = PeftModel.from_pretrained(model, checkpoint)
                model = model.merge_and_unload()

            if len(checkpoints_to_merge) > 0:
                logger.info("Merged {} model checkpoint(s).".format(len(checkpoints_to_merge)))

            if latest_checkpoint is not None: # resume lora training or quantized inference
                model = PeftModel.from_pretrained(model, latest_checkpoint, is_trainable=is_trainable)

        if is_trainable and latest_checkpoint is None: # create new lora weights while training
            if len(finetuning_args.lora_target) == 1 and finetuning_args.lora_target[0] == "all":
                target_modules = find_all_linear_modules(model, model_args.quantization_bit)
            else:
                target_modules = finetuning_args.lora_target

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=finetuning_args.lora_rank,
                lora_alpha=finetuning_args.lora_alpha,
                lora_dropout=finetuning_args.lora_dropout,
                target_modules=target_modules,
                modules_to_save=finetuning_args.additional_target
            )
            model = get_peft_model(model, lora_config)
            if id(model.peft_config) != id(model.base_model.peft_config): # https://github.com/huggingface/peft/issues/923
                model.base_model.peft_config = model.peft_config

    if model_args.checkpoint_dir is not None:
        logger.info("Loaded fine-tuned model from checkpoint(s): {}".format(",".join(model_args.checkpoint_dir)))

    return model
