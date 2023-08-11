import os
import torch
from typing import TYPE_CHECKING

from peft import (
    PeftModel,
    TaskType,
    LoraConfig,
    get_peft_model
)
from peft.utils import CONFIG_NAME, WEIGHTS_NAME

from llmtuner.extras.logging import get_logger
from llmtuner.extras.save_and_load import load_trainable_params

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

        for name, param in model.named_parameters():
            if not any(trainable_layer in name for trainable_layer in finetuning_args.trainable_layers):
                param.requires_grad_(False)
            else:
                param.data = param.data.to(torch.float32)

        if model_args.checkpoint_dir is not None:
            assert load_trainable_params(model, model_args.checkpoint_dir[0]), "Model checkpoint is not correctly loaded."

    if finetuning_args.finetuning_type == "lora":
        logger.info("Fine-tuning method: LoRA")
        latest_checkpoint = None

        if model_args.checkpoint_dir is not None:
            assert os.path.exists(os.path.join(model_args.checkpoint_dir[0], WEIGHTS_NAME)), \
                "Provided path ({}) does not contain a LoRA weight.".format(model_args.checkpoint_dir[0])
            assert os.path.exists(os.path.join(model_args.checkpoint_dir[0], CONFIG_NAME)), \
                "The given checkpoint may be not a LoRA checkpoint, please specify `--finetuning_type full/freeze` instead."

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
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=finetuning_args.lora_rank,
                lora_alpha=finetuning_args.lora_alpha,
                lora_dropout=finetuning_args.lora_dropout,
                target_modules=finetuning_args.lora_target
            )
            model = get_peft_model(model, lora_config)

    if model_args.checkpoint_dir is not None:
        logger.info("Loaded fine-tuned model from checkpoint(s): {}".format(",".join(model_args.checkpoint_dir)))

    return model
