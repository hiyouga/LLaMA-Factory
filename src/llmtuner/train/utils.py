from typing import TYPE_CHECKING, Optional, Union

import torch

from ..extras.logging import get_logger
from ..extras.packages import is_galore_available
from ..hparams import FinetuningArguments, ModelArguments
from ..model import load_model_and_tokenizer, load_valuehead_params


if is_galore_available():
    from galore_torch import GaLoreAdafactor, GaLoreAdamW, GaLoreAdamW8bit


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, Trainer
    from transformers.modeling_utils import PreTrainedModel
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import DataArguments


logger = get_logger(__name__)


def create_modelcard_and_push(
    trainer: "Trainer",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> None:
    kwargs = {
        "tasks": "text-generation",
        "finetuned_from": model_args.model_name_or_path,
        "dataset": [dataset.strip() for dataset in data_args.dataset.split(",")],
        "tags": ["llama-factory", finetuning_args.finetuning_type],
    }
    if not training_args.do_train:
        pass
    elif training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(license="other", **kwargs)  # prevent from connecting to hub


def create_ref_model(
    model_args: "ModelArguments", finetuning_args: "FinetuningArguments", add_valuehead: Optional[bool] = False
) -> Union["PreTrainedModel", "AutoModelForCausalLMWithValueHead"]:
    r"""
    Creates reference model for PPO/DPO training. Evaluation mode is not supported.

    The valuehead parameter is randomly initialized since it is useless for PPO training.
    """
    if finetuning_args.ref_model is not None:
        ref_model_args_dict = model_args.to_dict()
        ref_model_args_dict.update(
            dict(
                model_name_or_path=finetuning_args.ref_model,
                adapter_name_or_path=finetuning_args.ref_model_adapters,
                quantization_bit=finetuning_args.ref_model_quantization_bit,
            )
        )
        ref_model_args = ModelArguments(**ref_model_args_dict)
        ref_finetuning_args = FinetuningArguments(finetuning_type="lora")
        ref_model, _ = load_model_and_tokenizer(
            ref_model_args, ref_finetuning_args, is_trainable=False, add_valuehead=add_valuehead
        )
        logger.info("Created reference model from {}".format(finetuning_args.ref_model))
    else:
        if finetuning_args.finetuning_type == "lora":
            ref_model = None
        else:
            ref_model, _ = load_model_and_tokenizer(
                model_args, finetuning_args, is_trainable=False, add_valuehead=add_valuehead
            )
            logger.info("Created reference model from the model itself.")

    return ref_model


def create_reward_model(
    model: "AutoModelForCausalLMWithValueHead", model_args: "ModelArguments", finetuning_args: "FinetuningArguments"
) -> "AutoModelForCausalLMWithValueHead":
    r"""
    Creates reward model for PPO training.
    """
    if finetuning_args.reward_model_type == "api":
        assert finetuning_args.reward_model.startswith("http"), "Please provide full url."
        logger.info("Use reward server {}".format(finetuning_args.reward_model))
        return finetuning_args.reward_model
    elif finetuning_args.reward_model_type == "lora":
        model.pretrained_model.load_adapter(finetuning_args.reward_model, "reward")
        for name, param in model.named_parameters():  # https://github.com/huggingface/peft/issues/1090
            if "default" in name:
                param.data = param.data.to(torch.float32)  # trainable params should in fp32
        vhead_params = load_valuehead_params(finetuning_args.reward_model, model_args)
        assert vhead_params is not None, "Reward model is not correctly loaded."
        model.register_buffer("reward_head_weight", vhead_params["v_head.summary.weight"], persistent=False)
        model.register_buffer("reward_head_bias", vhead_params["v_head.summary.bias"], persistent=False)
        model.register_buffer(
            "default_head_weight", torch.zeros_like(vhead_params["v_head.summary.weight"]), persistent=False
        )
        model.register_buffer(
            "default_head_bias", torch.zeros_like(vhead_params["v_head.summary.bias"]), persistent=False
        )
        logger.info("Loaded adapter weights of reward model from {}".format(finetuning_args.reward_model))
        return None
    else:
        reward_model_args_dict = model_args.to_dict()
        reward_model_args_dict.update(
            dict(
                model_name_or_path=finetuning_args.reward_model,
                adapter_name_or_path=finetuning_args.reward_model_adapters,
                quantization_bit=finetuning_args.reward_model_quantization_bit,
            )
        )
        reward_model_args = ModelArguments(**reward_model_args_dict)
        reward_finetuning_args = FinetuningArguments(finetuning_type="lora")
        reward_model, _ = load_model_and_tokenizer(
            reward_model_args, reward_finetuning_args, is_trainable=False, add_valuehead=True
        )
        logger.info("Loaded full weights of reward model from {}".format(finetuning_args.reward_model))
        logger.warning("Please ensure the ppo model and reward model share SAME tokenizer and vocabulary.")
        return reward_model


def create_custom_optimzer(
    model: "PreTrainedModel", training_args: "Seq2SeqTrainingArguments", finetuning_args: "FinetuningArguments"
) -> Optional["torch.optim.Optimizer"]:
    if not finetuning_args.use_galore:
        return None

    galore_params = []
    galore_targets = finetuning_args.galore_target.split(",")

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(target in name for target in galore_targets):
            galore_params += list(filter(lambda p: p.requires_grad, module.parameters()))

    id_galore_params = [id(p) for p in galore_params]
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    non_galore_params = [p for p in trainable_params if id(p) not in id_galore_params]

    # define param groups as galore_params and non_galore_params
    param_groups = [
        {"params": non_galore_params},
        {
            "params": galore_params,
            "rank": finetuning_args.galore_rank,
            "update_proj_gap": finetuning_args.galore_update_interval,
            "scale": finetuning_args.galore_scale,
            "proj_type": finetuning_args.galore_proj_type,
        },
    ]
    if training_args.optim == "adamw_torch":
        optimizer = GaLoreAdamW(
            param_groups,
            lr=training_args.learning_rate,
            eps=training_args.adam_epsilon,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
        )
    elif training_args.optim in ["adamw_bnb_8bit", "adamw_8bit", "paged_adamw_8bit"]:
        optimizer = GaLoreAdamW8bit(
            param_groups,
            lr=training_args.learning_rate,
            eps=training_args.adam_epsilon,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
            optim_bits=8,
            is_paged="paged" in training_args.optim,
        )
    elif training_args.optim == "adafactor":
        optimizer = GaLoreAdafactor(
            param_groups,
            lr=training_args.learning_rate,
        )
    else:
        raise NotImplementedError("Unknow optim: {}".format(training_args.optim))

    logger.info("Using GaLore optimizer, may cause hanging at the start of training, wait patiently.")
    return optimizer
