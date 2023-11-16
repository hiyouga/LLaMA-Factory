import torch
from typing import TYPE_CHECKING, Literal, Union

from llmtuner.extras.logging import get_logger
from llmtuner.hparams import ModelArguments, FinetuningArguments
from llmtuner.model import load_model_and_tokenizer, load_valuehead_params

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from trl import AutoModelForCausalLMWithValueHead


logger = get_logger(__name__)


def create_ref_model(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    stage: Literal["ppo", "dpo"]
) -> Union["PreTrainedModel", "AutoModelForCausalLMWithValueHead"]:
    r"""
    Creates reference model for PPO/DPO training. Evaluation mode is not supported.

    The valuehead parameter is randomly initialized since it is useless for PPO training.
    """
    if finetuning_args.ref_model is not None:
        ref_model_args_dict = model_args.to_dict()
        ref_model_args_dict.update(dict(
            model_name_or_path=finetuning_args.ref_model,
            checkpoint_dir=finetuning_args.ref_model_checkpoint,
            quantization_bit=finetuning_args.ref_model_quantization_bit
        ))
        ref_model_args = ModelArguments(**ref_model_args_dict)
        ref_finetuning_args = FinetuningArguments(finetuning_type="lora")
        ref_model, _ = load_model_and_tokenizer(ref_model_args, ref_finetuning_args, is_trainable=False, stage=stage)
        logger.info("Created reference model from {}".format(finetuning_args.ref_model))
    else:
        if finetuning_args.finetuning_type == "lora":
            ref_model = None
        else:
            ref_model, _ = load_model_and_tokenizer(model_args, finetuning_args, is_trainable=False, stage=stage)
            logger.info("Created reference model from the model itself.")

    return ref_model


def create_reward_model(
    model: "AutoModelForCausalLMWithValueHead",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments"
) -> "AutoModelForCausalLMWithValueHead":
    r"""
    Creates reward model for PPO training.
    """
    if finetuning_args.reward_model_type == "lora":
        model.pretrained_model.load_adapter(finetuning_args.reward_model, "reward")
        for name, param in model.named_parameters(): # https://github.com/huggingface/peft/issues/1090
            if "default" in name:
                param.data = param.data.to(torch.float32) # trainable params should in fp32
        vhead_params = load_valuehead_params(finetuning_args.reward_model, model_args)
        assert vhead_params is not None, "Reward model is not correctly loaded."
        model.register_buffer("reward_head_weight", vhead_params["v_head.summary.weight"], persistent=False)
        model.register_buffer("reward_head_bias", vhead_params["v_head.summary.bias"], persistent=False)
        model.register_buffer("default_head_weight", torch.zeros_like(vhead_params["v_head.summary.weight"]), persistent=False)
        model.register_buffer("default_head_bias", torch.zeros_like(vhead_params["v_head.summary.bias"]), persistent=False)
        logger.info("Loaded adapter weights of reward model from {}".format(finetuning_args.reward_model))
        return None
    else:
        reward_model_args_dict = model_args.to_dict()
        reward_model_args_dict.update(dict(
            model_name_or_path=finetuning_args.reward_model,
            checkpoint_dir=finetuning_args.reward_model_checkpoint,
            quantization_bit=finetuning_args.reward_model_quantization_bit
        ))
        reward_model_args = ModelArguments(**reward_model_args_dict)
        reward_finetuning_args = FinetuningArguments(finetuning_type="lora")
        reward_model, _ = load_model_and_tokenizer(reward_model_args, reward_finetuning_args, is_trainable=False, stage="ppo")
        logger.info("Load full weights of reward model from {}".format(finetuning_args.reward_model))
        logger.warning("Please ensure the ppo model and reward model share SAME tokenizer and vocabulary.")
        return reward_model
