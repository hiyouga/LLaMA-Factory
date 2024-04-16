from typing import TYPE_CHECKING

import torch
from peft import LoraConfig, LoraModel, PeftModel, TaskType, get_peft_model
from transformers.integrations import is_deepspeed_zero3_enabled

from ..extras.logging import get_logger
from .utils import QuantizationMethod, find_all_linear_modules, find_expanded_modules


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

    from ..hparams import FinetuningArguments, ModelArguments


logger = get_logger(__name__)


def init_adapter(
    model: "PreTrainedModel", model_args: "ModelArguments", finetuning_args: "FinetuningArguments", is_trainable: bool
) -> "PreTrainedModel":
    r"""
    Initializes the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """

    if (not is_trainable) and model_args.adapter_name_or_path is None:
        logger.info("Adapter is not found at evaluation, load the base model.")
        return model

    if finetuning_args.finetuning_type != "lora" and getattr(model, "quantization_method", None):
        raise ValueError("You can only use lora for quantized models.")

    if finetuning_args.finetuning_type == "full" and is_trainable:
        logger.info("Fine-tuning method: Full")
        if (not finetuning_args.pure_bf16) and (not finetuning_args.use_badam):
            model = model.float()

    if finetuning_args.finetuning_type == "freeze" and is_trainable:
        logger.info("Fine-tuning method: Freeze")
        num_layers = (
            getattr(model.config, "num_hidden_layers", None)
            or getattr(model.config, "num_layers", None)
            or getattr(model.config, "n_layer", None)
        )
        if not num_layers:
            raise ValueError("Current model does not support freeze tuning.")

        if finetuning_args.use_llama_pro:
            if num_layers % finetuning_args.num_layer_trainable != 0:
                raise ValueError(
                    "`num_layers` {} should be divisible by `num_layer_trainable` {}.".format(
                        num_layers, finetuning_args.num_layer_trainable
                    )
                )

            stride = num_layers // finetuning_args.num_layer_trainable
            trainable_layer_ids = range(stride - 1, num_layers + stride - 1, stride)
        elif finetuning_args.num_layer_trainable > 0:  # fine-tuning the last n layers if num_layer_trainable > 0
            trainable_layer_ids = range(num_layers - finetuning_args.num_layer_trainable, num_layers)
        else:  # fine-tuning the first n layers if num_layer_trainable < 0
            trainable_layer_ids = range(-finetuning_args.num_layer_trainable)

        freeze_modules = {"all"}
        for name, _ in model.named_modules():
            if ".0." in name:
                freeze_modules.add(name.split(".0.")[-1].split(".")[0])

        trainable_layers = []
        for module_name in finetuning_args.name_module_trainable:
            if module_name not in freeze_modules:
                raise ValueError(
                    "Module {} is not found, please choose from {}".format(module_name, ", ".join(freeze_modules))
                )

            for idx in trainable_layer_ids:
                trainable_layers.append(".{:d}.{}".format(idx, module_name if module_name != "all" else ""))

        for name, param in model.named_parameters():
            if any(trainable_layer in name for trainable_layer in trainable_layers):
                if (not finetuning_args.pure_bf16) and (not finetuning_args.use_badam):
                    param.data = param.data.to(torch.float32)
            else:
                param.requires_grad_(False)

        logger.info("Set trainable layers: {}".format(",".join(map(str, trainable_layer_ids))))

    if finetuning_args.finetuning_type == "lora":
        logger.info("Fine-tuning method: {}".format("DoRA" if finetuning_args.use_dora else "LoRA"))
        adapter_to_resume = None

        if model_args.adapter_name_or_path is not None:
            is_mergeable = True
            if getattr(model, "quantization_method", None):  # merge lora in quantized model is unstable
                assert len(model_args.adapter_name_or_path) == 1, "Quantized model only accepts a single adapter."
                is_mergeable = False

            if is_deepspeed_zero3_enabled():
                assert len(model_args.adapter_name_or_path) == 1, "Cannot use multiple adapters in DeepSpeed ZeRO-3."
                is_mergeable = False

            if (is_trainable and not finetuning_args.create_new_adapter) or (not is_mergeable):
                adapter_to_merge = model_args.adapter_name_or_path[:-1]
                adapter_to_resume = model_args.adapter_name_or_path[-1]
            else:
                adapter_to_merge = model_args.adapter_name_or_path

            for adapter in adapter_to_merge:
                model: "LoraModel" = PeftModel.from_pretrained(
                    model, adapter, offload_folder=model_args.offload_folder
                )
                model = model.merge_and_unload()

            if len(adapter_to_merge) > 0:
                logger.info("Merged {} adapter(s).".format(len(adapter_to_merge)))

            if adapter_to_resume is not None:  # resume lora training
                model = PeftModel.from_pretrained(
                    model, adapter_to_resume, is_trainable=is_trainable, offload_folder=model_args.offload_folder
                )

        if is_trainable and adapter_to_resume is None:  # create new lora weights while training
            if len(finetuning_args.lora_target) == 1 and finetuning_args.lora_target[0] == "all":
                target_modules = find_all_linear_modules(model)
            else:
                target_modules = finetuning_args.lora_target

            if finetuning_args.use_llama_pro:
                target_modules = find_expanded_modules(model, target_modules, finetuning_args.num_layer_trainable)

            if (
                finetuning_args.use_dora
                and getattr(model, "quantization_method", None) is not None
                and getattr(model, "quantization_method", None) != QuantizationMethod.BITS_AND_BYTES
            ):
                raise ValueError("DoRA is not compatible with PTQ-quantized models.")

            peft_kwargs = {
                "r": finetuning_args.lora_rank,
                "target_modules": target_modules,
                "lora_alpha": finetuning_args.lora_alpha,
                "lora_dropout": finetuning_args.lora_dropout,
                "use_rslora": finetuning_args.use_rslora,
                "modules_to_save": finetuning_args.additional_target,
            }

            if model_args.use_unsloth:
                from unsloth import FastLanguageModel  # type: ignore

                unsloth_peft_kwargs = {
                    "model": model,
                    "max_seq_length": model_args.model_max_length,
                    "use_gradient_checkpointing": "unsloth",
                }
                model = FastLanguageModel.get_peft_model(**peft_kwargs, **unsloth_peft_kwargs)
            else:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    use_dora=finetuning_args.use_dora,
                    **peft_kwargs,
                )
                model = get_peft_model(model, lora_config)

        if (not finetuning_args.pure_bf16) and (not finetuning_args.use_badam):
            for param in filter(lambda p: p.requires_grad, model.parameters()):
                param.data = param.data.to(torch.float32)

        if model_args.adapter_name_or_path is not None:
            logger.info("Loaded adapter(s): {}".format(",".join(model_args.adapter_name_or_path)))

    return model
