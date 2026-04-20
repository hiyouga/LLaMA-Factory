# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Literal, TypedDict, Union

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from ...config import InputArgument, get_args
from ...core.model_engine import ModelEngine
from ...utils import logging
from ...utils.plugin import BasePlugin
from ...utils.types import HFModel


logger = logging.get_logger(__name__)


class LoraConfigDict(TypedDict, total=False):
    name: Literal["lora"]
    """Plugin name."""
    r: int
    """Lora rank."""
    lora_alpha: int
    """Lora alpha."""
    lora_dropout: float
    """Lora dropout."""
    target_modules: Union[list[str], str]
    """Target modules."""
    use_rslora: bool
    """Use RS-LoRA."""
    use_dora: bool
    """Use DoRA."""
    modules_to_save: list[str]
    """Modules to save."""
    adapter_name_or_path: Union[list[str], str]
    """Path to the adapter(s)."""
    export_dir: str
    """Path to the export directory."""
    export_size: int
    """Shard size for the export model."""
    export_hub_model_id: str
    """Hub model ID for the export model."""
    infer_dtype: Literal["auto", "float16", "float32", "bfloat16"]
    """Inference data type for the export model."""
    export_legacy_format: bool
    """Use legacy format for the export model."""


class FreezeConfigDict(TypedDict, total=False):
    name: Literal["freeze"]
    """Plugin name."""
    freeze_trainable_layers: int
    """Freeze trainable layers."""
    freeze_trainable_modules: Union[list[str], str]
    """Freeze trainable modules."""
    freeze_extra_modules: list[str]
    """Freeze extra modules."""
    cast_trainable_params_to_fp32: bool
    """Cast trainable params to fp32."""


class PeftPlugin(BasePlugin):
    def __call__(self, model: HFModel, config: dict, is_train: bool) -> HFModel:
        return super().__call__(model, config, is_train)


def _find_all_linear_modules(model: HFModel) -> list[str]:
    r"""Find all available modules to apply LoRA."""
    forbidden_modules = {"lm_head", "output_layer", "output"}
    module_names = set()
    for name, module in model.named_modules():
        if any(forbidden_module in name for forbidden_module in forbidden_modules):
            continue

        if "Linear" in module.__class__.__name__ and "Embedding" not in module.__class__.__name__:
            module_names.add(name.split(".")[-1])

    return list(module_names)


def merge_adapters(model: HFModel, adapter_name_or_path: Union[list[str], str]) -> HFModel:
    if not isinstance(adapter_name_or_path, list):
        adapter_name_or_path = [adapter_name_or_path]

    for adapter_path in adapter_name_or_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        logger.info_rank0(f"Merged adapter from {adapter_path}")

    return model


def load_adapter(model: HFModel, adapter_name_or_path: Union[list[str], str], is_train: bool) -> HFModel:
    r"""Loads adapter(s) into the model.

    Determine adapter usage based on mode:
    - Training: Load the single adapter for continued training.
    - Inference: Merge all adapters to clean up the model.
    - Unmergeable: Keep the single adapter active without merging.
    """
    if not isinstance(adapter_name_or_path, list):
        adapter_name_or_path = [adapter_name_or_path]

    # TODO
    # Adapters fix for deepspeed and quant
    # Adapters fix for vision

    if is_train and len(adapter_name_or_path) > 1:
        raise ValueError(
            "When `adapter_name_or_path` is provided for training, only a single LoRA adapter is supported. "
            "Training will continue on the specified adapter. "
            "Please merge multiple adapters before starting a new LoRA adapter."
        )

    if is_train:
        adapter_to_merge = []
        adapter_to_resume = adapter_name_or_path[0]
    else:
        adapter_to_merge = adapter_name_or_path
        adapter_to_resume = None

    if adapter_to_merge:
        model = merge_adapters(model, adapter_to_merge)

    if adapter_to_resume is not None:
        model = PeftModel.from_pretrained(model, adapter_to_resume, is_trainable=is_train)
        if is_train:
            logger.info_rank0(
                f"Resuming training from existing LoRA adapter at {adapter_to_resume}. "
                "LoRA hyperparameters will be loaded from the adapter itself; "
                "the current LoRA configuration will be ignored. "
                "Merge the adapter into the base model before training if you want to start a new adapter."
            )

    return model


@PeftPlugin("lora").register()
def get_lora_model(model: HFModel, config: LoraConfigDict, is_train: bool = False) -> HFModel:
    adapter_name_or_path = config.get("adapter_name_or_path")

    if adapter_name_or_path:
        return load_adapter(model, adapter_name_or_path, is_train)

    logger.info_rank0("Fine-tuning method: LoRA")

    target_modules = config.get("target_modules", "all")

    # Handle target modules
    if target_modules == "all":
        target_modules = _find_all_linear_modules(model)
    elif isinstance(target_modules, str):
        target_modules = [target_modules]

    logger.info_rank0(f"LoRA target modules: {target_modules}")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=not is_train,
        r=config.get("r", 8),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.05),
        use_rslora=config.get("use_rslora", False),
        use_dora=config.get("use_dora", False),
        target_modules=target_modules,
        modules_to_save=config.get("modules_to_save", None),
    )

    model = get_peft_model(model, peft_config)

    if is_train:
        model.print_trainable_parameters()

    return model


@PeftPlugin("freeze").register()
def get_freeze_model(model: HFModel, config: FreezeConfigDict, is_train: bool = False) -> HFModel:
    logger.info_rank0("Fine-tuning method: Freeze")

    if not is_train:
        return model

    freeze_trainable_layers = config.get("freeze_trainable_layers", 2)
    freeze_trainable_modules = config.get("freeze_trainable_modules", ["all"])
    freeze_extra_modules = config.get("freeze_extra_modules", [])
    cast_trainable_params_to_fp32 = config.get("cast_trainable_params_to_fp32", True)

    if isinstance(freeze_trainable_modules, str):
        freeze_trainable_modules = [module.strip() for module in freeze_trainable_modules.split(",")]

    if isinstance(freeze_extra_modules, str):
        freeze_extra_modules = [module.strip() for module in freeze_extra_modules.split(",")]

    # Get number of layers
    num_layers = (
        getattr(model.config, "num_hidden_layers", None)
        or getattr(model.config, "num_layers", None)
        or getattr(model.config, "n_layer", None)
    )

    if not num_layers:
        raise ValueError("Current model does not support freeze tuning.")

    if freeze_trainable_layers > 0:
        # last n layers
        trainable_layer_ids = range(max(0, num_layers - freeze_trainable_layers), num_layers)
    else:
        # first n layers
        trainable_layer_ids = range(min(-freeze_trainable_layers, num_layers))

    # Identify hidden and non-hidden modules
    hidden_modules = set()
    non_hidden_modules = set()
    for name, _ in model.named_parameters():
        if ".0." in name:
            hidden_modules.add(name.split(".0.")[-1].split(".")[0])
        elif ".1." in name:
            hidden_modules.add(name.split(".1.")[-1].split(".")[0])

        if re.search(r"\.\d+\.", name) is None:
            non_hidden_modules.add(name.split(".")[-2])

    # Build list of trainable layer patterns
    trainable_layers = []
    for module_name in freeze_trainable_modules:
        if module_name == "all":
            for idx in trainable_layer_ids:
                trainable_layers.append(f".{idx:d}.")
        elif module_name in hidden_modules:
            for idx in trainable_layer_ids:
                trainable_layers.append(f".{idx:d}.{module_name}")
        else:
            raise ValueError(f"Module {module_name} not found in hidden modules: {hidden_modules}")

    # Add extra modules
    if freeze_extra_modules:
        for module_name in freeze_extra_modules:
            if module_name in non_hidden_modules:
                trainable_layers.append(module_name)
            else:
                raise ValueError(f"Module {module_name} not found in non-hidden modules: {non_hidden_modules}")

    # TODO
    # Multi-modal special handling

    # Set requires_grad
    forbidden_modules = {"quant_state", "quantization_weight", "qweight", "qzeros", "scales"}
    for name, param in model.named_parameters():
        if any(trainable_layer in name for trainable_layer in trainable_layers) and not any(
            forbidden_module in name for forbidden_module in forbidden_modules
        ):
            param.requires_grad_(True)
            if cast_trainable_params_to_fp32:
                param.data = param.data.to(torch.float32)  # Cast to fp32 for stability
        else:
            param.requires_grad_(False)

    logger.info_rank0(f"Set trainable layers: {trainable_layers}")

    # Count trainable params for verification
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info_rank0(
        f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.4f}"
    )

    return model


def merge_and_export_model(args: InputArgument = None):
    model_args, _, _, _ = get_args(args)

    export_config = model_args.peft_config
    if export_config is None:
        raise ValueError("Please specify peft_config to merge and export model.")

    export_dir = export_config.get("export_dir")
    if export_dir is None:
        raise ValueError("Please specify export_dir.")

    export_size = export_config.get("export_size", 5)
    export_hub_model_id = export_config.get("export_hub_model_id")
    infer_dtype = export_config.get("infer_dtype", "auto")
    export_legacy_format = export_config.get("export_legacy_format", False)

    adapters = None
    if export_config.get("name") == "lora":
        adapters = export_config.get("adapter_name_or_path")
    else:
        raise ValueError("Currently merge and export model function is only supported for lora.")

    if adapters is None:
        raise ValueError("Please set adapter_name_or_path to merge adapters into base model.")

    logger.info_rank0("Loading model for export...")
    model_engine = ModelEngine(model_args, is_train=False)
    model = model_engine.model
    tokenizer = model_engine.processor

    if infer_dtype == "auto":
        if model.config.torch_dtype == torch.float32 and torch.cuda.is_bf16_supported():
            model = model.to(torch.bfloat16)
            logger.info_rank0("Converted model to bfloat16.")
    else:
        target_dtype = getattr(torch, infer_dtype)
        model = model.to(target_dtype)
        logger.info_rank0(f"Converted model to {infer_dtype}.")

    logger.info_rank0(f"Exporting model to {export_dir}...")
    model.save_pretrained(
        export_dir,
        max_shard_size=f"{export_size}GB",
        safe_serialization=not export_legacy_format,
    )
    if tokenizer is not None:
        try:
            if hasattr(tokenizer, "padding_side"):
                tokenizer.padding_side = "left"
            tokenizer.save_pretrained(export_dir)
        except Exception as e:
            logger.warning(f"Failed to save tokenizer: {e}")

    if export_hub_model_id:
        logger.info_rank0(f"Pushing to hub: {export_hub_model_id}...")
        model.push_to_hub(export_hub_model_id)
        if tokenizer is not None:
            tokenizer.push_to_hub(export_hub_model_id)

    logger.info_rank0("Model exported successfully.")
