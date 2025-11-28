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

"""
V1 PEFT (Parameter-Efficient Fine-Tuning) Plugin

Core features extracted from V0: LoRA initialization/loading, adapter merging, LoRA+ optimizer
"""

from typing import Any, Optional, Union

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import PeftModel, TaskType, get_peft_model
from transformers import PreTrainedModel
from transformers.integrations import is_deepspeed_zero3_enabled

from llamafactory.v1.config.model_args import LoraArguments
from llamafactory.v1.extras.types import HFModel


# Try to import bitsandbytes for quantized model support
try:
    import bitsandbytes as bnb

    _BNB_LINEAR_CLASSES = tuple(
        cls for cls in [getattr(bnb.nn, "Linear4bit", None), getattr(bnb.nn, "Linear8bitLt", None)] if cls is not None
    )
except ImportError:
    _BNB_LINEAR_CLASSES = ()


class PeftPlugin:
    """PEFT Plugin: Unified LoRA operation interface"""

    @staticmethod
    def _find_all_linear_modules(model: HFModel, freeze_vision_tower: bool = True) -> list[str]:
        """Automatically detect all linear layers in the model"""
        linear_cls = (torch.nn.Linear, torch.nn.Conv1d) + _BNB_LINEAR_CLASSES

        module_names = set()
        for name, module in model.named_modules():
            if freeze_vision_tower and any(k in name for k in ["vision_tower", "visual", "vision_model"]):
                continue
            if isinstance(module, linear_cls):
                module_names.add(name.split(".")[-1])

        return list(module_names)

    @classmethod
    def apply(
        cls,
        model: HFModel,
        lora_args: LoraArguments,
        adapter_path: Optional[str] = None,
        is_trainable: bool = True,
        freeze_vision_tower: bool = True,
    ) -> HFModel:
        """Apply LoRA to model (initialize new adapter or load existing adapter)

        Args:
            model: Base model
            lora_args: LoRA configuration
            adapter_path: Adapter path, if provided will load existing adapter
            is_trainable: Whether the model is trainable
            freeze_vision_tower: Whether to freeze vision tower

        Returns:
            Model with LoRA applied

        Examples:
            # Initialize new adapter
            model = PeftPlugin.apply(model, lora_args)

            # Load existing adapter
            model = PeftPlugin.apply(model, lora_args, adapter_path="path/to/adapter")
        """
        # Load existing adapter
        if adapter_path is not None:
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=is_trainable)
            return model

        # Create new adapter
        if not is_trainable:
            return model

        # Check compatibility
        quantization_method = getattr(model, "quantization_method", None)
        if quantization_method is not None:
            if lora_args.use_dora and quantization_method != "bnb":
                raise ValueError("DoRA is not compatible with PTQ-quantized models (except BitsAndBytes)")
            if lora_args.pissa_init:
                raise ValueError("Quantized models do not support PiSSA initialization")

        # Determine target modules
        if lora_args.lora_target == "all":
            target_modules = cls._find_all_linear_modules(model, freeze_vision_tower)
        else:
            target_modules = [m.strip() for m in lora_args.lora_target.split(",")]

        # Build PEFT configuration
        peft_kwargs = {
            "r": lora_args.lora_rank,
            "lora_alpha": lora_args.lora_alpha,
            "lora_dropout": lora_args.lora_dropout,
            "target_modules": target_modules,
            "use_rslora": lora_args.use_rslora,
            "use_dora": lora_args.use_dora,
            "modules_to_save": [m.strip() for m in lora_args.additional_target.split(",")]
            if lora_args.additional_target
            else None,
        }

        # Handle PiSSA initialization
        if lora_args.pissa_init:
            if lora_args.pissa_iter == -1:
                peft_kwargs["init_lora_weights"] = "pissa"
            else:
                peft_kwargs["init_lora_weights"] = f"pissa_niter_{lora_args.pissa_iter}"

        peft_config = PeftLoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, **peft_kwargs)
        model = get_peft_model(model, peft_config)

        return model

    @classmethod
    def merge(
        cls,
        model: HFModel,
        adapter_paths: Optional[Union[str, list[str]]] = None,
        unload: bool = True,
    ) -> HFModel:
        """Merge adapter(s) into base model

        Args:
            model: Model (can be base model or PeftModel with loaded adapter)
            adapter_paths: Adapter path(s), can be single path or list of paths. If None, merge loaded adapter
            unload: Whether to unload PEFT wrapper

        Returns:
            Merged model

        Examples:
            # Scenario 1: model is already PeftModel, merge directly
            merged = PeftPlugin.merge(peft_model)

            # Scenario 2: Pass base model and adapter path
            merged = PeftPlugin.merge(base_model, adapter_paths="path/to/adapter")

            # Scenario 3: Merge multiple adapters
            merged = PeftPlugin.merge(base_model, adapter_paths=["adapter1", "adapter2"])
        """
        # Scenario 1: model is already PeftModel, merge directly
        if adapter_paths is None:
            if not isinstance(model, PeftModel):
                raise TypeError(f"Model must be PeftModel or provide adapter_paths parameter")
            return model.merge_and_unload() if unload else model

        # Scenario 2&3: Load and merge adapter(s)
        if isinstance(adapter_paths, str):
            adapter_paths = [adapter_paths]

        if not adapter_paths:
            return model

        # Check constraints
        quantization_method = getattr(model, "quantization_method", None)
        if quantization_method is not None and len(adapter_paths) > 1:
            raise ValueError(f"Quantized models only support single adapter, got {len(adapter_paths)}")
        if is_deepspeed_zero3_enabled() and len(adapter_paths) > 1:
            raise ValueError(f"DeepSpeed ZeRO-3 does not support multiple adapters, got {len(adapter_paths)}")

        # Load and merge sequentially
        for adapter_path in adapter_paths:
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()

        return model

    @classmethod
    def save_merged_model(
        cls,
        model: HFModel,
        output_path: str,
        safe_serialization: bool = True,
        max_shard_size: str = "5GB",
    ) -> None:
        """Save merged model to disk

        Args:
            model: Merged model to save
            output_path: Output directory path
            safe_serialization: Whether to use safetensors format
            max_shard_size: Maximum shard size for sharded checkpoints
        """
        model.save_pretrained(output_path, safe_serialization=safe_serialization, max_shard_size=max_shard_size)