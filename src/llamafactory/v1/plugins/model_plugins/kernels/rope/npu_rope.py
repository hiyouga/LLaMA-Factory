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

import sys

import torch

from .....extras.types import HFModel
from ....trainer_plugins.distributed.accelerate import is_torch_npu_available
from ..constants import DeviceType, KernelType
from ..registry import KERNEL_REGISTRY, MetaRoPEKernel


def _apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    import torch_npu

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed, k_embed


def _apply_multimodal_rotary_pos_emb_qwen25_vl(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with multimodal sections (Qwen2-VL)."""
    import torch_npu

    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed, k_embed


class NpuRoPEKernel(MetaRoPEKernel):
    device = DeviceType.NPU
    kernel = _apply_rotary_pos_emb

    @classmethod
    def register_kernel(cls, kernel_type=KernelType.ROPE, device_type=DeviceType.NPU):
        KERNEL_REGISTRY.register(kernel_type, device_type, cls)

    @classmethod
    def apply(cls, model, **kwargs) -> 'HFModel':
        """Apply RoPE acceleration by monkey-patching `apply_rotary_pos_emb`.

        This function iterates through the model's modules to find attention layers,
        identifies the module where they are defined, and replaces the original
        `apply_rotary_pos_emb` function in that module's namespace with the
        NPU-accelerated version from this file.
        """
        if not is_torch_npu_available():
            return model

        _modules = set()
        for module in model.modules():
            if "Attention" in module.__class__.__name__:
                module_name = module.__class__.__module__
                if module_name in _modules:
                    continue
                try:
                    target_module = sys.modules[module_name]
                    if hasattr(target_module, "apply_rotary_pos_emb"):
                        if getattr(target_module, "apply_rotary_pos_emb") is not cls.kernel:
                            setattr(target_module, "apply_rotary_pos_emb", cls.kernel)
                            _modules.add(module_name)
                except Exception:
                    pass
        return model


class NpuQwen2VLRoPEKernel(MetaRoPEKernel):
    device = DeviceType.NPU
    kernel = _apply_multimodal_rotary_pos_emb_qwen25_vl

    @classmethod
    def register_kernel(cls, kernel_type=KernelType.ROPE, device_type=DeviceType.NPU):
        KERNEL_REGISTRY.register(kernel_type, device_type, cls)

    @classmethod
    def apply(cls, model, **kwargs) -> 'HFModel':
        """Apply RoPE acceleration by monkey-patching `apply_rotary_pos_emb`.

        This function iterates through the model's modules to find attention layers,
        identifies the module where they are defined, and replaces the original
        `apply_rotary_pos_emb` function in that module's namespace with the
        NPU-accelerated version from this file.
        """
        _modules = set()
        for module in model.modules():
            if "Attention" in module.__class__.__name__:
                module_name = module.__class__.__module__
                if module_name in _modules:
                    continue
                try:
                    target_module = sys.modules[module_name]
                    if hasattr(target_module, "apply_multimodal_rotary_pos_emb"):
                        if getattr(target_module, "apply_multimodal_rotary_pos_emb") is not cls.kernel:
                            setattr(target_module, "apply_multimodal_rotary_pos_emb", cls.kernel)
                            _modules.add(module_name)
                except Exception:
                    pass
        return model
