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

"""The definition of NPU fused RoPE kernels.

Init Phase:
1. Define RoPE forward functions.
2. Register NPU fused RoPE kernel.

"""

import sys

import torch

from ......accelerator.helper import DeviceType
from ......utils.logging import get_logger
from ......utils.types import HFModel
from ...base import BaseKernel
from ...registry import register_kernel


logger = get_logger(__name__)

try:
    import torch_npu
except ImportError:
    pass


def _apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    r"""Applies Rotary Position Embedding to the query and key tensors using NPU optimization.

    Args:
        q (Tensor): Query tensor.
        k (Tensor): Key tensor.
        cos (Tensor): Cosine part of embedding.
        sin (Tensor): Sine part of embedding.
        position_ids (Tensor, optional): Position IDs. Default: ``None``.
        unsqueeze_dim (int): Dimension to unsqueeze cos and sin. Default: 1.

    Returns:
        tuple: (q_embed, k_embed) The embedded query and key tensors.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed, k_embed


def _apply_multimodal_rotary_pos_emb_qwen25_vl(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    r"""Applies Rotary Position Embedding with multimodal sections (Qwen2-VL) on NPU.

    Args:
        q (Tensor): Query tensor.
        k (Tensor): Key tensor.
        cos (Tensor): Cosine part of embedding.
        sin (Tensor): Sine part of embedding.
        mrope_section (Tensor): Multimodal RoPE section.
        unsqueeze_dim (int): Dimension to unsqueeze cos and sin. Default: 1.

    Returns:
        tuple: (q_embed, k_embed) The embedded query and key tensors.
    """
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


@register_kernel
class NpuRoPEKernel(BaseKernel):
    r"""NPU Kernel for Rotary Position Embedding."""

    _kernel_id = "npu_fused_rope"
    _device = DeviceType.NPU

    @classmethod
    def apply(cls, **kwargs) -> "HFModel":
        r"""Apply RoPE acceleration by monkey-patching `apply_rotary_pos_emb`.

        This function iterates through the model's modules to find attention layers,
        identifies the module where they are defined, and replaces the original
        `apply_rotary_pos_emb` function in that module's namespace with the
        NPU-accelerated version from this file.

        Args:
            **kwargs: Keyword arguments containing the model.

        Returns:
            HFModel: The model with patched RoPE functions.

        Raises:
            RuntimeError: If dependencies are not met.
            ValueError: If the model is not provided.
        """
        if not cls.check_deps():
            raise RuntimeError(f"torch_npu is not available but {cls.__name__} was called.")
        model = kwargs.get("model", None)
        if model is None:
            raise ValueError(f"HFModel instance is required for {cls.__name__}.")
        _modules = set()
        for module in model.modules():
            if "Attention" in module.__class__.__name__:
                module_name = module.__class__.__module__
                if module_name in _modules:
                    continue
                try:
                    target_module = sys.modules[module_name]
                    if hasattr(target_module, "apply_rotary_pos_emb"):
                        if getattr(target_module, "apply_rotary_pos_emb") is not _apply_rotary_pos_emb:
                            setattr(target_module, "apply_rotary_pos_emb", _apply_rotary_pos_emb)
                            _modules.add(module_name)
                    if hasattr(target_module, "apply_multimodal_rotary_pos_emb"):
                        if (
                            getattr(target_module, "apply_multimodal_rotary_pos_emb")
                            is not _apply_multimodal_rotary_pos_emb_qwen25_vl
                        ):
                            setattr(
                                target_module,
                                "apply_multimodal_rotary_pos_emb",
                                _apply_multimodal_rotary_pos_emb_qwen25_vl,
                            )
                            _modules.add(module_name)
                except Exception as e:
                    logger.warning_rank0_once(f"Failed to apply RoPE kernel to module {module_name}: {e}")
        return model
