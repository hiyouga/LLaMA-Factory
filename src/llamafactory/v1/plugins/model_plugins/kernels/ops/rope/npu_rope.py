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

"""NPU fused RoPE kernel."""

import sys

import torch

from ......accelerator.helper import DeviceType
from ......utils.logging import get_logger
from ......utils.types import HFModel
from ...base import BaseKernel


logger = get_logger(__name__)

try:
    import torch_npu
except ImportError:
    pass


def _apply_npu_rotary_emb(q, k, cos, sin):
    """Apply NPU rotary embedding with automatic Partial RoPE detection."""
    rotary_dim = cos.shape[-1]
    query_dim = q.shape[-1]

    if rotary_dim < query_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

        q_embed = torch_npu.npu_rotary_mul(q_rot, cos, sin).to(q.dtype)
        k_embed = torch_npu.npu_rotary_mul(k_rot, cos, sin).to(k.dtype)

        q_embed = torch.cat([q_embed, q_pass], dim=-1)
        k_embed = torch.cat([k_embed, k_pass], dim=-1)
    else:
        q_embed = torch_npu.npu_rotary_mul(q, cos, sin).to(q.dtype)
        k_embed = torch_npu.npu_rotary_mul(k, cos, sin).to(k.dtype)

    return q_embed, k_embed


def _apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply Rotary Position Embedding using NPU optimization."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return _apply_npu_rotary_emb(q, k, cos, sin)


def _apply_multimodal_rotary_pos_emb_qwen25_vl(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Apply multimodal RoPE with section support (Qwen2-VL) on NPU."""
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    return _apply_npu_rotary_emb(q, k, cos, sin)


class NpuRoPEKernel(BaseKernel):
    """NPU fused RoPE kernel."""

    _device = DeviceType.NPU

    @classmethod
    def _apply(cls, **kwargs) -> HFModel:
        model = kwargs.get("model")
        if model is None:
            raise ValueError("HFModel instance is required.")

        _modules = set()
        for module in model.modules():
            if "Attention" not in module.__class__.__name__:
                continue
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
