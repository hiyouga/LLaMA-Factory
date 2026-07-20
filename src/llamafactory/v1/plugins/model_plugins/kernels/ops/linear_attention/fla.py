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

"""Flash Linear Attention adapters backed by FSDPTurbo's device operator registry."""

from ......accelerator.helper import DeviceType
from ......utils import logging
from ......utils.types import HFModel
from ...base import BaseKernel
from ...registry import register_kernel


logger = logging.get_logger(__name__)

CHUNK_GATED_DELTA_RULE = "chunk_gated_delta_rule"
FUSED_RECURRENT_GATED_DELTA_RULE = "fused_recurrent_gated_delta_rule"
FLASH_LINEAR_ATTENTION_KERNELS = (
    CHUNK_GATED_DELTA_RULE,
    FUSED_RECURRENT_GATED_DELTA_RULE,
)
SUPPORTED_CHUNK_SIZES = (16, 32, 64)


class _FlashLinearAttentionOpKernel(BaseKernel):
    _device = [DeviceType.CUDA, DeviceType.NPU]
    _op_name = ""

    @classmethod
    def check_deps(cls) -> bool:
        try:
            import fla.ops.gated_delta_rule  # noqa: F401
            from fsdp_turbo.ops import apply_fla_ops  # noqa: F401
        except ImportError:
            return False

        return super().check_deps()

    @classmethod
    def apply(cls, **kwargs) -> HFModel:
        model = kwargs.get("model")
        strict = kwargs.get("strict", False)
        if model is None:
            raise ValueError(f"HFModel instance is required for {cls.__name__}.")
        if not cls.check_deps():
            if strict:
                raise RuntimeError(f"Dependencies for {cls.__name__} are unavailable.")
            return model

        from fsdp_turbo.ops import apply_fla_ops

        configured_kwargs = kwargs.get("op_kwargs")
        op_kwargs = {cls._op_name: configured_kwargs} if configured_kwargs else None
        patched = apply_fla_ops(model, [cls._op_name], op_kwargs=op_kwargs, strict=strict)
        if patched:
            logger.info_rank0(f"Kernel {cls.get_kernel_id()} updated {patched} module callables.")
        return model


@register_kernel
class ChunkGatedDeltaRuleKernel(_FlashLinearAttentionOpKernel):
    _kernel_id = CHUNK_GATED_DELTA_RULE
    _op_name = CHUNK_GATED_DELTA_RULE


@register_kernel
class FusedRecurrentGatedDeltaRuleKernel(_FlashLinearAttentionOpKernel):
    _kernel_id = FUSED_RECURRENT_GATED_DELTA_RULE
    _op_name = FUSED_RECURRENT_GATED_DELTA_RULE
