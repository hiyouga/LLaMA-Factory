import hashlib
import importlib
import os
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Optional, Union

import transformers
from transformers.dynamic_module_utils import get_relative_import_files
from transformers.utils.hub import HF_MODULES_CACHE

from ...extras import logging
from . import rms_norm, rope, swiglu
from . import sdpa_attention as npu_sdpa_attention


logger = logging.get_logger()

_HF_REMOTE_CODE_LOCK = threading.Lock()


def _patch_sdpa_forward():
    r"""The purpose of this patch is to enable the native SDPA forward function of transformers to adapt to the SDPA interface of NPU.

    If not, calling the SDPA interface is still in the eagle mode.
    """
    from transformers.integrations import sdpa_attention
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface

    sdpa_attention.sdpa_attention_forward = npu_sdpa_attention.sdpa_attention_forward
    AttentionInterface._global_mapping["sdpa"] = npu_sdpa_attention.sdpa_attention_forward
    ALL_ATTENTION_FUNCTIONS["sdpa"] = npu_sdpa_attention.sdpa_attention_forward


def _patch_rmsnorm(module: ModuleType, class_name: str):
    setattr(module, class_name, rms_norm.NpuRMSNorm)


def _patch_rope(module: ModuleType, func_name: str):
    setattr(module, func_name, rope.apply_rotary_pos_emb)


def _patch_swiglu(module: ModuleType, class_name: str):
    setattr(getattr(module, class_name), "forward", swiglu.npu_swiglu_forward)


def _original_get_dynamic_module(
        class_name: str,
        module_path: Union[str, os.PathLike],
        *,
        force_reload: bool = False,
):
    """Get dynamic module from py file, copied from transformers.dynamic_module_utils.get_class_in_module."""
    name = os.path.normpath(module_path)
    if name.endswith(".py"):
        name = name[:-3]
    name = name.replace(os.path.sep, ".")
    module_file: Path = Path(HF_MODULES_CACHE) / module_path
    with _HF_REMOTE_CODE_LOCK:
        if force_reload:
            sys.modules.pop(name, None)
            importlib.invalidate_caches()
        cached_module: Optional[ModuleType] = sys.modules.get(name)
        module_spec = importlib.util.spec_from_file_location(name, location=module_file)

        # Hash the module file and all its relative imports to check if we need to reload it
        module_files: list[Path] = [module_file] + sorted(map(Path, get_relative_import_files(module_file)))
        module_hash: str = hashlib.sha256(b"".join(bytes(f) + f.read_bytes() for f in module_files)).hexdigest()

        module: ModuleType
        if cached_module is None:
            module = importlib.util.module_from_spec(module_spec)
            # insert it into sys.modules before any loading begins
            sys.modules[name] = module
        else:
            module = cached_module
        if getattr(module, "__transformers_module_hash__", "") != module_hash:
            module_spec.loader.exec_module(module)
            module.__transformers_module_hash__ = module_hash
    return module


def _dynamic_patch_flash_attention(sdpa_attention_cls: str, module: ModuleType, forward,  **kwargs):
    _patch_sdpa_forward()
    setattr(getattr(module, sdpa_attention_cls), "forward", forward)


def _dynamic_patch_rmsnorm(rmsnorm_cls: str, module: ModuleType, **kwargs):
    setattr(module, rmsnorm_cls, rms_norm.NpuRMSNorm)


def _dynamic_patch_rope(rope_cls: str, module: ModuleType, **kwargs):
    setattr(module, rope_cls, rope.apply_rotary_pos_emb)


def _dynamic_patch_swiglu(swiglu_cls: str, npu_swiglu_forward, module: ModuleType, **kwargs):
    setattr(getattr(module, swiglu_cls), "forward", npu_swiglu_forward)


def _patch_dynamic_fused_ops():
    def _get_dynamic_module(
            class_name: str,
            module_path: Union[str, os.PathLike],
            *,
            force_reload: bool = False,
    ):
        module = _original_get_dynamic_module(class_name, module_path, force_reload=force_reload)
        if module.__name__.endswith("modeling_internlm3"):
            _dynamic_patch_flash_attention("InternLM3SdpaAttention", module, npu_sdpa_attention.internlm3_sdpa_forward)
            _dynamic_patch_rmsnorm("InternLM3RMSNorm", module)
            _dynamic_patch_rope("apply_rotary_pos_emb", module)
            _dynamic_patch_swiglu("InternLM3MLP", swiglu.npu_swiglu_forward, module)
        if module.__name__.endswith("modeling_internlm2"):
            _dynamic_patch_flash_attention("InternLM2SdpaAttention", module, npu_sdpa_attention.internlm2_sdpa_forward)
            _dynamic_patch_rmsnorm("InternLM2RMSNorm", module)
            _dynamic_patch_rope("apply_rotary_pos_emb", module)
            _dynamic_patch_swiglu("InternLM2MLP", swiglu.npu_internlm2_swiglu_forward, module)
        return module

    def _get_class_in_module(
            class_name: str,
            module_path: Union[str, os.PathLike],
            *,
            force_reload: bool = False,
    ):
        module = _get_dynamic_module(class_name=class_name, module_path=module_path, force_reload=force_reload)
        return getattr(module, class_name)

    transformers.dynamic_module_utils.get_class_in_module = _get_class_in_module


def apply_fused_ops(config):
    from transformers.models.qwen2 import modeling_qwen2
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
    from transformers.models.qwen2_moe import modeling_qwen2_moe
    from transformers.models.qwen3 import modeling_qwen3
    from transformers.models.qwen3_moe import modeling_qwen3_moe

    _patch_dynamic_fused_ops()
    if "Qwen2ForCausalLM" in getattr(config, "architectures", []):
        _patch_sdpa_forward()
        _patch_rmsnorm(modeling_qwen2, "Qwen2RMSNorm")
        _patch_rope(modeling_qwen2, "apply_rotary_pos_emb")
        _patch_swiglu(modeling_qwen2, "Qwen2MLP")

    if "Qwen2MoeForCausalLM" in getattr(config, "architectures", []):
        _patch_sdpa_forward()
        _patch_rmsnorm(modeling_qwen2_moe, "Qwen2MoeRMSNorm")
        _patch_rope(modeling_qwen2_moe, "apply_rotary_pos_emb")
        _patch_swiglu(modeling_qwen2_moe, "Qwen2MoeMLP")

    if "Qwen3ForCausalLM" in getattr(config, "architectures", []):
        _patch_sdpa_forward()
        _patch_rmsnorm(modeling_qwen3, "Qwen3RMSNorm")
        _patch_rope(modeling_qwen3, "apply_rotary_pos_emb")
        _patch_swiglu(modeling_qwen3, "Qwen3MLP")

    if "Qwen3MoeForCausalLM" in getattr(config, "architectures", []):
        _patch_sdpa_forward()
        _patch_rmsnorm(modeling_qwen3_moe, "Qwen3MoeRMSNorm")
        _patch_rope(modeling_qwen3_moe, "apply_rotary_pos_emb")
        _patch_swiglu(modeling_qwen3_moe, "Qwen3MoeMLP")

    if "Qwen2_5_VLForConditionalGeneration" in getattr(config, "architectures", []):
        _patch_sdpa_forward()
        _patch_rmsnorm(modeling_qwen2_5_vl, "Qwen2RMSNorm")
        _patch_swiglu(modeling_qwen2_5_vl, "Qwen2MLP")
        _patch_swiglu(modeling_qwen2_5_vl, "Qwen2_5_VLMLP")
        setattr(modeling_qwen2_5_vl, "apply_multimodal_rotary_pos_emb", rope.apply_multimodal_rotary_pos_emb_qwen25_vl)
