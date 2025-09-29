# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's PEFT library.
# https://github.com/huggingface/peft/blob/v0.10.0/src/peft/peft_model.py
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

import gc
import os
import socket
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import torch
import torch.distributed as dist
import transformers.dynamic_module_utils
from huggingface_hub.utils import WeakFileLock
from transformers import AutoConfig, InfNanRemoveLogitsProcessor, LogitsProcessorList
from transformers.dynamic_module_utils import get_relative_imports
from transformers.utils import (
    is_torch_bf16_gpu_available,
    is_torch_cuda_available,
    is_torch_mps_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)
from transformers.utils.versions import require_version

from . import logging


_is_fp16_available = is_torch_npu_available() or is_torch_cuda_available()
try:
    _is_bf16_available = is_torch_bf16_gpu_available() or (is_torch_npu_available() and torch.npu.is_bf16_supported())
except Exception:
    _is_bf16_available = False


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..hparams import ModelArguments


logger = logging.get_logger(__name__)


class AverageMeter:
    r"""Compute and store the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_version(requirement: str, mandatory: bool = False) -> None:
    r"""Optionally check the package version."""
    if is_env_enabled("DISABLE_VERSION_CHECK") and not mandatory:
        logger.warning_rank0_once("Version checking has been disabled, may lead to unexpected behaviors.")
        return

    if "gptmodel" in requirement or "autoawq" in requirement:
        pip_command = f"pip install {requirement} --no-build-isolation"
    else:
        pip_command = f"pip install {requirement}"

    if mandatory:
        hint = f"To fix: run `{pip_command}`."
    else:
        hint = f"To fix: run `{pip_command}` or set `DISABLE_VERSION_CHECK=1` to skip this check."

    require_version(requirement, hint)


def check_dependencies() -> None:
    r"""Check the version of the required packages."""
    check_version("transformers>=4.49.0,<=4.55.0")
    check_version("datasets>=2.16.0,<=3.6.0")
    check_version("accelerate>=1.3.0,<=1.12.0")
    check_version("peft>=0.14.0,<=0.15.2")
    check_version("trl>=0.8.6,<=0.9.6")


def calculate_tps(dataset: list[dict[str, Any]], metrics: dict[str, float], stage: Literal["sft", "rm"]) -> float:
    r"""Calculate effective tokens per second."""
    effective_token_num = 0
    for data in dataset:
        if stage == "sft":
            effective_token_num += len(data["input_ids"])
        elif stage == "rm":
            effective_token_num += len(data["chosen_input_ids"]) + len(data["rejected_input_ids"])

    result = effective_token_num * metrics["epoch"] / metrics["train_runtime"]
    return result / dist.get_world_size() if dist.is_initialized() else result


def count_parameters(model: "torch.nn.Module") -> tuple[int, int]:
    r"""Return the number of trainable parameters and number of all parameters in the model."""
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by itemsize
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "quant_storage") and hasattr(param.quant_storage, "itemsize"):
                num_bytes = param.quant_storage.itemsize
            elif hasattr(param, "element_size"):  # for older pytorch version
                num_bytes = param.element_size()
            else:
                num_bytes = 1

            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def get_current_device() -> "torch.device":
    r"""Get the current available device."""
    if is_torch_xpu_available():
        device = "xpu:{}".format(os.getenv("LOCAL_RANK", "0"))
    elif is_torch_npu_available():
        device = "npu:{}".format(os.getenv("LOCAL_RANK", "0"))
    elif is_torch_mps_available():
        device = "mps:{}".format(os.getenv("LOCAL_RANK", "0"))
    elif is_torch_cuda_available():
        device = "cuda:{}".format(os.getenv("LOCAL_RANK", "0"))
    else:
        device = "cpu"

    return torch.device(device)


def get_world_size() -> int:
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return 1


def _peak_tflops_lookup(device_name: str, dtype: "torch.dtype") -> Optional[float]:
    """Return approximate peak TFLOPs per GPU for known accelerators and dtype.

    Supports A100, H100, H200, B200 for bf16/fp16. Values are ballpark and should
    be overridden via PEAK_TFLOPS_PER_GPU for accurate MFU reporting.
    """
    name = device_name.upper()
    is_bf16 = dtype == torch.bfloat16
    # Approximate theoretical peaks (per-GPU, BF16/FP16)
    peaks = {
        "A100": 312.0 if is_bf16 else 312.0,  # FP16/BF16 similar on A100
        "H100": 989.0 if is_bf16 else 1979.0,  # H100 BF16 vs FP16 TensorCores
        "H200": 1415.0 if is_bf16 else 2829.0,  # indicative
        "B200": 2000.0 if is_bf16 else 4000.0,  # placeholder until confirmed
    }
    for key, val in peaks.items():
        if key in name:
            return val
    return None


def _parse_peak_env() -> Optional[float]:
    val = os.getenv("PEAK_TFLOPS_PER_GPU")
    if not val:
        return None
    try:
        # allow comma-separated list; take first entry for per-GPU reporting
        first = str(val).split(",")[0].strip()
        return float(first)
    except Exception:
        logger.warning_rank0_once(f"Invalid PEAK_TFLOPS_PER_GPU={val!r}; ignoring override.")
        return None


def compute_mfu_from_trainer(trainer: Any, train_runtime: float) -> Optional[dict[str, float]]:
    """Compute per-GPU achieved TFLOPs and MFU using HF Trainer's total_flos.

    Returns a dict with keys: achieved_tflops_per_gpu, mfu_percent.
    Returns None if total_flos is unavailable.
    """
    # total_flos is cumulative FLOPs over the run, as tracked by HF Trainer.
    total_flos = getattr(getattr(trainer, "state", None), "total_flos", None)
    if not isinstance(total_flos, (int, float)) or total_flos <= 0 or train_runtime <= 0:
        return None

    world = get_world_size()
    # Convert FLOPs to TFLOPs/s per GPU. We divide by world size to get per-GPU rate.
    achieved_tflops_per_gpu = (total_flos / train_runtime) / (1e12 * max(world, 1))

    # Determine peak TFLOPs per GPU
    dtype = getattr(getattr(trainer, "model", None), "dtype", None) or torch.bfloat16
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    peak = _parse_peak_env() or _peak_tflops_lookup(device_name, dtype)
    if peak is None or peak <= 0:
        logger.warning_rank0_once(
            "MFU: Unknown peak TFLOPs for device '%s' and dtype '%s'. Set PEAK_TFLOPS_PER_GPU to report MFU.",
            device_name,
            str(dtype),
        )
        return {"achieved_tflops_per_gpu": achieved_tflops_per_gpu}

    mfu = 100.0 * achieved_tflops_per_gpu / peak
    return {"achieved_tflops_per_gpu": achieved_tflops_per_gpu, "mfu_percent": mfu}


def _compute_model_flops_from_cfg(
    cfg: Any,
    total_batch_size: int,
    seq_length: int,
    include_backward: bool = True,
    include_recompute: bool = False,
    include_flashattn: bool = False,
) -> int:
    """Estimate FLOPs per optimizer step using model config.

    Mirrors scripts/stat_utils/cal_mfu.py to keep estimates comparable.
    """
    hidden_size = getattr(cfg, "hidden_size", None)
    vocab_size = getattr(cfg, "vocab_size", None)
    intermediate_size = getattr(cfg, "intermediate_size", None)
    num_attention_heads = getattr(cfg, "num_attention_heads", None)
    num_key_value_heads = getattr(cfg, "num_key_value_heads", None)
    num_hidden_layers = getattr(cfg, "num_hidden_layers", None)
    tie_word_embeddings = getattr(cfg, "tie_word_embeddings", False)

    BASE = 2  # gemm (add + mul)

    # mlp module
    mlp_flops_per_token = 3 * BASE * hidden_size * intermediate_size  # up, gate, down
    mlp_flops = total_batch_size * seq_length * num_hidden_layers * mlp_flops_per_token

    # attn projector module
    q_flops_per_token = BASE * hidden_size * hidden_size
    o_flops_per_token = BASE * hidden_size * hidden_size
    k_flops_per_token = BASE * hidden_size * hidden_size * num_key_value_heads // num_attention_heads
    v_flops_per_token = BASE * hidden_size * hidden_size * num_key_value_heads // num_attention_heads
    attn_proj_flops_per_token = q_flops_per_token + o_flops_per_token + k_flops_per_token + v_flops_per_token
    attn_proj_flops = total_batch_size * seq_length * num_hidden_layers * attn_proj_flops_per_token

    # attn sdpa module (scaled dot-product attention)
    sdpa_flops_per_layer = 2 * BASE * hidden_size * seq_length * seq_length  # (q * k^T) * v
    sdpa_flops = total_batch_size * num_hidden_layers * sdpa_flops_per_layer

    # embedding module
    embedding_flops_per_token = hidden_size * vocab_size
    embedding_flops = total_batch_size * seq_length * embedding_flops_per_token
    if tie_word_embeddings is False:
        embedding_flops *= 2

    non_embedding_flops = mlp_flops + attn_proj_flops + sdpa_flops
    non_embedding_coeff, embedding_coeff = 1, 1
    if include_backward:
        non_embedding_coeff += 2
        embedding_coeff += 2

    if include_recompute:
        non_embedding_coeff += 1

    total_flops = non_embedding_coeff * non_embedding_flops + embedding_coeff * embedding_flops

    if include_flashattn:
        total_flops += sdpa_flops

    return int(total_flops)


def _compute_model_flops(
    model_name_or_path: str,
    total_batch_size: int,
    seq_length: int,
    include_backward: bool = True,
    include_recompute: bool = False,
    include_flashattn: bool = False,
) -> int:
    """Wrapper to load AutoConfig and compute FLOPs."""
    cfg = AutoConfig.from_pretrained(model_name_or_path)
    return _compute_model_flops_from_cfg(
        cfg,
        total_batch_size,
        seq_length,
        include_backward=include_backward,
        include_recompute=include_recompute,
        include_flashattn=include_flashattn,
    )


def compute_mfu_theoretical_from_trainer(
    trainer: Any,
    model_name_or_path: str,
    total_batch_size: int,
    seq_length: int,
    steps_per_second: float,
) -> Optional[dict[str, float]]:
    """Compute theoretical MFU from model config and measured steps/s.

    Per-GPU achieved TFLOPs/s = (steps/s * FLOPs_per_step) / (1e12 * world_size)
    MFU% uses the same peak lookup/override as compute_mfu_from_trainer.
    """
    try:
        if steps_per_second is None or steps_per_second <= 0:
            return None

        world = get_world_size()
        # Prefer existing model.config to avoid network calls; fallback to AutoConfig
        cfg = getattr(getattr(trainer, "model", None), "config", None)
        if cfg is not None:
            flops_per_step = _compute_model_flops_from_cfg(cfg, total_batch_size, seq_length)
        else:
            flops_per_step = _compute_model_flops(model_name_or_path, total_batch_size, seq_length)
        cluster_flops_per_s = steps_per_second * flops_per_step
        achieved_tflops_per_gpu = cluster_flops_per_s / (1e12 * max(world, 1))

        # Determine peak TFLOPs per GPU
        dtype = getattr(getattr(trainer, "model", None), "dtype", None) or torch.bfloat16
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        peak = _parse_peak_env() or _peak_tflops_lookup(device_name, dtype)
        if peak is None or peak <= 0:
            logger.warning_rank0_once(
                "MFU (theoretical): Unknown peak TFLOPs for device '%s' and dtype '%s'. Set PEAK_TFLOPS_PER_GPU.",
                device_name,
                str(dtype),
            )
            return {"achieved_tflops_per_gpu_theoretical": achieved_tflops_per_gpu}

        mfu = 100.0 * achieved_tflops_per_gpu / peak
        return {
            "achieved_tflops_per_gpu_theoretical": achieved_tflops_per_gpu,
            "mfu_percent_theoretical": mfu,
        }
    except Exception:
        return None


def get_device_count() -> int:
    r"""Get the number of available devices."""
    if is_torch_xpu_available():
        return torch.xpu.device_count()
    elif is_torch_npu_available():
        return torch.npu.device_count()
    elif is_torch_mps_available():
        return torch.mps.device_count()
    elif is_torch_cuda_available():
        return torch.cuda.device_count()
    else:
        return 0


def get_logits_processor() -> "LogitsProcessorList":
    r"""Get logits processor that removes NaN and Inf logits."""
    logits_processor = LogitsProcessorList()
    logits_processor.append(InfNanRemoveLogitsProcessor())
    return logits_processor


def get_current_memory() -> tuple[int, int]:
    r"""Get the available and total memory for the current device (in Bytes)."""
    if is_torch_xpu_available():
        return torch.xpu.mem_get_info()
    elif is_torch_npu_available():
        return torch.npu.mem_get_info()
    elif is_torch_mps_available():
        return torch.mps.current_allocated_memory(), torch.mps.recommended_max_memory()
    elif is_torch_cuda_available():
        return torch.cuda.mem_get_info()
    else:
        return 0, -1


def get_peak_memory() -> tuple[int, int]:
    r"""Get the peak memory usage (allocated, reserved) for the current device (in Bytes)."""
    if is_torch_xpu_available():
        return torch.xpu.max_memory_allocated(), torch.xpu.max_memory_reserved()
    elif is_torch_npu_available():
        return torch.npu.max_memory_allocated(), torch.npu.max_memory_reserved()
    elif is_torch_mps_available():
        return torch.mps.current_allocated_memory(), -1
    elif is_torch_cuda_available():
        return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    else:
        return 0, -1


def has_tokenized_data(path: "os.PathLike") -> bool:
    r"""Check if the path has a tokenized dataset."""
    return os.path.isdir(path) and len(os.listdir(path)) > 0


def infer_optim_dtype(model_dtype: Optional["torch.dtype"]) -> "torch.dtype":
    r"""Infer the optimal dtype according to the model_dtype and device compatibility."""
    if _is_bf16_available and (model_dtype == torch.bfloat16 or model_dtype is None):
        return torch.bfloat16
    elif _is_fp16_available:
        return torch.float16
    else:
        return torch.float32


def is_accelerator_available() -> bool:
    r"""Check if the accelerator is available."""
    return (
        is_torch_xpu_available() or is_torch_npu_available() or is_torch_mps_available() or is_torch_cuda_available()
    )


def is_env_enabled(env_var: str, default: str = "0") -> bool:
    r"""Check if the environment variable is enabled."""
    return os.getenv(env_var, default).lower() in ["true", "y", "1"]


def numpify(inputs: Union["NDArray", "torch.Tensor"]) -> "NDArray":
    r"""Cast a torch tensor or a numpy array to a numpy array."""
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.cpu()
        if inputs.dtype == torch.bfloat16:  # numpy does not support bfloat16 until 1.21.4
            inputs = inputs.to(torch.float32)

        inputs = inputs.numpy()

    return inputs


def skip_check_imports() -> None:
    r"""Avoid flash attention import error in custom model files."""
    if not is_env_enabled("FORCE_CHECK_IMPORTS"):
        transformers.dynamic_module_utils.check_imports = get_relative_imports


def torch_gc() -> None:
    r"""Collect the device memory."""
    gc.collect()
    if is_torch_xpu_available():
        torch.xpu.empty_cache()
    elif is_torch_npu_available():
        torch.npu.empty_cache()
    elif is_torch_mps_available():
        torch.mps.empty_cache()
    elif is_torch_cuda_available():
        torch.cuda.empty_cache()


def try_download_model_from_other_hub(model_args: "ModelArguments") -> str:
    if (not use_modelscope() and not use_openmind()) or os.path.exists(model_args.model_name_or_path):
        return model_args.model_name_or_path

    if use_modelscope():
        check_version("modelscope>=1.14.0", mandatory=True)
        from modelscope import snapshot_download  # type: ignore
        from modelscope.hub.api import HubApi  # type: ignore

        if model_args.ms_hub_token:
            api = HubApi()
            api.login(model_args.ms_hub_token)

        revision = "master" if model_args.model_revision == "main" else model_args.model_revision
        with WeakFileLock(os.path.abspath(os.path.expanduser("~/.cache/llamafactory/modelscope.lock"))):
            model_path = snapshot_download(
                model_args.model_name_or_path,
                revision=revision,
                cache_dir=model_args.cache_dir,
            )

        return model_path

    if use_openmind():
        check_version("openmind>=0.8.0", mandatory=True)
        from openmind.utils.hub import snapshot_download  # type: ignore

        with WeakFileLock(os.path.abspath(os.path.expanduser("~/.cache/llamafactory/openmind.lock"))):
            model_path = snapshot_download(
                model_args.model_name_or_path,
                revision=model_args.model_revision,
                cache_dir=model_args.cache_dir,
            )

        return model_path


def use_modelscope() -> bool:
    return is_env_enabled("USE_MODELSCOPE_HUB")


def use_openmind() -> bool:
    return is_env_enabled("USE_OPENMIND_HUB")


def use_ray() -> bool:
    return is_env_enabled("USE_RAY")


def find_available_port() -> int:
    r"""Find an available port on the local machine."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def fix_proxy(ipv6_enabled: bool = False) -> None:
    r"""Fix proxy settings for gradio ui."""
    os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"
    if ipv6_enabled:
        os.environ.pop("http_proxy", None)
        os.environ.pop("HTTP_PROXY", None)
