import gc
import os
import torch
from typing import TYPE_CHECKING, Tuple
from transformers import InfNanRemoveLogitsProcessor, LogitsProcessorList

try:
    from transformers.utils import (
        is_torch_bf16_cpu_available,
        is_torch_bf16_gpu_available,
        is_torch_cuda_available,
        is_torch_npu_available
    )
    _is_fp16_available = is_torch_npu_available() or is_torch_cuda_available()
    _is_bf16_available = is_torch_bf16_gpu_available() or is_torch_bf16_cpu_available()
except ImportError:
    _is_fp16_available = torch.cuda.is_available()
    try:
        _is_bf16_available = torch.cuda.is_bf16_supported()
    except:
        _is_bf16_available = False

if TYPE_CHECKING:
    from transformers import HfArgumentParser
    from llmtuner.hparams import ModelArguments


class AverageMeter:
    r"""
    Computes and stores the average and current value.
    """
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


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by 2
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def get_current_device() -> torch.device:
    import accelerate
    if accelerate.utils.is_xpu_available():
        device = "xpu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif accelerate.utils.is_npu_available():
        device = "npu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif torch.cuda.is_available():
        device = "cuda:{}".format(os.environ.get("LOCAL_RANK", "0"))
    else:
        device = "cpu"

    return torch.device(device)


def get_logits_processor() -> "LogitsProcessorList":
    r"""
    Gets logits processor that removes NaN and Inf logits.
    """
    logits_processor = LogitsProcessorList()
    logits_processor.append(InfNanRemoveLogitsProcessor())
    return logits_processor


def infer_optim_dtype(model_dtype: torch.dtype) -> torch.dtype:
    r"""
    Infers the optimal dtype according to the model_dtype and device compatibility.
    """
    if _is_bf16_available and model_dtype == torch.bfloat16:
        return torch.bfloat16
    elif _is_fp16_available:
        return torch.float16
    else:
        return torch.float32


def torch_gc() -> None:
    r"""
    Collects GPU memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def try_download_model_from_ms(model_args: "ModelArguments") -> None:
    if not use_modelscope() or os.path.exists(model_args.model_name_or_path):
        return

    try:
        from modelscope import snapshot_download # type: ignore
        revision = "master" if model_args.model_revision == "main" else model_args.model_revision
        model_args.model_name_or_path = snapshot_download(
            model_args.model_name_or_path,
            revision=revision,
            cache_dir=model_args.cache_dir
        )
    except ImportError:
        raise ImportError("Please install modelscope via `pip install modelscope -U`")


def use_modelscope() -> bool:
    return bool(int(os.environ.get("USE_MODELSCOPE_HUB", "0")))
