import platform

import accelerate
import datasets
import peft
import torch
import transformers
import trl
from transformers.integrations import is_deepspeed_available
from transformers.utils import is_bitsandbytes_available, is_torch_cuda_available, is_torch_npu_available

from .packages import is_vllm_available


VERSION = "0.7.2.dev0"


def print_env() -> None:
    info = {
        "`llamafactory` version": VERSION,
        "Platform": platform.platform(),
        "Python version": platform.python_version(),
        "PyTorch version": torch.__version__,
        "Transformers version": transformers.__version__,
        "Datasets version": datasets.__version__,
        "Accelerate version": accelerate.__version__,
        "PEFT version": peft.__version__,
        "TRL version": trl.__version__,
    }

    if is_torch_cuda_available():
        info["PyTorch version"] += " (GPU)"
        info["GPU type"] = torch.cuda.get_device_name()

    if is_torch_npu_available():
        info["PyTorch version"] += " (NPU)"
        info["NPU type"] = torch.npu.get_device_name()
        info["CANN version"] = torch.version.cann

    if is_deepspeed_available():
        import deepspeed  # type: ignore

        info["DeepSpeed version"] = deepspeed.__version__

    if is_bitsandbytes_available():
        import bitsandbytes

        info["Bitsandbytes version"] = bitsandbytes.__version__

    if is_vllm_available():
        import vllm

        info["vLLM version"] = vllm.__version__

    print("\n" + "\n".join(["- {}: {}".format(key, value) for key, value in info.items()]) + "\n")
