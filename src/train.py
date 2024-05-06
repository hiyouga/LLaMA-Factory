import os
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from llmtuner.train.tuner import run_exp


def main():
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    run_exp()


if __name__ == "__main__":
    use_jit_compile = os.getenv('JIT_COMPILE', 'False').lower() in ['true', '1']
    torch.npu.set_compile_mode(jit_compile=use_jit_compile)
    main()
