import os
from transformers import is_torch_npu_available
from llmtuner.train.tuner import run_exp


def main():
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    run_exp()


if __name__ == "__main__":
    if is_torch_npu_available():
        use_jit_compile = os.getenv('JIT_COMPILE', 'False').lower() in ['true', '1']
        torch.npu.set_compile_mode(jit_compile=use_jit_compile)
    main()
