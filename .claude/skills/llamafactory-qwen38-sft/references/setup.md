# Setup and compatibility

Run commands from the LlamaFactory repository root. Treat the current checkout's `README`, `pyproject.toml`, `requirements/`, examples, and source as authoritative.

## Record the starting state

Before installing or launching anything, record:

- the LlamaFactory commit and selected Python environment;
- accelerator type, driver and CUDA or CANN version, and selected device IDs;
- device utilization, process ownership, and available device memory;
- free space in the model-cache and training-output filesystems.

On Ascend, source the matching toolkit `set_env.sh` in every shell that invokes LlamaFactory. Confirm that the environment, `llamafactory-cli`, `torchrun`, and `accelerate` resolve from the intended installation.

## Install the selected stack

Use an isolated environment and install in this order:

1. Install the hardware-compatible PyTorch stack. On Ascend, use the torch and torch-npu pair pinned by `requirements/npu.txt` and ensure `pyyaml` is installed before importing torch-npu.
2. Install the LlamaFactory checkout with `python -m pip install -e .`.
3. Install only the optional dependencies required by the selected training path.
4. Install `transformers==5.8.0`, the version validated for this workflow and accepted by this checkout.
5. For HyperParallel, install the validated checkout with `python -m pip install --no-deps -e <hyperparallel-checkout>`. Do not select its accelerator extras, because they can replace the torch stack selected in step 1.

For Ascend QLoRA, build the NPU backend from the `multi-backend-refactor` branch of `bitsandbytes-foundation/bitsandbytes` as documented by this repository. Use CMake with `COMPUTE_BACKEND=npu`; a regular CUDA-oriented bitsandbytes wheel is not an acceptance result.

Prefer compiling bitsandbytes into a wheel in an isolated build environment, then install that wheel into the training environment with `--no-deps`. This keeps the required compiler toolchain from replacing the selected Python or PyTorch stack. Record the bitsandbytes commit and wheel checksum. After installation, inspect `libbitsandbytes_npu.so` with `readelf` and `ldd`: every dependency must resolve, and its RPATH or RUNPATH must not point to a disposable build environment. If the compiler injected an absolute build path, produce a runtime-relative wheel and repeat the NPU quantization gate.

Do not disable LlamaFactory's version check. If `pyproject.toml` and `check_dependencies()` declare different bounds, obey the stricter bound. Record declared constraints and installed versions separately from the smoke-test result.

## Dependency gates

Run the gates required by the selected path:

```bash
python -m pip check
llamafactory-cli version
python -c 'import torch, transformers, accelerate, peft; assert transformers.__version__ == "5.8.0"; print(torch.__version__, transformers.__version__, accelerate.__version__, peft.__version__)'
python -c 'import torch_npu; print(torch_npu.__version__)'  # Ascend only
python -c 'import bitsandbytes'                            # QLoRA only
python -c 'import hyper_parallel; from hyper_parallel.integration.llamafactory import HyperParallelArguments, parallelize_model; print(hyper_parallel.__file__, HyperParallelArguments())'  # HyperParallel only
```

For Ascend QLoRA, also run a small NPU tensor-quantization operation; importing bitsandbytes alone does not prove that its NPU backend works. For HyperParallel, verify the printed checkout and both integration symbols. Importing only `HyperParallelArguments` is insufficient.

Investigate dependency conflicts in the selected environment rather than suppressing them. A declared HyperParallel/Transformers metadata mismatch is an installation constraint, not permission to ignore other conflicts or skip the smoke test.

## Model compatibility and cache gates

Before a full download or training launch, confirm that the current checkout maps `Qwen3.8-27B` to `Qwen/Qwen3.8-27B` and template `qwen3_8`. Confirm with Transformers 5.8.0 that:

- `AutoConfig.model_type` is `qwen3_5`;
- the model class is `Qwen3_5ForConditionalGeneration`;
- the tokenizer, processor, and LlamaFactory template load.

Before declaring a local model complete, require:

- every shard named by `model.safetensors.index.json`, validated against the resolved upstream revision;
- `config.json`, tokenizer files, `preprocessor_config.json`, and `video_preprocessor_config.json`;
- no relevant incomplete download files;
- read access for the user that will run training.

Stop at a failed gate, preserve its log, and report the failure without substituting another model, backend, or training method.
