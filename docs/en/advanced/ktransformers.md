# KTransformers LoRA SFT

KTransformers (KT) executes routed MoE experts on CPU while LLaMA-Factory remains responsible for data,
LoRA arguments, and the training entry point. The production scope is routed-BF16 and routed-INT8 LoRA.

KT has one user configuration source: the training YAML. Accelerate YAML contains FSDP2 settings only.
LLaMA-Factory derives LoRA rank, alpha, dropout, activation policy, and local runtime capacity.

```yaml
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_target: all

use_kt: true
disable_gradient_checkpointing: false
kt_cpu_activation: retain
kt_config:
  kt_expert_weight_format: bf16
  kt_backend: AMXBF16
  kt_num_threads: 96
  kt_tp_enabled: true
  kt_threadpool_count: 2
  kt_max_cache_depth: 2
```

Routed INT8 additionally requires matching expert and BF16 non-expert artifacts:

```yaml
kt_weight_path: /abs/path/to/routed-int8-experts
kt_non_expert_weight_path: /abs/path/to/bf16-non-expert-cache
kt_config:
  kt_expert_weight_format: int8
  kt_backend: auto
  kt_weight_lifecycle: persistent
```

Launch the standard training entry point through Accelerate:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file examples/ktransformers/accelerate/fsdp2_kt_bf16.yaml \
  src/train.py examples/ktransformers/train_lora/qwen3_5moe_lora_sft_kt.yaml
```

The saved adapter contains both standard PEFT and fused expert LoRA artifacts. `adapter_name_or_path` loads both
in a fresh process; `resume_from_checkpoint` restores training after reconstructing the same optimizer groups.
Distributed optimizer resume currently requires the same world size. Missing, tampered, or mismatched artifacts
fail closed instead of falling back to the source checkpoint.

Do not combine KT with a second Transformers/FSDP checkpoint wrapper or Unsloth GC, and do not put `kt_config`
in the Accelerate YAML. See the BF16 and INT8 examples under `examples/ktransformers/train_lora/`.
