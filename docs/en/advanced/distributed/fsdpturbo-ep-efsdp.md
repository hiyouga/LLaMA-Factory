# FSDPTurbo EP/EFSDP and LlamaFactory FSDP2/CP Design

Chinese version: [FSDPTurbo EP/EFSDP 与 LlamaFactory FSDP2/CP 设计说明](../../../zh/advanced/distributed/fsdpturbo-ep-efsdp.md)

This document describes the current implementation of the `fsdpturbo` distributed plugin. Its core principle is a clear separation of responsibilities:

- FSDPTurbo owns expert parallelism (EP), expert parameter sharding (EFSDP), and device operator registration.
- LlamaFactory owns process initialization, the base DeviceMesh, outer FSDP2, CP, model initialization, and weight loading.
- The LlamaFactory integration layer combines the two parameter layouts and handles gradient norms across meshes.

This document corresponds to the following committed revisions:

- LlamaFactory `fsdpturbo_plugin`: `4033d57c`
- FSDPTurbo `fsdpturbo_plugin`: `19f187a`

## 1. Configuration Boundaries

Training uses a single `dist_config` entry:

```yaml
dist_config:
  name: fsdpturbo
  cp_size: 1
  ep_size: 16
  ep_dispatcher: eager
  param_dtype: bf16
  ep_modules:
    - model.language_model.layers.{*}.mlp.experts
  ep_fsdp_modules:
    - model.language_model.layers.{*}.mlp
```

The fields have the following responsibilities:

- `ep_modules`: expert modules parallelized by FSDPTurbo EP.
- `ep_fsdp_modules`: expert containers sharded by FSDPTurbo EFSDP.
- `fsdp_ignored_modules`: modules explicitly excluded from the outer LlamaFactory FSDP2 path.
- `hook_modules` and `fsdp_implementation`: optional hook and implementation settings for FSDPTurbo EFSDP.
- `param_dtype`: parameter dtype used during model initialization for FSDPTurbo full tuning.

`ep_fsdp_modules` determines the EFSDP targets. Non-expert parameters such as attention, embeddings, and the LM head do not enter the FSDPTurbo EFSDP plan. They remain managed by the outer LlamaFactory FSDP2 layer.

Model-specific defaults are managed by the `FSDPTurboEPModelSpec` registry. A registration can provide default `ep_modules`, `ep_fsdp_modules`, and a model preparation function; explicit YAML settings always take precedence. Only a `qwen3_moe` adapter is currently built in. Qwen3.5 MoE uses the explicit module patterns in the example so that model-specific handling is not embedded in the generic distributed entry point.

## 2. Mesh Initialization

LlamaFactory's `DistributedInterface` initializes only its existing model and data meshes. It is unaware of EP and EFSDP and does not expose an extra mesh registration interface for distributed plugins. The FSDPTurbo expert topology is independently created and owned by `FSDPTurboParallelState` in the plugin module:

```text
run_sft / run_dpo / run_rm
  -> DistributedInterface(dist_config)
     -> initialize LlamaFactory model/data meshes
  -> DistributedPlugin("fsdpturbo")
     -> FSDPTurboFSDP2Engine.__init__()
        -> FSDPTurboParallelState.initialize()
           -> initialize and retain the expert parent mesh and submeshes
```

`FSDPTurboParallelState` creates a four-dimensional expert parent mesh:

```text
(edp, efsdp, ep, expert_cp)
```

Its current dimensions are calculated as follows:

```text
dp_size       = world_size / cp_size
ep_fsdp_size  = dp_size / ep_size
edp_size      = dp_size / (ep_size * ep_fsdp_size)
mesh_shape    = (edp_size, ep_fsdp_size, ep_size, cp_size)
```

The state object retains `edp_mesh`, `efsdp_mesh`, `ep_mesh`, and `expert_cp_mesh`. Model sharding and gradient norm logic inside the plugin read expert communication domains from this state, while other LlamaFactory backends do not need to implement or know about these interfaces. Initialization validates that `ep_size` is positive and divides `dp_size`; repeated initialization also rejects topology changes.

## 3. Model Sharding Order

The wrapping order must remain "expert side first, outer FSDP2 second":

```text
DistributedPlugin("fsdpturbo")
  -> FSDPTurboFSDP2Engine.shard_model(model)
     -> prepare_model_ep(model)
        -> expert_parallelize_modules(model, ep_mesh, ep_plan)
        -> expert_fully_shard_modules(model, efsdp_mesh, ep_plan, fsdp_plan)
        -> collect expert parameters as ignored_params
     -> FSDP2Engine.prepare_model(model, ignored_params=...)
        -> apply outer fully_shard to the remaining Transformer Layers and root module
```

This prevents the same expert parameter from being managed by both EFSDP and outer FSDP2. Outer FSDP2 continues to reuse LlamaFactory's model initialization, checkpoint, and save flows.

`ep_dispatcher` supports the FSDPTurbo implementations `eager`, `fused`, `mc2`, and `domino`. The official example and numerical validation currently use `eager`, which selects the portable PyTorch implementation from the registry while tensors remain on the current accelerator. `fused`, `mc2`, and `domino` select the corresponding device fusion or communication-computation overlap implementations and require their operator and dtype constraints to be satisfied by the runtime environment.

## 4. FSDPTurbo Public Entry Point

LlamaFactory depends only on the stable public FSDPTurbo API:

```python
from fsdp_turbo.distributed import (
    EPPlanConfig,
    FSDPPlanConfig,
    expert_fully_shard_modules,
    expert_parallelize_modules,
    module_name_match,
)
```

The import occurs inside `prepare_model_ep()`, so other distributed backends remain importable when FSDPTurbo is not installed. LlamaFactory does not depend on internal file paths under `fsdp_turbo.distributed`.

## 5. Gradient Norms

Outer and expert parameters can belong to different DTensor meshes and therefore cannot be passed together to a single standard `clip_grad_norm_()` call. The `fsdpturbo` plugin groups parameters by their owning mesh and computes local p-power sums:

- Non-expert parameters are reduced over the DP and CP groups.
- Expert parameters are reduced over the EFSDP, EP, and expert-CP groups retained by `FSDPTurboParallelState`.
- After the global norm is assembled, the same clipping coefficient is applied to every local gradient.

A zero-gradient warmup runs during startup so that the required collectives are initialized before training begins. This is currently a backend-specific implementation for `fsdpturbo`; other backends retain their existing gradient norm paths until the upstream distributed plugin interface is decoupled.

## 6. Weight Loading

LlamaFactory retains the `init_on_meta` and safetensors loading flow. The parent `FSDP2Engine` loader dynamically invokes the FSDPTurbo engine override through `self._copy_weights(...)`, so the method is not dead code. It supports DTensors with multiple `Shard` placements by calculating the local slice for the current rank along each mesh dimension in sequence. Model save and checkpoint interfaces continue to reuse the LlamaFactory FSDP2 implementation.

## 7. Kernel Plugin

FLA operators do not belong in the distributed configuration. Operator selection is handled through an independent `kernel_config`:

```yaml
kernel_config:
  name: flash-linear-attention
  include_kernels: chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
  chunk_size: 32
```

The call path is:

```text
ModelEngine
  -> KernelPlugin("flash-linear-attention")
     -> LlamaFactory kernel registry
        -> fsdp_turbo.ops.apply_fla_ops()
           -> FSDPTurbo device operator registry
              -> FLA backend implementation
```

`chunk_size` accepts `16`, `32`, and `64`, with a default of `64`. The kernel plugin and distributed plugin are independent. Explicitly selecting `flash-linear-attention` installs only the listed FLA operators. With `name: auto` and `include_kernels: auto`, LlamaFactory applies every registered default kernel whose dependencies are available before distributed sharding. This allows FLA, RMSNorm, and other non-expert operators to work together with FSDPTurbo EP. FSDPTurbo subsequently replaces the target expert module's `forward`, so the final expert execution path is selected by `ep_dispatcher`; an MoE kernel discovered during the auto stage is not retained as a separate second expert execution path.

## 8. CP Runtime Constraints and Validation Scope

When `init_on_meta` constructs the model, it must propagate `attn_implementation` in the same way as the `from_pretrained` path. Otherwise, the model falls back to a non-FlashAttention implementation and Ulysses CP cannot start. Before calling Hugging Face FlashAttention, Ulysses reconstructs the global attention mask. Only two-dimensional position IDs participate in packed-sequence detection. Multi-axis position IDs such as Qwen3.5 mRoPE have already been consumed by rotary embedding and must not be passed to the FlashAttention packed-sequence detection logic.

The current implementation has completed the following four BF16 AdamW full SFT validations with Qwen3.5-35B-A3B on 16 Ascend A3 dies. Performance is calculated from timestamps of consecutive logged training steps and excludes initialization and compilation before the first step as well as model saving after training:

| CP | EP | EFSDP | Checkpoint | Kernel / Dispatcher | Steps | Loss (first -> last) | Performance | Result |
| ---: | ---: | ---: | --- | --- | ---: | --- | ---: | --- |
| 1 | 16 | 1 | Off | FLA (chunk size 16) / eager | 100 | 1.3345 -> 0.1205 | 1.93 s/it | Passed and saved |
| 1 | 16 | 1 | Off | FLA (chunk size 16) / fused | 100 | 1.3367 -> 0.1326 | 1.95 s/it | Passed and saved |
| 2 | 4 | 2 | On | auto kernels (including FLA) / eager | 20 | 1.8095 -> 1.4910 | 6.74 s/it | Passed and saved |
| 2 | 4 | 2 | Off | auto kernels (including FLA) / eager | 100 | 1.8095 -> 0.6139 | 5.77 s/it | Passed and saved |
