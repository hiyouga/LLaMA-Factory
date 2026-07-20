# FSDPTurbo EP/EFSDP 与 LlamaFactory FSDP2/CP 设计说明

本文描述 `fsdpturbo` distributed plugin 的当前实现。核心原则是保持两侧职责清晰：

- FSDPTurbo 负责专家并行（EP）、专家参数分片（EFSDP）和设备算子注册。
- LlamaFactory 负责进程初始化、基础 DeviceMesh、外层 FSDP2、CP、模型初始化与权重加载。
- LlamaFactory 的集成层负责把两套参数布局组合起来，并处理跨 Mesh 的梯度范数。

## 1. 配置边界

训练入口使用一个 `dist_config`：

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

字段职责如下：

- `ep_modules`：交给 FSDPTurbo 执行 EP 的专家模块。
- `ep_fsdp_modules`：交给 FSDPTurbo 执行 EFSDP 的专家容器。
- `fsdp_ignored_modules`：需要显式排除在 LlamaFactory 外层 FSDP2 之外的模块。
- `hook_modules`、`fsdp_implementation`：FSDPTurbo EFSDP 的可选 hook 和实现配置。
- `param_dtype`：FSDPTurbo full tuning 在模型初始化阶段使用的参数 dtype。

EFSDP 的目标由 `ep_fsdp_modules` 决定。Attention、Embedding、LM Head 等非专家参数不进入
FSDPTurbo EFSDP plan，而是继续由 LlamaFactory 外层 FSDP2 管理。

## 2. Mesh 初始化

所有通信 Mesh 都由 `DistributedInterface` 在模型构建前初始化，避免插件在模型包装阶段临时创建
ProcessGroup。distributed plugin 只声明拓扑，不直接操作全局通信状态。

```text
run_sft / run_dpo / run_rm
  -> initialize_distributed_interface(dist_config)
     -> DistributedPlugin("fsdpturbo").mesh_specs(strategy)
        -> get_fsdpturbo_mesh_specs(strategy)
     -> DistributedInterface(dist_config, mesh_spec_factory=...)
        -> 初始化 LlamaFactory model/data mesh
        -> _init_extra_meshes(mesh_specs)
```

`get_fsdpturbo_mesh_specs()` 声明专家侧四维父 Mesh：

```text
(edp, efsdp, ep, expert_cp)
```

当前尺寸计算为：

```text
dp_size       = world_size / cp_size
ep_fsdp_size  = dp_size / ep_size
edp_size      = dp_size / (ep_size * ep_fsdp_size)
mesh_shape    = (edp_size, ep_fsdp_size, ep_size, cp_size)
```

`DistributedInterface` 保存父 Mesh 的生命周期，并通过 `Dim.EP` 和 `Dim.EFSDP` 暴露对应子 Mesh。
FSDPTurbo plugin 只通过 `get_device_mesh()` 获取它们，不重复创建通信域。

## 3. 模型切分顺序

模型包装顺序必须保持为“专家侧优先，外层 FSDP2 随后”：

```text
DistributedPlugin("fsdpturbo")
  -> FSDPTurboFSDP2Engine.shard_model(model)
     -> prepare_model_ep(model)
        -> expert_parallelize_modules(model, ep_mesh, ep_plan)
        -> expert_fully_shard_modules(model, efsdp_mesh, ep_plan, fsdp_plan)
        -> 收集专家参数作为 ignored_params
     -> FSDP2Engine.prepare_model(model, ignored_params=...)
        -> 对剩余 Transformer Layer 和根模块执行 outer fully_shard
```

这样可以避免同一专家参数同时被 EFSDP 和外层 FSDP2 管理。外层 FSDP2 仍复用 LlamaFactory
原有的初始化、checkpoint 和保存流程。

## 4. FSDPTurbo 公共入口

LlamaFactory 只依赖 FSDPTurbo 的稳定公共 API：

```python
from fsdp_turbo.distributed import (
    EPPlanConfig,
    FSDPPlanConfig,
    expert_fully_shard_modules,
    expert_parallelize_modules,
    module_name_match,
)
```

导入发生在 `prepare_model_ep()` 内，因此没有安装 FSDPTurbo 时，其他 distributed backend 仍可正常导入。
LlamaFactory 不依赖 `fsdp_turbo.distributed` 下的内部文件路径。

## 5. 梯度范数

外层参数和专家参数可能属于不同 DTensor Mesh，不能直接放入一次标准 `clip_grad_norm_()`。
`fsdpturbo` plugin 按参数所属 Mesh 分组计算局部 p 次方和：

- 非专家参数沿 DP 和 CP group 汇总。
- 专家参数沿 EFSDP、EP 和 CP group 汇总。
- 汇总得到全局范数后，对所有本地梯度应用同一个 clipping coefficient。

启动阶段会执行一次零梯度 warmup，使相关 collective 在正式训练前完成初始化。

## 6. 权重加载

LlamaFactory 保留 `init_on_meta` 和 safetensors 加载流程。FSDPTurbo engine 的 `_copy_weights()`
支持包含多个 `Shard` placement 的 DTensor，按各 Mesh 维度依次计算当前 rank 对应的本地切片。
模型保存和 checkpoint 接口继续复用 LlamaFactory FSDP2 实现。

## 7. Kernel plugin

FLA 算子不属于 distributed config。算子选择通过独立的 `kernel_config` 完成：

```yaml
kernel_config:
  name: flash-linear-attention
  include_kernels: chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
  chunk_size: 32
```

调用链如下：

```text
ModelEngine
  -> KernelPlugin("flash-linear-attention")
     -> LlamaFactory kernel registry
        -> fsdp_turbo.ops.apply_fla_ops()
           -> FSDPTurbo device operator registry
              -> FLA backend implementation
```

`chunk_size` 当前支持 `16`、`32` 和 `64`，默认值为 `64`。Kernel plugin 与 distributed plugin
彼此独立。显式选择 `flash-linear-attention` 时只安装所列 FLA 算子；使用 `name: auto` 和
`include_kernels: auto` 时，FLA 可以与自动发现的 RMSNorm/MoE kernel 一起应用，并与
FSDPTurbo EP 组合使用。

## 8. CP 运行约束与验证范围

`init_on_meta` 构造模型时必须与 `from_pretrained` 路径一样传递 `attn_implementation`，否则模型会退回
非 FlashAttention 实现，Ulysses CP 无法启动。Ulysses 在调用 Hugging Face FlashAttention 前重建全局
attention mask；只有二维 position IDs 才参与 packed-sequence 检测。Qwen3.5 mRoPE 等多轴 position IDs
已经在 rotary embedding 中消费，不应传入 FlashAttention 的 packed-sequence 检测逻辑。

Qwen3.5-35B-A3B 已在 8 die A3 上完成 10 步 full SFT smoke：

```text
world_size=8, cp_size=2, ep_size=4, efsdp_size=1, edp_size=1
FlashAttention2 + FLA, activation checkpointing enabled, cutoff_len=128
```

训练连续产生 loss 和全局 grad norm，完成第 10 步并成功聚合、保存模型。该运行验证了非平凡 CP 与 EP
组合的 forward、backward、混合 Mesh 梯度范数和保存通路。由于 8 die 下 AdamW 双状态缓存超出单卡显存，
smoke 使用无状态 SGD；这不改变模型并行和通信通路，但不构成 AdamW 容量验证。

当前尺寸公式在合法整除时令 `edp_size=1`，因此非平凡 EDP 尚未验证。EFSDP 还要求被切分参数的目标
维度可被 `efsdp_size` 整除；例如 `efsdp_size=3` 不能切分形状为 `[256, 2048]` 的目标维度。后续扩展
EDP 或自动拓扑选择时，应在创建 Mesh 前显式校验这些约束。
