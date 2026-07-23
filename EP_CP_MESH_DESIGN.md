# FSDPTurbo EP/EFSDP 与 LlamaFactory FSDP2/CP 设计说明

English version: [EP_CP_MESH_DESIGN_EN.md](EP_CP_MESH_DESIGN_EN.md)

本文描述 `fsdpturbo` distributed plugin 的当前实现。核心原则是保持两侧职责清晰：

- FSDPTurbo 负责专家并行（EP）、专家参数分片（EFSDP）和设备算子注册。
- LlamaFactory 负责进程初始化、基础 DeviceMesh、外层 FSDP2、CP、模型初始化与权重加载。
- LlamaFactory 的集成层负责把两套参数布局组合起来，并处理跨 Mesh 的梯度范数。

本文对应以下已提交版本：

- LlamaFactory `fsdpturbo_plugin`：`4033d57c`
- FSDPTurbo `fsdpturbo_plugin`：`19f187a`

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

模型相关的默认适配由 `FSDPTurboEPModelSpec` 注册表管理。注册项可以提供默认 `ep_modules`、
`ep_fsdp_modules` 和模型准备函数；YAML 显式配置始终优先。当前只内置了 `qwen3_moe` 适配，
Qwen3.5 MoE 使用示例中的显式模块模式，避免把单一模型的特殊处理固化到通用分布式入口。

## 2. Mesh 初始化

LlamaFactory 的 `DistributedInterface` 只初始化自身原有的 model/data mesh。它不感知 EP、EFSDP，
也不为 distributed plugin 提供额外 mesh 注册接口。FSDPTurbo 的专家拓扑由插件文件内的
`FSDPTurboParallelState` 独立创建和持有：

```text
run_sft / run_dpo / run_rm
  -> DistributedInterface(dist_config)
     -> 初始化 LlamaFactory model/data mesh
  -> DistributedPlugin("fsdpturbo")
     -> FSDPTurboFSDP2Engine.__init__()
        -> FSDPTurboParallelState.initialize()
           -> 初始化并保存 expert parent mesh 及其子 mesh
```

`FSDPTurboParallelState` 创建专家侧四维父 Mesh：

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

状态对象保存 `edp_mesh`、`efsdp_mesh`、`ep_mesh` 和 `expert_cp_mesh`。插件内部模型切分和梯度范数
都从这个状态对象读取专家通信域；LlamaFactory 其他 backend 不需要实现或感知这些接口。状态初始化
会校验 `ep_size` 为正数且能够整除 `dp_size`，重复初始化时也会拒绝拓扑发生变化。

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

`ep_dispatcher` 支持 FSDPTurbo 提供的 `eager`、`fused`、`mc2` 和 `domino`。当前正式示例和
精度验证使用 `eager`：它选择注册表中的可移植 PyTorch 实现，张量仍在当前加速设备上计算；
`fused`、`mc2` 和 `domino` 会选择相应的设备融合或通信计算重叠实现，并要求运行环境满足对应
算子及 dtype 约束。

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
- 专家参数沿 `FSDPTurboParallelState` 保存的 EFSDP、EP 和 expert-CP group 汇总。
- 汇总得到全局范数后，对所有本地梯度应用同一个 clipping coefficient。

启动阶段会执行一次零梯度 warmup，使相关 collective 在正式训练前完成初始化。
当前这是 `fsdpturbo` backend 的专用实现；其他 backend 继续保留原有梯度范数路径，等待上游
distributed plugin 解耦后再统一公共接口。

## 6. 权重加载

LlamaFactory 保留 `init_on_meta` 和 safetensors 加载流程。父类 `FSDP2Engine` 的加载器通过
`self._copy_weights(...)` 动态调用 FSDPTurbo engine 的覆写实现，因此该方法不是未使用代码。
它支持包含多个 `Shard` placement 的 DTensor，按各 Mesh 维度依次计算当前 rank 对应的本地切片。
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
`include_kernels: auto` 时，LlamaFactory 会在分布式切分前依次应用所有已注册且依赖满足的默认
kernel，因此 FLA、RMSNorm 等非专家算子可以与 FSDPTurbo EP 组合使用。FSDPTurbo 随后会替换
目标专家模块的 `forward`，所以专家计算的最终路径由 `ep_dispatcher` 决定；auto 阶段发现的 MoE
kernel 不会作为独立的第二条专家执行路径保留下来。

## 8. CP 运行约束与验证范围

`init_on_meta` 构造模型时必须与 `from_pretrained` 路径一样传递 `attn_implementation`，否则模型会退回
非 FlashAttention 实现，Ulysses CP 无法启动。Ulysses 在调用 Hugging Face FlashAttention 前重建全局
attention mask；只有二维 position IDs 才参与 packed-sequence 检测。Qwen3.5 mRoPE 等多轴 position IDs
已经在 rotary embedding 中消费，不应传入 FlashAttention 的 packed-sequence 检测逻辑。

截至 2026-07-23，当前实现已在 16 die A3 上用 Qwen3.5-35B-A3B 完成以下四组 BF16、AdamW full SFT
验证。表中性能按相邻训练日志步的时间戳计算，不包含首步前的初始化、编译和训练后的模型保存时间：

| 日志 | CP | EP | EFSDP | Checkpoint | Kernel / Dispatcher | 步数 | Loss（首步 -> 末步） | 性能 | 结果 |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- | ---: | --- |
| `fla_ep16_eager_100_20260723.log` | 1 | 16 | 1 | 关闭 | FLA（chunk size 16）/ eager | 100 | 1.3345 -> 0.1205 | 1.93 s/it | 通过并完成保存 |
| `fla_ep16_fused_100_20260723.log` | 1 | 16 | 1 | 关闭 | FLA（chunk size 16）/ fused | 100 | 1.3367 -> 0.1326 | 1.95 s/it | 通过并完成保存 |
| `fsdpturbo_refactor_cp2_ep4_efsdp2_adamw_20step.log` | 2 | 4 | 2 | 开启 | auto kernels（含 FLA）/ eager | 20 | 1.8095 -> 1.4910 | 6.74 s/it | 通过并完成保存 |
| `fsdpturbo_cp2_ep4_efsdp2_eager_noac_100_20260723.log` | 2 | 4 | 2 | 关闭 | auto kernels（含 FLA）/ eager | 100 | 1.8095 -> 0.6139 | 5.77 s/it | 通过并完成保存 |

四组训练均持续产生有限的 loss 和全局 grad norm，没有出现 NaN、Inf 或运行时错误，并完成模型聚合与保存。
EP16 下 eager 与 fused 的训练步性能接近，说明该短序列配置的主要瓶颈不在 dispatcher。相同 CP/EP/EFSDP
拓扑下，关闭 activation checkpointing 后从 6.74 s/it 降至 5.77 s/it，训练步吞吐约提升 14.4%；100 步
loss 的整体下降也证明非平凡 CP、EP、EFSDP 组合可以稳定执行 forward、backward、AdamW 更新和
mixed-mesh 梯度通信。

这些结果用于功能、数值和阶段性性能验证，不代表严格的跨配置基准：EP16 两组使用 `cutoff_len=256` 和
显式 FLA kernel，CP2/EP4/EFSDP2 两组使用 `cutoff_len=128` 和 auto kernels，因此前后两类数据不应直接
比较。EFSDP 还要求被切分参数的目标维度可被 `efsdp_size` 整除；例如 `efsdp_size=3` 不能切分形状为
`[256, 2048]` 的目标维度，创建 Mesh 前应显式校验该约束。
