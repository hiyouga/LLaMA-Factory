# 重构 v1 插件体系，并拆分分布式后端配置与 mesh 拓扑

## 背景

之前的插件实现里有几类问题混在一起：

- `BasePlugin` 以 method dict 的形式注册，函数式插件和多方法插件的形态不统一；
- 不同插件各自处理注册和懒加载：kernel 会扫描 `ops` 目录并靠 import 触发注册，rendering 会在调用时按 template 名动态 import，distributed 又是同一个 name 下分散注册多个 method，整体调用链比较抽象；
- 很多插件直接消费 dict 配置，参数校验比较分散；
- `dist_config` 同时包含后端私有参数和 mesh 拓扑，而且会在 `DistributedInterface`、trainer、model loading 等位置被重复解析，调用链比较绕；
- kernel、batching、rendering、sequence parallel 等插件分布在不同风格的目录和注册方式里。

这会让配置语义、注册时机和调用链都比较难追。尤其是 DeepSpeed 的并行行为来自自己的 config file，并不会消费 v1 accelerator 的 mesh 字段；而 FSDP2 / sequence parallel 才会依赖 mesh。把这些字段都塞进 `dist_config`，容易让参数看起来设置了但实际不生效。

## 主要改动

### 1. 统一插件底座

每个插件族都会继承并新建一个自己的 plugin 基类，例如 `DistributedPlugin`、`BatchingPlugin`、`KernelPlugin`、`RenderingPlugin`。这样做是为了让每个插件族拥有独立的 registry，避免不同插件族之间共享注册表，也让每个族可以只暴露自己需要的调用接口。

`BasePlugin.register()` 现在在一个 plugin name 下只注册一个实现对象：

- 如果注册的是函数，就走 `Plugin(name)(...)` 调用；
- 如果注册的是 staticmethod group class，注册的是类对象本身，不会初始化这个类；调用时走 `Plugin(name).method(...)`，由 `Plugin` 路由壳把方法访问转发到这个类对象上；
- `Plugin(name)` 本身只是一个路由壳，实际调用会解析到 registry 里对应的函数或类。

同时新增了 `parse_params()`，用于把插件配置解析成对应的 dataclass params，例如 `FSDP2Params`、`DeepSpeedParams`、`LoraParams`、`FreezeParams`、`BnbParams`。这样插件内部不再到处写 dict `.get()`，未知字段和缺失字段也能更早报错。

`ensure_methods_implemented()` 是给 staticmethod group class 用的契约检查。因为这些插件实现类只作为方法组使用，不会被实例化，ABC 默认的 abstractmethod 检查不会触发；所以这里改成在类定义阶段检查必要方法是否都实现。

### 2. 收敛插件注册入口

这次把各插件族的注册入口收敛到对应的 `interface.py`，避免多套注册/懒加载方式并存：

- `trainer_plugins/distributed`：删除 `hub.py`，新增 `base.py` 和 `interface.py`；
- `trainer_plugins/batching.py`：改成 `PaddingFreeBatcher` / `DynamicBatcher` 两个 staticmethod group；
- `model_plugins/rendering`：从 `rendering.py` + `templates/` 整理为 `rendering/interface.py` + 具体模板文件；
- `model_plugins/sequence_parallel`：从 `parallelization/` 迁移为独立子包，并拆分 `interface.py`、`loss.py`、`ulysses.py`、`seq_comm.py`；
- `model_plugins/peft.py` 和 `model_plugins/quantization.py`：接入 typed params；
- 训练入口、checkpoint、SFT/RM trainer 同步改为新的插件调用方式。

### 3. 重构 kernel 插件

#### 注册方式

原来的 kernel 会扫描 `ops` 目录，import 每个文件后依赖 decorator 写入额外的 `Registry`。这次删除了独立的 `kernels/registry.py`，改为在 `kernels/interface.py` 中显式注册 kernel：

- `npu_fused_rmsnorm`
- `npu_fused_rope`
- `npu_fused_swiglu`
- `npu_fused_moe`
- `cuda_fused_moe`
- `liger_kernel`
- `auto`

具体 kernel 文件仍然保留在 `kernels/ops/` 下，但注册关系集中在 `interface.py` 里，调用入口更直接。

#### 调用方式

新增 `apply_kernels(model, config)` 作为统一入口。`kernel_config.name` 现在表示要应用的 kernel 名称，并且支持逗号分割：

```yaml
kernel_config:
  name: npu_fused_moe,npu_fused_rmsnorm
```

`auto` 仍然保留，但含义更明确：它会按内置顺序尝试一组默认 NPU kernel，当前包括 moe、rmsnorm、rope、swiglu；如果某个 kernel 因设备或依赖不满足抛出 `RuntimeError`，会跳过并继续尝试下一个。

`liger_kernel` 仍然作为单独 kernel 注册。训练场景下如果配置里没有显式设置 `require_logits`，会默认保留 logits，避免影响需要基于 logits 计算 loss weights 的路径。

#### BaseKernel

`BaseKernel` 改成 template method：

- `apply()` 负责先执行 `check_deps()`；
- 具体 kernel 实现 `_apply()`；
- 设备或依赖不满足时通过异常表达，而不是静默返回 `False`。

### 4. 拆分 mesh 拓扑和分布式后端配置

原来 `dist_config` 同时包含两类信息：一类是后端选择和后端私有参数，另一类是 accelerator 使用的 mesh 拓扑。它还会在 `DistributedInterface`、trainer、model loading 等位置被重复解析，语义比较绕。

这次把 `dp_size`、`cp_size`、`cp_mode`、`mp_replicate_size`、`mp_shard_size`、`dist_timeout` 移到 `TrainingArguments` 顶层，并组装为 `MeshConfig`。`DistributedInterface` 只消费 `MeshConfig` 来初始化 device mesh，不再解析或持有完整 `dist_config`。

`dist_config` 现在只负责分布式后端本身，例如：

```yaml
dist_config:
  name: fsdp2
  dcp_path: null
```

mesh / sequence parallel 参数放在训练参数顶层：

```yaml
cp_size: 2
cp_mode: ulysses
```

这样 DeepSpeed 和 FSDP2 的边界也更清楚：DeepSpeed 的并行行为来自自己的 config file，不消费这套 mesh；FSDP2 / sequence parallel 才走 `MeshConfig`。

### 5. DeepSpeed ZeRO-3 特例处理

DeepSpeed ZeRO-3 的模型加载需要在模型初始化前设置 transformers / accelerate 相关状态。这个逻辑比较特殊，所以没有让 `ModelEngine` 再去读 `dist_config`。

现在是在 `TrainingArguments.__post_init__` 中提前注册 DeepSpeed backend config，然后 `ModelEngine` 只查询 `is_deepspeed_zero3_enabled()`。如果需要 ZeRO-3 初始化，再调用无参的 `setup_deepspeed_zero3_model_loading()`；结束后统一 teardown。

这让 ZeRO-3 判断和普通 distributed mesh 初始化解耦，也避免 `DistributedInterface` 为了模型加载逻辑继续保存完整 `dist_config`。
