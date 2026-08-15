# 模型插件

模型插件由 ModelEngine 在加载过程中调用。

## InitPlugin

注册 `init_on_default`、`init_on_meta`、`init_on_rank0`，返回模型创建使用
的 `torch.device`。

## PeftPlugin

注册 `lora` 与 `freeze`。两者分别通过 `LoraParams` 和 `FreezeParams`
严格解析配置。LoRA 还负责 adapter 加载、合并与导出。

## QuantizationPlugin

注册 `auto` 与 `bnb`。插件修改 `from_pretrained` 的 init kwargs，不直接
替换已经加载的权重。

## KernelPlugin

Kernel 在模型加载和 PEFT 处理后应用。调用流程见
[融合算子加速](kernel-acceleration/overview.md)。

## Sequence Parallel Plugins

`SequenceParallelModelPlugin("ulysses")` 修改模型 forward 需要的通信；
`SequenceParallelLossPlugin("sequence_parallel_loss")` 处理 loss 聚合。

## Chat Template 迁移

旧 `RenderingPlugin` 和 `plugins/model_plugins/templates/` 已删除。Chat
template 统一由 `core/rendering/` 调用 Hugging Face 模板。
