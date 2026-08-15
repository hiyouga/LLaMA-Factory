# 架构概览

v1 以“参数解析 → Engine → Trainer/Sampler → Plugin”为主路径。

## 代码目录

```text
src/llamafactory/v1/
├── launcher.py                 CLI 路由与 torchrun 重启
├── config/                     四个顶层 dataclass
├── accelerator/                设备、进程组和 DeviceMesh
├── core/
│   ├── data_engine.py          数据加载与全局索引
│   ├── model_engine.py         Processor、Renderer、Model
│   ├── rendering/              HF chat template 渲染
│   ├── base_trainer.py         通用训练循环
│   └── base_sampler.py         通用推理入口
├── plugins/
│   ├── data_plugins/
│   ├── model_plugins/
│   ├── trainer_plugins/
│   └── sampler_plugins/
├── trainers/                   SFT、DPO、RM
└── samplers/                   CLI sampler
```

## SFT 训练流程

```text
llamafactory-cli sft config.yaml
  → launcher.launch()
  → get_args()
  → DistributedInterface(training_args)
  → DataEngine(train_dataset)
  → ModelEngine(model_args, is_train=True)
  → SFTTrainer(...).fit()
  → save_model()
```

多设备时 `launcher` 先使用相同子命令和参数通过 `torchrun` 重启。

## DPO/RM 训练流程

DPO 与 RM 复用 DataEngine、ModelEngine 和 BaseTrainer：

- DPO 使用 `DPOSample`，计算 policy/reference 的 chosen/rejected log-prob。
- RM 强制 `model_class=cls`，使用 score head 计算 pairwise ranking loss。

## 可扩展组件

- 新原始数据格式：`DataConverterPlugin`
- 新数据来源：`DataLoaderPlugin`
- 新 PEFT、量化或初始化方法：对应模型插件
- 新分布式后端：`DistributedPlugin`
- 新优化器：`OptimizerPlugin`
- 新模型算子：`KernelPlugin` + `BaseKernel`
- 新训练目标：继承 `BaseTrainer`

Chat template 不再是插件扩展点；由 tokenizer/processor 的 Hugging Face
chat template 或 `custom_chat_template` 提供。
