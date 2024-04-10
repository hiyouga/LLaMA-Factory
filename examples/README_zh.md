我们提供了多样化的示例脚本。

```
examples/
├── lora_single_gpu/
│   ├── pretrain.sh: 进行预训练
│   ├── sft.sh: 进行指令监督微调
│   ├── reward.sh: 进行奖励模型训练
│   ├── ppo.sh: 进行 PPO 训练
│   ├── dpo.sh: 进行 DPO 训练
│   ├── orpo.sh: 进行 ORPO 训练
│   ├── prepare.sh: 保存预处理后的数据集
│   └── predict.sh: 进行批量预测
├── qlora_single_gpu/
│   ├── bitsandbytes.sh: 微调 4/8 比特 BNB 模型
│   ├── gptq.sh: 微调 4/8 比特 GPTQ 模型
│   ├── awq.sh: 微调 4 比特 AWQ 模型
│   └── aqlm.sh: 微调 2 比特 AQLM 模型
├── lora_multi_gpu/
│   ├── single_node.sh: 使用 Accelerate 进行单节点训练
│   └── multi_node.sh: 使用 Accelerate 进行多节点训练
├── full_multi_gpu/
│   ├── single_node.sh: 使用 DeepSpeed 进行单节点训练
│   └── multi_node.sh: 使用 DeepSpeed 进行多节点训练
├── merge_lora/
│   ├── merge.sh: 将 LoRA 权重合并到预训练模型中
│   └── quantize.sh: 使用 AutoGPTQ 量化模型
├── inference/
│   ├── cli_demo.sh: 启动命令行推理接口
│   ├── api_demo.sh: 启动 OpenAI 风格 API
│   ├── web_demo.sh: 启动浏览器推理接口
│   └── evaluate.sh: 在 MMLU 数据集上评测模型
└── extras/
    ├── galore/
    │   └── sft.sh: 使用 GaLore 训练模型
    ├── loraplus/
    │   └── sft.sh: 使用 LoRA+ 训练模型
    ├── llama_pro/
    │   ├── expand.sh: 扩展模型中的层
    │   └── sft.sh: 训练扩展后的模型
    └── fsdp_qlora/
        └── sft.sh: 使用 FSDP 微调量化模型
```
