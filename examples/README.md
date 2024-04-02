We provide diverse examples about fine-tuning LLMs.

```
examples/
├── lora_single_gpu/
│   ├── pt.sh: Pre-training
│   ├── sft.sh: Supervised fine-tuning
│   ├── reward.sh: Reward modeling
│   ├── ppo.sh: PPO training
│   ├── dpo.sh: DPO training
│   ├── orpo.sh: ORPO training
│   ├── prepare.sh: Save tokenized dataset
│   └── predict.sh: Batch prediction
├── qlora_single_gpu/
│   ├── bitsandbytes.sh
│   ├── gptq.sh
│   ├── awq.sh
│   └── aqlm.sh
├── lora_multi_gpu/
│   ├── single_node.sh
│   └── multi_node.sh
├── full_multi_gpu/
│   ├── single_node.sh
│   └── multi_node.sh
├── merge_lora/
│   ├── merge.sh
│   └── quantize.sh
├── inference/
│   ├── cli_demo.sh
│   ├── api_demo.sh
│   ├── web_demo.sh
│   └── evaluate.sh
└── extras/
    ├── galore/
    │   └── sft.sh
    ├── loraplus/
    │   └── sft.sh
    ├── llama_pro/
    │   ├── expand.sh
    │   └── sft.sh
    └── fsdp_qlora/
        └── sft.sh
```
