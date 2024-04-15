We provide diverse examples about fine-tuning LLMs.

```
examples/
├── lora_single_gpu/
│   ├── pretrain.sh: Do continuous pre-training using LoRA
│   ├── sft.sh: Do supervised fine-tuning using LoRA
│   ├── reward.sh: Do reward modeling using LoRA
│   ├── ppo.sh: Do PPO training using LoRA
│   ├── dpo.sh: Do DPO training using LoRA
│   ├── orpo.sh: Do ORPO training using LoRA
│   ├── prepare.sh: Save tokenized dataset
│   └── predict.sh: Do batch predict
├── qlora_single_gpu/
│   ├── bitsandbytes.sh: Fine-tune 4/8-bit BNB models
│   ├── gptq.sh: Fine-tune 4/8-bit GPTQ models
│   ├── awq.sh: Fine-tune 4-bit AWQ models
│   └── aqlm.sh: Fine-tune 2-bit AQLM models
├── lora_multi_gpu/
│   ├── single_node.sh: Fine-tune model with Accelerate on single node
│   └── multi_node.sh: Fine-tune model with Accelerate on multiple nodes
├── full_multi_gpu/
│   ├── single_node.sh: Fine-tune model with DeepSpeed on single node
│   └── multi_node.sh: Fine-tune model with DeepSpeed on multiple nodes
├── merge_lora/
│   ├── merge.sh: Merge LoRA weights into the pre-trained models
│   └── quantize.sh: Quantize fine-tuned model with AutoGPTQ
├── inference/
│   ├── cli_demo.sh: Launch a command line interface
│   ├── api_demo.sh: Launch an OpenAI-style API
│   ├── web_demo.sh: Launch a web interface
│   └── evaluate.sh: Evaluate model on the MMLU benchmark
└── extras/
    ├── galore/
    │   └── sft.sh: Fine-tune model with GaLore
    ├── badam/
    │   └── sft.sh: Fine-tune model with BAdam
    ├── loraplus/
    │   └── sft.sh: Fine-tune model with LoRA+
    ├── llama_pro/
    │   ├── expand.sh: Expand layers in the model
    │   └── sft.sh: Fine-tune expanded model
    ├── fsdp_qlora/
    │    └── sft.sh: Fine-tune quantized model with FSDP
    └── MoD/
        ├── sft.sh: Fine-tune model with MoD
        └── freeze_sft.sh: Fine-tune model with MoD freezing most of the layers
```
