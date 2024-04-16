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
│   └── predict.sh: Do batch predict and compute BLEU and ROUGE scores after LoRA tuning
├── qlora_single_gpu/
│   ├── bitsandbytes.sh: Fine-tune 4/8-bit BNB models using QLoRA
│   ├── gptq.sh: Fine-tune 4/8-bit GPTQ models using QLoRA
│   ├── awq.sh: Fine-tune 4-bit AWQ models using QLoRA
│   └── aqlm.sh: Fine-tune 2-bit AQLM models using QLoRA
├── lora_multi_gpu/
│   ├── single_node.sh: Fine-tune model with Accelerate on single node using LoRA
│   └── multi_node.sh: Fine-tune model with Accelerate on multiple nodes using LoRA
├── full_multi_gpu/
│   ├── single_node.sh: Full fine-tune model with DeepSpeed on single node
│   ├── multi_node.sh: Full fine-tune model with DeepSpeed on multiple nodes
│   └── predict.sh: Do batch predict and compute BLEU and ROUGE scores after full tuning
├── merge_lora/
│   ├── merge.sh: Merge LoRA weights into the pre-trained models
│   └── quantize.sh: Quantize the fine-tuned model with AutoGPTQ
├── inference/
│   ├── cli_demo.sh: Launch a command line interface with LoRA adapters
│   ├── api_demo.sh: Launch an OpenAI-style API with LoRA adapters
│   ├── web_demo.sh: Launch a web interface with LoRA adapters
│   └── evaluate.sh: Evaluate model on the MMLU/CMMLU/C-Eval benchmarks with LoRA adapters
└── extras/
    ├── galore/
    │   └── sft.sh: Fine-tune model with GaLore
    ├── badam/
    │   └── sft.sh: Fine-tune model with BAdam
    ├── loraplus/
    │   └── sft.sh: Fine-tune model using LoRA+
    ├── llama_pro/
    │   ├── expand.sh: Expand layers in the model
    │   └── sft.sh: Fine-tune the expanded model
    └── fsdp_qlora/
        └── sft.sh: Fine-tune quantized model with FSDP+QLoRA
```
