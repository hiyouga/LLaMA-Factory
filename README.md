<div align="center">
<h1>
  360-LLaMA-Factory: A More User-Friendly Post-Training Tool with Sequence Parallelism
</h1>
</div>
<br>
<p align="center">
  We add sequence parallelism into LLaMA-Factory, supporting SFT and DPO with (zigzag) ring attention and other parallelism modes. Please refer to this README and the <a href="https://ai.360.com"> original LLaMA-Factory README</a>.
</p>

<br>

## Installation & Usage

Following the installation guide of LLaMA-Factory, you only need one additional dependency:

`pip install ring-flash-attn`

To train SFT or DPO with sequence parallelism, you only need to note two arguments:

```shell
deepspeed --hostfile=hostfile.4nodes src/train.py \
    ...
    --cutoff_len 100000 \  # the long sequence length you would like to train on
    --sequence_parallel_size 4
```

You could also refer to `360-example.sh`.


## SFT

| **Device**     | **Model**         | **Sequence Parallel Size** | **Max Length** | **Full Parameter SFT** |
|----------------|-------------------|----------------------------|----------------|------------------------|
| **8 × A100**   | **Qwen2.5-7B**   | 4                          | 100k            | ✅                     |
| **8 × A100**   | **Qwen2.5-7B**   | 8                          | 200k            | ✅                     |
| **8 × A100**   | **Qwen2.5-14B**  | 4                          | 100k            | ✅                     |
| **8 × A100**   | **Qwen2.5-14B**  | 8                          | 150k            | ✅                     |
| **32 × A100**  | **Qwen2.5-72B**  | 8                          | 80k            | ✅                     |
| **32 × A100**  | **Qwen2.5-72B**  | 16                         | 100k            | ✅                     |

## DPO

| **Device**     | **Model**         | **Sequence Parallel Size** | **Max Length** | **Full Parameter DPO** |
|----------------|-------------------|----------------------------|----------------|------------------------|
| **8 × A100**   | **Qwen2.5-7B**   | 4                          | 32k            | ✅                     |
| **8 × A100**   | **Qwen2.5-7B**   | 8                          | 84k            | ✅                     |
| **8 × A100**   | **Qwen2.5-14B**  | 4                          | 32k            | ✅                     |
| **8 × A100**   | **Qwen2.5-14B**  | 8                          | 72k            | ✅                     |
| **32 × A100**  | **Qwen2.5-72B**  | 8                          | 46k            | ✅                     |
| **32 × A100**  | **Qwen2.5-72B**  | 16                         | 86k            | ✅                     |

### Results
1. **7B** model supports up to **84k** sequence length training on a single machine with 8 GPUs.
2. **14B** model supports up to **72k** sequence length training on a single machine with 8 GPUs.
3. **72B** model supports up to **84k** sequence length training on a 32-GPU distributed setup.

Our methods demonstrate exceptional long-context handling capabilities, effectively supporting large-scale, long-sequence tasks across various hardware configurations.

## Features
- **Various Methods**: Support advanced algorithms, derived from LLama-factory, support PPO, DPO, KTO, ORPO, etc.
- **Sequence Parallel**: Supports long text sequence parallel and implements DPO sequence parallel based on RingAttention.

## Todo list
- [ ] **Precompute**: Precompute the logits of the reference model to reduce video memory usage


## Citation

If you find our work helpful, please kindly cite as:

```bibtex
@software{360-llama-factory,
  author = {Haosheng Zou, Xiaowei Lv, Shousheng Jia and Xiangzheng Zhang},
  title = {360-LLaMA-Factory},
  url = {https://github.com/Qihoo360/360-LLaMA-Factory},
  year = {2024}
}
```

## Acknowledgement

This repo benefits from [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl), [QLoRA](https://github.com/artidoro/qlora) and [FastChat](https://github.com/lm-sys/FastChat). Thanks for their wonderful works.


## Draft
先只开通用优化：zero3 offload, gradient checkpointing，全参微调。调研其他影响DPO的通用优化。
q25-instruct 7B 14B 单机压测DPO最大长度（sp8）、32k和128k典型长度需要sp多少
72B 4台机器压测

logits.float()和logits.contiguous()注掉

precompute可以测，先不commit


## reproducibility 实验复现

训练loss曲线，开序列并行 vs 不开序列并行
需要都用SequentialSampler

DPO on Qwen2-1.5B-Instruct,
starts both at 0.6931 (or 0.6914 depending on dtype and device) without any dropout
