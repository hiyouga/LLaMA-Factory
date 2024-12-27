<div align="center">
<h1>
  360-LLaMA-Factory: More User-Friendly Post-Training with Sequence Parallelism
</h1>
</div>
<p align="center">
  We added Sequence Parallelism (SP) into LLaMA-Factory, with plug-and-play usage requiring only one extra dependency and one extra argument.
  We believe LLaMA-Factory with SP is more user-friendly than the few existing SP frameworks due to the wider popularity and fuller functionality of LLaMA-Factory.
  
  We support most post-training methods such as SFT and DPO-variants (DPO, NCA, ORPO, etc.) with <a href="https://github.com/zhuzilin/ring-flash-attention"> (zigzag) ring attention</a> SP.
  Under SP of 8 GPUs, with zero3-offload and gradient checkpointing **but no other optimization**, **full-parameter SFT** could reach lengths of 210k(7B) and 120k(72B), while **full-parameter DPO** 84k(7B) and 46k(72B).
  
  Please refer to this README and the <a href="https://github.com/hiyouga/LLaMA-Factory/blob/main/README.md"> original LLaMA-Factory README</a>.
</p>

<br>

## Introduction

360-LLaMA-Factory is extremely easy to install and use. We build upon LLaMA-Factory and ring-flash-attention and their contributions are well acknowledged. We also plan to incorporate DeepSpeed Ulysses and llama3-style SP as alternative sequence parallelism modes.

Installation & Usage is almost the same as the original LLaMA-Factory.
Compared to the few existing SP frameworks, 360-LLaMA-Factory has fewer potential bugs and better and fuller functionality thanks to LLaMA-Factory.

### Installation & Usage

Following the installation guide of LLaMA-Factory, you only need one extra dependency:

```shell
pip install ring-flash-attn
```

The rest is the same with LLaMA-Factory, except that you clone and use this repo:
```shell
git clone https://github.com/Qihoo360/360-LLaMA-Factory.git
cd 360-LLaMA-Factory
pip install -e ".[torch,metrics]"  # or `pip install --no-deps -e .`
                                   # no need to pip install if you use it as `deepspeed src/train.py`
```


To train with sequence parallelism, you only need one extra argument:

```shell
deepspeed --hostfile=hostfile.4nodes src/train.py \
    --sequence_parallel_size 4 \  # the extra argument regarding sequence parallelism
    --cutoff_len 100000 \         # LLaMA-Factory's original argument, the sequence length you'd like to trian on. Data would be first padded to this length and then preprocessed with sequence parallelism
```

You could also refer to `360-example.sh` for SFT and DPO scripts.


### Comparison with existing SP frameworks

We'll make a table here
frameworks with SP:
Swift: wrong loss
xTuner: no maintance, incompatabiliy with later transformers and model versions, logging and saving mechanism tricky
OpenRLHF: logging and saving mechanism tricky, cleaner (fewer dependencies) but fewer functionalities
other frameworks no SP


## benchmark

### SFT

| **Device**     | **Model**         | **Sequence Parallel Size** | **Max Length** | **Full Parameter SFT** |
|----------------|-------------------|----------------------------|----------------|------------------------|
| **8 × A100**   | **Qwen2.5-7B**   | 8                          | 210k            | ✅                     |
| **8 × A100**   | **Qwen2.5-14B**  | 8                          | 175k            | ✅                     |
| **32 × A100**  | **Qwen2.5-72B**  | 8                          | 128k            | ✅                     |

### DPO

| **Device**     | **Model**         | **Sequence Parallel Size** | **Max Length** | **Full Parameter DPO** |
|----------------|-------------------|----------------------------|----------------|------------------------|
| **8 × A100**   | **Qwen2.5-7B**   | 4                          | 32k            | ✅                     |
| **8 × A100**   | **Qwen2.5-7B**   | 8                          | 84k            | ✅                     |
| **8 × A100**   | **Qwen2.5-14B**  | 4                          | 32k            | ✅                     |
| **8 × A100**   | **Qwen2.5-14B**  | 8                          | 72k            | ✅                     |
| **32 × A100**  | **Qwen2.5-72B**  | 8                          | 46k            | ✅                     |
| **32 × A100**  | **Qwen2.5-72B**  | 16                         | 86k            | ✅                     |

### Correctness
SFT/DPO loss curves
SP on/off, almost overlap

<!-- ### Results
1. **7B** model supports up to **84k** sequence length training on a single machine with 8 GPUs.
2. **14B** model supports up to **72k** sequence length training on a single machine with 8 GPUs.
3. **72B** model supports up to **84k** sequence length training on a 32-GPU distributed setup.

Our methods demonstrate exceptional long-context handling capabilities, effectively supporting large-scale, long-sequence tasks across various hardware configurations. -->

<!-- ## Features
- **Various Methods**: Support advanced algorithms, derived from LLama-factory, support PPO, DPO, KTO, ORPO, etc.
- **Sequence Parallel**: Supports long text sequence parallel and implements DPO sequence parallel based on RingAttention. -->


## Todo list
- [ ] **Precompute**: Precompute the logits of the reference model to reduce memory during DPO
- [ ] **logits.float()/contiguous()**: huge memory consumption on long sequences, could be optimized
- [ ] **Overall SP**: add SP to train/pt, rm, kto
- [ ] **Other SP modes**: DeepSpeed Ulysses and llama3-style SP


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

This repo benefits from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention), [EasyContext](https://github.com/jzhang38/EasyContext) and all repos they benefited from. Thanks for their wonderful works.
