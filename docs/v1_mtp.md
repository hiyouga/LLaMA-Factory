# Multi-Token Prediction (MTP) — v1 architecture

This feature adds Multi-Token Prediction (MTP) support to the v1 architecture, adapted
from the FSDP2 MTP implementation in
[MindSpeed-LLM](https://gitcode.com/Ascend/MindSpeed-LLM) (`mindspeed_llm/fsdp2/models/common/mtp.py`).

MTP appends `K` extra prediction heads to a decoder-only causal LM. Each head `k`
predicts the token at offset `k + 2` from the current position (the main head predicts
offset `1`). The total training loss is:

```
total_loss = lm_loss + loss_scale * mtp_loss
```

where `mtp_loss` is the mean of the per-head cross-entropy losses (weighted by
`loss_weights`, like the main SFT loss), and `loss_scale` defaults to `0.3`.

## How it works

- `MultiTokenPredictionBlock` (`src/llamafactory/v1/plugins/model_plugins/mtp.py`) holds
  `K` heads. Each head reuses the base model's decoder layer class. The block owns shared
  `enorm`/`hnorm` norms, `e_proj`/`h_proj` projections and a `final_layernorm`, exactly as
  in MindSpeed-LLM.
- `MTPModelPlugin` attaches the block to the model as `model.mtp` and patches
  `model.forward` so the model output carries `mtp_logits` (a list of per-head logits)
  during training. The MTP loss is computed by the trainer through `compute_mtp_loss`.
- Under FSDP2, the MTP heads' inner decoder layers are sharded automatically: the generic
  `FSDP2Engine.prepare_model` wraps every module of the base model's decoder-layer class,
  which includes `mtp.layers.*.layer`.

## Usage

Add an `mtp_config` block to your v1 YAML:

```yaml
model: Qwen/Qwen3-0.6B
model_class: llm
template: qwen3_nothink

mtp_config:
  name: mtp
  num_layers: 1   # number of MTP heads (K)
  loss_scale: 0.3 # optional, default 0.3

dist_config:
  name: fsdp2
  dcp_path: null

train_dataset: data/v1_sft_demo.yaml
output_dir: outputs/test_mtp_fsdp2
micro_batch_size: 1
cutoff_len: 2048
learning_rate: 1.0e-4
max_steps: 10
```

See `examples/v1/train_full/train_full_mtp_fsdp2.yaml` for a complete example.

## Compatibility

MTP currently targets Llama/Qwen3/Mistral-style models that expose `model.model.layers`,
`model.model.rotary_emb`, `model.model.norm` and `model.lm_head`. The MTP heads are
randomly initialized; loading a base checkpoint that does not contain `mtp.*` keys will
leave them at their initialization (missing-key warnings are expected).
