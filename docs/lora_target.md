## LoRA Target Parameter

### What is lora_target?

The `lora_target` parameter controls which layers in your model get LoRA adapters during fine-tuning. You can apply LoRA to all compatible layers or target specific modules.

### Basic Usage

```yaml
# Apply to all linear modules (recommended default)
lora_target: all

# Target specific modules
lora_target: q_proj,v_proj
```

### Common Module Names

Different models use different naming conventions:

**LLaMA/Mistral/Alpaca:**
- Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- MLP: `gate_proj`, `up_proj`, `down_proj`

**ChatGLM:**
- Attention: `query_key_value`
- MLP: `dense`, `dense_h_to_4h`, `dense_4h_to_h`

**Qwen:**
- Attention: `c_attn`, `c_proj`
- MLP: `w1`, `w2`

### Finding Module Names

The easiest way is to set `lora_target: all` and check the training logs for "Found linear modules:" - this shows all available modules for your model.

### Examples

**Memory-constrained training:**
```yaml
# Target only attention layers
lora_target: q_proj,k_proj,v_proj,o_proj
lora_rank: 8
```

**Task-specific fine-tuning:**
```yaml
# Mix attention and MLP modules
lora_target: q_proj,v_proj,gate_proj,down_proj
lora_rank: 16
```

### Tips

- Start with `all` unless you have memory constraints
- For classification tasks, attention modules often suffice
- For knowledge-intensive tasks, include MLP modules
- Wrong module names will cause errors - double-check against your model

### Troubleshooting

**"Module not found" error:** Check module names match your model architecture. Use `lora_target: all` first to see available options.

**High memory usage:** Target fewer modules or just attention layers.

**Poor results:** Try using `all` or including both attention and MLP modules.
