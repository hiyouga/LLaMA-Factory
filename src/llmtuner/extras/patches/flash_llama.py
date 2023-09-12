# coding=utf-8
# Modified from:
# [1] https://huggingface.co/Birchlabs/flash_llama/blob/main/modeling_flash_llama.py
# [2] https://github.com/lm-sys/FastChat/blob/main/fastchat/train/llama2_flash_attn_monkey_patch.py
# [3] https://huggingface.co/togethercomputer/LLaMA-2-7B-32K/blob/main/modeling_flash_llama.py
# [4] https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# With fix from Alex Birch: https://huggingface.co/togethercomputer/LLaMA-2-7B-32K/discussions/17

import torch
from typing import TYPE_CHECKING, Optional, Tuple
from transformers.utils import logging

if TYPE_CHECKING:
    from transformers.models.llama.configuration_llama import LlamaConfig

try:
    from flash_attn.flash_attn_interface import (
        flash_attn_kvpacked_func,
        flash_attn_varlen_kvpacked_func
    )
    from flash_attn.bert_padding import pad_input, unpad_input
    print(">>>> FlashAttention installed")
except ImportError:
    raise ImportError("Please install FlashAttention from https://github.com/Dao-AILab/flash-attention")

try:
    from flash_attn.layers.rotary import apply_rotary_emb_func
    print(">>>> Flash RoPE installed")
except ImportError:
    raise ImportError("Please install RoPE kernels from https://github.com/Dao-AILab/flash-attention")


logger = logging.get_logger(__name__)


class LlamaRMSNorm(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype) # for fp32 weight


class FlashRotaryEmbedding(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        scaling_factor=1.0,
        pos_idx_in_fp32=True,
        device=None
    ):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.scaling_factor = scaling_factor
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            if scale_base is not None else None
        )
        self.register_buffer("scale", scale)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _compute_inv_freq(self, device=None):
        return 1 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        if (
            seqlen > self._seq_len_cached or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                t /= self.scaling_factor
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self.inv_freq.to(torch.float32)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                t /= self.scaling_factor
                inv_freq = self.inv_freq
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    (torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2) / self.scale_base
                )
                scale = self.scale.to(device=power.device) ** power.unsqueeze(-1)
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seqlen_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        q: (batch, seqlen, nheads, headdim)
        k: (batch, seqlen, nheads, headdim)
        seqlen_offset: can be used in generation where the qkv being passed in is only the last
        token in the batch.
        """
        self._update_cos_sin_cache(q.shape[1] + seqlen_offset, device=q.device, dtype=q.dtype)
        if self.scale is None:
            return apply_rotary_emb_func(
                q, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:],
                self.interleaved, True # inplace=True
            ), apply_rotary_emb_func(
                k, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:],
                self.interleaved, True # inplace=True
            )
        else:
            assert False


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    r"""
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, slen, _, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, :, None, :].expand(batch, slen, 2, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, 2, num_key_value_heads * n_rep, head_dim)


class LlamaAttention(torch.nn.Module):

    def __init__(self, config: "LlamaConfig"):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = torch.nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = torch.nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = torch.nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.register_buffer(
            "norm_factor",
            torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype()),
            persistent=False,
        )

        if self.config.rope_scaling is None:
            scaling_factor = 1
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            assert scaling_type == "linear"

        self.rotary_emb = FlashRotaryEmbedding(
            self.head_dim, base=10000, interleaved=False, scaling_factor=scaling_factor
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, h_size = hidden_states.size()

        has_layer_past = past_key_value is not None

        if has_layer_past:
            past_kv = past_key_value[0]
            past_len = past_key_value[1]
        else:
            past_len = 0

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        q, k = self.rotary_emb(q, k, past_len)

        kv = torch.stack([k, v], 2)
        kv = repeat_kv(kv, self.num_key_value_groups)

        # Cache QKV values
        if has_layer_past:
            new_len = past_len+q.size(1)
            if new_len > past_kv.size(1):
                past_kv = torch.cat(
                    [past_kv, torch.empty(bsz, 256, 2, kv.size(3), kv.size(4), dtype=kv.dtype, device=kv.device)],
                    dim=1
                )
            past_kv[:, past_len:new_len] = kv
            kv = past_kv[:, :new_len]
        else:
            past_kv = kv

        past_key_value = (past_kv, past_len + q.size(1)) if use_cache else None

        if attention_mask is not None:
            # varlen, ignore padding tokens, efficient for large batch with many paddings
            logger.warning_once("padded sequences is less efficient")

            unpadded_kv, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(kv, attention_mask)
            unpadded_q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, attention_mask[:, -q.size(1):])
            attn_outputs = flash_attn_varlen_kvpacked_func(
                unpadded_q, unpadded_kv, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                dropout_p=0.0, softmax_scale=1.0 / self.norm_factor,
                causal=(not has_layer_past), return_attn_probs=output_attentions
            )

            attn_output = attn_outputs[0] if output_attentions else attn_outputs
            attn_output = pad_input(attn_output, indices_q, bsz, q_len).reshape(bsz, q_len, h_size)
            attn_weights = attn_outputs[2] if output_attentions else None

        else:
            # no padding tokens, more efficient
            attn_outputs = flash_attn_kvpacked_func(
                q, kv, dropout_p=0.0, softmax_scale=1.0 / self.norm_factor,
                causal=(not has_layer_past), return_attn_probs=output_attentions
            )
            attn_output = attn_outputs[0] if output_attentions else attn_outputs
            attn_output = attn_output.reshape(bsz, q_len, h_size)
            attn_weights = attn_outputs[2] if output_attentions else None

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Disable the transformation of the attention mask in LlamaModel as flash attention
# takes a boolean key_padding_mask. Fills in the past kv length for use in forward.
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    if past_key_values_length > 0 and attention_mask is not None:
        attention_mask = torch.cat(
            (
                torch.full(
                    (input_shape[0], past_key_values_length),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                ),
                attention_mask
            ),
            dim=-1
        )

    if attention_mask is not None and torch.all(attention_mask):
        return None  # This uses the faster call when training with full samples

    return attention_mask
