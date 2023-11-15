import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.utils import logging
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb

try:
    from transformers.models.llama.modeling_llama import repeat_kv
except ImportError:
    print("Please upgrade `transformers`.")

from llmtuner.extras.packages import is_flash_attn2_available


if is_flash_attn2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func # type: ignore
    from flash_attn.bert_padding import pad_input, unpad_input # type: ignore


logger = logging.get_logger(__name__)


# Modified from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
class LlamaShiftShortAttention(LlamaAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None: # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        if getattr(self, "num_key_value_groups"):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        if getattr(self.config, "group_size_ratio", None) and self.training: # shift
            groupsz = int(q_len * getattr(self.config, "group_size_ratio"))
            assert q_len % groupsz == 0, "q_len {} should be divisible by group size {}.".format(q_len, groupsz)
            num_groups = q_len // groupsz
            def shift(state: torch.Tensor) -> torch.Tensor:
                state = state.transpose(1, 2) # output: (bsz, seq_len, n_heads, head_dim)
                state = torch.cat((
                    state[:, :, :self.num_heads//2], state[:, :, self.num_heads//2:].roll(-groupsz//2, dims=1)
                ), dim=2)
                return state.reshape(bsz * num_groups, groupsz, self.num_heads, self.head_dim).transpose(1, 2)

            query_states, key_states, value_states = shift(query_states), shift(key_states), shift(value_states)
            if attention_mask is not None:
                attention_mask = attention_mask[:, :, :groupsz, :groupsz].repeat(num_groups, 1, 1, 1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states) # (bsz, :, seq_len, :) or (bsz*n_group, :, groupsz, :)
        attn_output = attn_output.transpose(1, 2).contiguous()

        if getattr(self.config, "group_size_ratio", None) and self.training: # shift back
            attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)
            attn_output = torch.cat((
                attn_output[:, :, :self.num_heads//2], attn_output[:, :, self.num_heads//2:].roll(groupsz//2, dims=1)
            ))

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaFlashAttention2(LlamaAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # FlashAttention requires the input to have the shape (bsz, seq_len, n_heads, head_dim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None: # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # cast to half precision
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            logger.warning_once("The input hidden states seems to be silently casted in float32.")
            query_states = query_states.to(self.config.torch_dtype)
            key_states = key_states.to(self.config.torch_dtype)
            value_states = value_states.to(self.config.torch_dtype)

        if getattr(self, "num_key_value_groups", None):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        query_states = query_states.transpose(1, 2) # (bsz, seq_len, n_heads, head_dim)
        key_states = key_states.transpose(1, 2) # (bsz, seq_len, n_heads, head_dim)
        value_states = value_states.transpose(1, 2) # (bsz, seq_len, n_heads, head_dim)

        if getattr(self.config, "group_size_ratio", None) and self.training: # shift
            groupsz = int(q_len * getattr(self.config, "group_size_ratio"))
            assert q_len % groupsz == 0, "q_len {} should be divisible by group size {}.".format(q_len, groupsz)
            num_groups = q_len // groupsz
            def shift(state: torch.Tensor) -> torch.Tensor:
                state = torch.cat((
                    state[:, :, :self.num_heads//2], state[:, :, self.num_heads//2:].roll(-groupsz//2, dims=1)
                ), dim=2)
                return state.reshape(bsz * num_groups, groupsz, self.num_heads, self.head_dim)

            query_states, key_states, value_states = shift(query_states), shift(key_states), shift(value_states)
            if attention_mask is not None:
                attention_mask = attention_mask.reshape(bsz * num_groups, groupsz)

        if attention_mask is not None:
            logger.warning_once("Padded sequences are less efficient in FlashAttention.")
            # -q_len: assumes left padding when q_len != kv_len
            unpadded_q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(query_states, attention_mask[:, -q_len:])
            unpadded_k, _, cu_seqlens_k, max_seqlen_k = unpad_input(key_states, attention_mask)
            unpadded_v, _, _, _ = unpad_input(value_states, attention_mask)
            attn_output_unpad = flash_attn_varlen_func(
                unpadded_q,
                unpadded_k,
                unpadded_v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=None,
                causal=True,
            )
            attn_output = pad_input(attn_output_unpad, indices_q, bsz, q_len)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, 0.0, softmax_scale=None, causal=True
            )

        if getattr(self.config, "group_size_ratio", None) and self.training: # shift back
            attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)
            attn_output = torch.cat((
                attn_output[:, :, :self.num_heads//2], attn_output[:, :, self.num_heads//2:].roll(groupsz//2, dims=1)
            ))

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Disable the transformation of the attention mask in LlamaModel as flash attention
# takes a boolean padding_mask. Fills in the past kv length for use in forward.
def _prepare_decoder_attention_mask(
    self,
    attention_mask: torch.Tensor,
    input_shape: torch.Tensor,
    inputs_embeds: torch.Tensor,
    past_key_values_length: int
) -> torch.Tensor:
    if attention_mask is not None and torch.all(attention_mask):
        return None  # This uses the faster call when training with full samples

    return attention_mask
