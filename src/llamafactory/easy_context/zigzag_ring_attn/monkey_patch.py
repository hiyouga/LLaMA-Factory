import transformers
from typing import List, Optional, Tuple, Union
import warnings
import torch
import torch.utils.checkpoint
from ring_flash_attn.zigzag_ring_flash_attn import zigzag_ring_flash_attn_func


def new_flash_attn_forward(
    self,
    query_states,
    key_states,
    value_states,
    attention_mask,
    query_length,
    dropout=0.0,
    softmax_scale=None,
    use_sliding_windows=False,
):
    if not self._flash_attn_uses_top_left_mask:
        causal = self.is_causal
    else:
        causal = self.is_causal and query_length != 1

    # Contains at least one padding token in the sequence
    assert attention_mask is None
    assert causal is True
    assert use_sliding_windows is False
    attn_output = zigzag_ring_flash_attn_func(
        query_states,
        key_states,
        value_states,
        dropout,
        softmax_scale,
        causal=causal,
    )

    return attn_output


def new_decoder_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    assert isinstance(
        self.self_attn, transformers.models.llama.modeling_llama.LlamaFlashAttention2
    ) or isinstance(
        self.self_attn,
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2,
    ), "Please toggle on the Flash Attention 2 implementation when using zigzag ring attention monkey patch."

    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def apply_zigzag_ring_attn_monkey_patch_llama():
    transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward = (
        new_flash_attn_forward
    )
    transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = (
        new_decoder_forward
    )


def apply_zigzag_ring_attn_monkey_patch_mistral():
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2._flash_attention_forward = (
        new_flash_attn_forward
    )
    transformers.models.mistral.modeling_mistral.MistralDecoderLayer.forward = (
        new_decoder_forward
    )
