import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Callable
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Model
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RotaryEmbedding, Qwen2DecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, eager_attention_forward, ALL_ATTENTION_FUNCTIONS
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

import torch
import copy

from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from dataclasses import dataclass

from .router import GroupedRouter 

class Qwen2RouterConfig(Qwen2Config):
    model_type = "qwen2_router"
    def __init__(
        self,
        ...
    ):
        super().__init__()
        ...

@dataclass
class Qwen2RouterOutput(CausalLMOutputWithPast):
    lm_loss: Optional[torch.FloatTensor] = None
    Router_loss: Optional[torch.FloatTensor] = None


class Qwen2ModelWithRouter():
    ...

class Qwen2RouterForCausalLM(Qwen2ForCausalLM):
    config_class = Qwen2RouterConfig

    def __init__(self, config: Qwen2RouterConfig):
        super().__init__(config)

        print(f"In Qwen2RouterForCausalLM, config.layer_stages: {config.layer_stages}")

        self.model = Qwen2ModelWithRouter(config)

        self.Router_rotary_emb = self._init_shared_Router_rope(config)
        self.compute_lm_loss = True
        # config.stage = 2 or any 2 in config.layer_stages will compute lm loss


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        
        if inputs_embeds is None:
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
        else:
            batch_size, seq_length = inputs_embeds.shape[:2]
            device = inputs_embeds.device

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_length, device=device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        ref_tensor = self.model.embed_tokens.weight  # for position embedding device and dtype
        
        Router_cos, Router_sin = self.Router_rotary_emb(ref_tensor, position_ids)
        kwargs['Router_position_embeddings'] = (Router_cos, Router_sin)        
        
        outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep, 
                **kwargs
            )
        
        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        # loss = None

        lm_loss = torch.tensor(0.0, device=logits.device)
        if labels is not None:
            lm_loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        
        # ⚠️ 这里
        Router_loss = getattr(outputs, "Router_loss", torch.tensor(0.0, device=logits.device))


        total_loss = lm_loss + Router_loss

        # Here in forward, both lm_loss and Router_loss are computed and can log into the terminal

        return Qwen2RouterOutput(
            loss=total_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            lm_loss=lm_loss,
            Router_loss=Router_loss,
        )