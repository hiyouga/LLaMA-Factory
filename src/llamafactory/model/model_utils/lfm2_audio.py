# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom model loader for LFM2.5-Audio models using liquid_audio package.

LFM2.5-Audio models use a custom architecture that requires the liquid_audio package
for proper model loading and audio processing.
"""

from typing import TYPE_CHECKING, Any, Optional

import torch
from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from ...extras import logging
from ...extras.packages import is_liquid_audio_available


if TYPE_CHECKING:
    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


class LFM2AudioConfig(PretrainedConfig):
    """Config class for LFM2.5-Audio models to enable HuggingFace compatibility."""

    model_type = "lfm2_audio"

    def __init__(
        self,
        vocab_size: int = 65536,
        hidden_size: int = 2048,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        codebooks: int = 8,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.codebooks = codebooks
        super().__init__(**kwargs)


class LFM2AudioModelForCausalLM(PreTrainedModel, GenerationMixin):
    """HuggingFace-compatible wrapper for LFM2AudioModel from liquid_audio.

    This wrapper enables LFM2.5-Audio models to be used with LLaMA-Factory's
    training pipeline while leveraging the liquid_audio package for model loading.
    """

    config_class = LFM2AudioConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Lfm2DecoderLayer", "ConformerBlock"]
    main_input_name = "input_ids"
    _supports_cache_class = True

    def __init__(self, config: LFM2AudioConfig):
        super().__init__(config)
        self._liquid_model = None
        self._is_loaded = False
        # Initialize generation_config for HuggingFace compatibility
        self.generation_config = GenerationConfig(
            eos_token_id=config.eos_token_id if hasattr(config, "eos_token_id") else 7,
            pad_token_id=config.pad_token_id if hasattr(config, "pad_token_id") else 0,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        config: Optional[LFM2AudioConfig] = None,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
        **kwargs,
    ) -> "LFM2AudioModelForCausalLM":
        """Load LFM2.5-Audio model using liquid_audio package."""
        if not is_liquid_audio_available():
            raise ImportError(
                "liquid-audio package is required for LFM2.5-Audio models. "
                "Please install it with: pip install liquid-audio"
            )

        from liquid_audio import LFM2AudioModel

        # Determine dtype
        if torch_dtype is None or torch_dtype == "auto":
            torch_dtype = torch.bfloat16

        # Determine device - liquid_audio expects string or torch.device, not dict
        if device_map is None or device_map == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif isinstance(device_map, dict):
            # Handle dict device_map like {'': device(type='cuda', index=0)}
            if "" in device_map:
                dev = device_map[""]
                device = str(dev) if hasattr(dev, "type") else "cuda"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
        elif isinstance(device_map, str):
            device = device_map
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info_rank0(f"Loading LFM2.5-Audio model from {pretrained_model_name_or_path}")
        logger.info_rank0(f"Using dtype={torch_dtype}, device={device}")

        # Load using liquid_audio
        liquid_model = LFM2AudioModel.from_pretrained(
            pretrained_model_name_or_path,
            dtype=torch_dtype,
            device=device,
            revision=kwargs.get("revision"),
        )

        # Create config from liquid model
        lfm_config = liquid_model.conf.lfm
        if config is None:
            config = LFM2AudioConfig(
                vocab_size=lfm_config.vocab_size,
                hidden_size=lfm_config.hidden_size,
                num_hidden_layers=lfm_config.num_hidden_layers,
                num_attention_heads=lfm_config.num_attention_heads,
                num_key_value_heads=lfm_config.num_key_value_heads,
                codebooks=liquid_model.conf.codebooks,
                torch_dtype=torch_dtype,
            )

        # Create wrapper instance
        wrapper = cls(config)
        wrapper._liquid_model = liquid_model
        wrapper._is_loaded = True

        return wrapper

    @property
    def model(self):
        """Return the underlying liquid_audio model."""
        return self._liquid_model

    @property
    def lfm(self):
        """Return the LFM2 backbone (HuggingFace Lfm2Model)."""
        if self._liquid_model is not None:
            return self._liquid_model.lfm
        return None

    def get_input_embeddings(self):
        """Get text embeddings from the LFM backbone."""
        if self.lfm is not None:
            return self.lfm.embed_tokens
        return None

    def set_input_embeddings(self, value):
        """Set text embeddings in the LFM backbone."""
        if self.lfm is not None:
            self.lfm.embed_tokens = value

    def get_output_embeddings(self):
        """LFM2 uses tied embeddings, return the same as input."""
        return self.get_input_embeddings()

    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings (tied with input)."""
        self.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Forward pass for training.

        For training, we use the LFM backbone directly for text-only forward pass.
        The audio processing is handled by the mm_plugin during data preprocessing.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self._liquid_model is None:
            raise RuntimeError("Model not loaded. Call from_pretrained first.")

        # Use the LFM backbone for forward pass
        lfm = self._liquid_model.lfm

        # Get embeddings
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = lfm.embed_tokens(input_ids)

        # Forward through LFM backbone
        outputs = lfm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state

        # Compute logits using tied embeddings
        logits = torch.nn.functional.linear(hidden_states, lfm.embed_tokens.weight)

        loss = None
        if labels is not None:
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Prepare inputs for generation."""
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

        if inputs_embeds is not None and past_key_values is None:
            model_inputs["inputs_embeds"] = inputs_embeds
            model_inputs["input_ids"] = None

        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorder cache for beam search."""
        if self.lfm is not None and hasattr(self.lfm, "_reorder_cache"):
            return self.lfm._reorder_cache(past_key_values, beam_idx)
        return past_key_values

    @classmethod
    def can_generate(cls) -> bool:
        """Return True to indicate this model can generate sequences."""
        return True

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        if self.lfm is not None:
            return next(self.lfm.parameters()).device
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the model."""
        if self.lfm is not None:
            return next(self.lfm.parameters()).dtype
        return torch.float32


def is_lfm2_audio_model(model_name_or_path: str) -> bool:
    """Check if the model is an LFM2.5-Audio model."""
    lfm2_audio_patterns = [
        "LFM2.5-Audio",
        "lfm2.5-audio",
        "lfm2-audio",
        "LFM2-Audio",
    ]
    return any(pattern.lower() in model_name_or_path.lower() for pattern in lfm2_audio_patterns)


def load_lfm2_audio_pretrained_model(
    model_args: "ModelArguments",
    **kwargs,
) -> "LFM2AudioModelForCausalLM":
    """Load LFM2.5-Audio model using liquid_audio package.

    Args:
        model_args: Model arguments containing model path and configuration.
        **kwargs: Additional arguments passed to from_pretrained.

    Returns:
        LFM2AudioModelForCausalLM: Loaded model wrapper.
    """
    if not is_liquid_audio_available():
        raise ImportError(
            "LFM2.5-Audio models require the liquid-audio package. Please install it with: pip install liquid-audio"
        )

    return LFM2AudioModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=kwargs.get("torch_dtype", "auto"),
        device_map=kwargs.get("device_map"),
        revision=model_args.model_revision,
        cache_dir=model_args.cache_dir,
        token=model_args.hf_hub_token,
        trust_remote_code=model_args.trust_remote_code,
    )
