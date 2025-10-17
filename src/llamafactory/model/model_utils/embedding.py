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

import math
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional

import torch
from transformers.integrations import is_deepspeed_zero3_enabled

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


logger = logging.get_logger(__name__)


def _noisy_mean_initialization(embed_weight: "torch.Tensor", num_new_tokens: int) -> None:
    """Initialize new token embeddings with mean + Gaussian noise.

    This is the default initialization method used by LlamaFactory.

    Args:
        embed_weight: The embedding weight matrix to initialize (shape: [vocab_size, embedding_dim])
        num_new_tokens: Number of new tokens added at the end of the embedding matrix
    """
    embedding_dim = embed_weight.size(1)
    avg_weight = embed_weight[:-num_new_tokens].mean(dim=0, keepdim=True)
    noise_weight = torch.empty_like(embed_weight[-num_new_tokens:])
    noise_weight.normal_(mean=0, std=(1.0 / math.sqrt(embedding_dim)))
    embed_weight[-num_new_tokens:] = avg_weight + noise_weight


def _description_based_initialization(
    embed_weight: "torch.Tensor",
    num_new_tokens: int,
    descriptions: dict[str, str],
    tokenizer: "PreTrainedTokenizer",
    model: "PreTrainedModel",
    add_noise: bool = False,
) -> None:
    """Initialize new token embeddings based on textual descriptions.

    For each new token, this function:
    1. Tokenizes its description text
    2. Gets embeddings of the description tokens
    3. Averages them to initialize the new token's embedding
    4. Optionally adds Gaussian noise

    Args:
        embed_weight: The embedding weight matrix to initialize (shape: [vocab_size, embedding_dim])
        num_new_tokens: Number of new tokens added
        descriptions: Dict mapping token string to its description text
                      e.g., {"<think>": "A token representing reasoning process"}
        tokenizer: The tokenizer instance
        model: The model instance (used to get input embeddings)
        add_noise: Whether to add Gaussian noise to the initialization

    Example:
        descriptions = {
            "<|START_OF_SVG|>": "Marks the beginning of an SVG document",
            "<|END_OF_SVG|>": "Marks the end of an SVG document"
        }
    """
    embedding_dim = embed_weight.size(1)

    for i, desc in enumerate(descriptions.values()):
        # Tokenize description text
        tokens = tokenizer(desc, return_tensors="pt", add_special_tokens=False)

        with torch.no_grad():
            token_ids = tokens["input_ids"][0]
            # Move to the same device as embed_weight
            device = embed_weight.device
            token_ids = token_ids.to(device)

            # Filter out new tokens (they don't have valid embeddings yet)
            valid_token_ids = token_ids[token_ids < (len(tokenizer) - num_new_tokens)]

            if len(valid_token_ids) == 0:
                # Fallback: use mean of all existing embeddings
                logger.warning_rank0(
                    f"Description for token {i + 1}/{num_new_tokens} contains no valid tokens. "
                    "Using mean of existing embeddings."
                )
                base_embedding = embed_weight[:-num_new_tokens].mean(dim=0)
            else:
                # Get embeddings of description tokens and average them
                token_embeds = model.get_input_embeddings()(valid_token_ids)
                base_embedding = token_embeds.mean(dim=0)

            # Add noise if requested (ensure correct device and dtype)
            if add_noise:
                noise = torch.randn_like(base_embedding) * (1.0 / math.sqrt(embedding_dim))
                embed_weight[-num_new_tokens + i] = base_embedding + noise
            else:
                embed_weight[-num_new_tokens + i] = base_embedding


def _initialize_embeddings(
    embed_weight: "torch.Tensor",
    num_new_tokens: int,
    init_method: str,
    new_special_tokens_config: Optional[dict],
    tokenizer: "PreTrainedTokenizer",
    model: "PreTrainedModel",
) -> None:
    """Single source of truth for embedding initialization.

    This function selects the appropriate initialization method and applies it.

    Args:
        embed_weight: The embedding weight matrix to initialize
        num_new_tokens: Number of new tokens added
        init_method: Initialization method ('noise_init', 'desc_init', 'desc_init_w_noise')
        new_special_tokens_config: Config dict with token descriptions (required for desc_init methods)
        tokenizer: The tokenizer instance
        model: The model instance
    """
    if init_method == "desc_init" and new_special_tokens_config:
        logger.info_rank0("Using semantic initialization (desc_init) for new special tokens")
        _description_based_initialization(
            embed_weight, num_new_tokens, new_special_tokens_config, tokenizer, model, add_noise=False
        )
    elif init_method == "desc_init_w_noise" and new_special_tokens_config:
        logger.info_rank0("Using semantic initialization with noise (desc_init_w_noise) for new special tokens")
        _description_based_initialization(
            embed_weight, num_new_tokens, new_special_tokens_config, tokenizer, model, add_noise=True
        )
    else:
        if init_method != "noise_init":
            logger.warning_rank0(
                f"init_method='{init_method}' requires descriptions config, falling back to 'noise_init'"
            )
        logger.info_rank0("Using noisy mean initialization (noise_init) for new special tokens")
        _noisy_mean_initialization(embed_weight, num_new_tokens)


def resize_embedding_layer(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    new_special_tokens_config: Optional[dict] = None,
    init_special_tokens: str = "noise_init",
) -> None:
    r"""Resize token embeddings and initialize new tokens.

    Args:
        model: The model to resize
        tokenizer: The tokenizer (used to get target vocab size)
        new_special_tokens_config: Optional dict with token descriptions for semantic initialization
        init_special_tokens: Initialization method ('noise_init', 'desc_init', 'desc_init_w_noise')
    """
    if is_deepspeed_zero3_enabled():
        import deepspeed  # type: ignore

        params = [model.get_input_embeddings().weight]
        if model.get_output_embeddings() is not None and not model.config.tie_word_embeddings:
            params.append(model.get_output_embeddings().weight)

        context_maybe_zero3 = deepspeed.zero.GatheredParameters(params, modifier_rank=0)
    else:
        context_maybe_zero3 = nullcontext()

    with context_maybe_zero3:
        current_embedding_size = model.get_input_embeddings().weight.size(0)

    if len(tokenizer) > current_embedding_size:
        if getattr(model, "quantization_method", None):
            raise ValueError("Cannot resize embedding layers of a quantized model.")

        if not isinstance(model.get_output_embeddings(), torch.nn.Linear):
            raise ValueError("Current model does not support resizing embedding layers.")

        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
        with context_maybe_zero3:
            new_embedding_size = model.get_input_embeddings().weight.size(0)
            num_new_tokens = new_embedding_size - current_embedding_size
            logger.info_rank0(
                f"Resizing embeddings: {current_embedding_size} -> {new_embedding_size} (+{num_new_tokens} tokens)"
            )

            # Initialize input embeddings
            _initialize_embeddings(
                model.get_input_embeddings().weight.data,
                num_new_tokens,
                init_special_tokens,
                new_special_tokens_config,
                tokenizer,
                model,
            )

            # Initialize output embeddings if not tied
            if model.get_output_embeddings() is not None and not model.config.tie_word_embeddings:
                _initialize_embeddings(
                    model.get_output_embeddings().weight.data,
                    num_new_tokens,
                    init_special_tokens,
                    new_special_tokens_config,
                    tokenizer,
                    model,
                )

        model.config.vocab_size = new_embedding_size
        logger.info_rank0(f"Resized token embeddings from {current_embedding_size} to {new_embedding_size}.")
