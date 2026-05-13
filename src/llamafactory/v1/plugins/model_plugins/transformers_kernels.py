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

"""Transformers kernel options for model loading."""

from typing import Any

from ...utils import logging


logger = logging.get_logger(__name__)


def update_transformers_kernels_kwargs(
    flash_attn: Any = "auto",
    init_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update ``init_kwargs`` with transformers kernel options for ``from_pretrained()``.

    Args:
        flash_attn: Attention implementation to use.
        init_kwargs: Existing kwargs dict to update. A new dict is created if ``None``.

    Returns:
        The updated ``init_kwargs`` dict.
    """
    init_kwargs = dict(init_kwargs) if init_kwargs else {}
    flash_attn = getattr(flash_attn, "value", flash_attn)

    if flash_attn in (None, "auto") or flash_attn == "sdpa":
        init_kwargs["attn_implementation"] = "sdpa"
    elif flash_attn == "disabled":
        init_kwargs["attn_implementation"] = "eager"
    elif flash_attn == "fa2":
        init_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        raise NotImplementedError(f"Unknown attention type: {flash_attn}")

    logger.info_rank0(f"Using attention implementation: {init_kwargs['attn_implementation']}.")
    return init_kwargs
