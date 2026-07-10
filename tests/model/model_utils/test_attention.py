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

import os

import pytest
from transformers.utils import is_flash_attn_2_available


# Compatible with Transformers v4 and Transformers v5
try:
    from transformers.utils import is_torch_sdpa_available
except ImportError:

    def is_torch_sdpa_available():
        return True


from llamafactory.extras.packages import is_transformers_version_greater_than
from llamafactory.train.test_utils import load_infer_model


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

INFER_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "template": "llama3",
}


@pytest.mark.xfail(is_transformers_version_greater_than("4.48"), reason="Attention refactor.")
def test_attention():
    attention_available = ["disabled"]
    if is_torch_sdpa_available():
        attention_available.append("sdpa")

    if is_flash_attn_2_available():
        attention_available.append("fa2")

    llama_attention_classes = {
        "disabled": "LlamaAttention",
        "sdpa": "LlamaSdpaAttention",
        "fa2": "LlamaFlashAttention2",
    }
    for requested_attention in attention_available:
        model = load_infer_model(flash_attn=requested_attention, **INFER_ARGS)
        for module in model.modules():
            if "Attention" in module.__class__.__name__:
                assert module.__class__.__name__ == llama_attention_classes[requested_attention]


def _is_flash_attn_4_available() -> bool:
    try:
        from transformers.utils import is_flash_attn_4_available
    except ImportError:
        return False

    return is_flash_attn_4_available()


def test_configure_attn_implementation_fa4():
    r"""`flash_attn: fa4` maps to the transformers `flash_attention_4` implementation string."""
    from types import SimpleNamespace

    from llamafactory.extras.constants import AttentionFunction
    from llamafactory.model.model_utils.attention import configure_attn_implementation

    config = SimpleNamespace(model_type="llama")
    model_args = SimpleNamespace(flash_attn=AttentionFunction.FA4)
    configure_attn_implementation(config, model_args)

    if _is_flash_attn_4_available():
        assert getattr(config, "_attn_implementation", None) == "flash_attention_4"
    else:
        # Unavailable fa4 must not crash and must not set the implementation.
        assert getattr(config, "_attn_implementation", None) is None


def test_fa4_vision_fallback_helper_head_dim_math():
    r"""`_fa4_vision_needs_fa2` derives vision head_dim from num_heads and applies the %32 rule.

    The fa4 backward-preprocess kernel crashes when head_dim is not a multiple of 32 (its padding
    granularity), so the fallback triggers on % 32 != 0 -- not the coarser % 64.
    """
    from types import SimpleNamespace

    from llamafactory.model.model_utils.attention import _fa4_vision_needs_fa2, _sub_config_head_dim

    bad_vision = SimpleNamespace(hidden_size=1152, num_heads=16)  # head_dim 72 (72 % 32 != 0) -> unsupported
    good_vision = SimpleNamespace(hidden_size=1024, num_heads=16)  # head_dim 64 (64 % 32 == 0) -> supported
    ok32_vision = SimpleNamespace(hidden_size=1536, num_heads=16)  # head_dim 96 (96 % 32 == 0) -> supported
    assert _sub_config_head_dim(bad_vision) == 72
    assert _sub_config_head_dim(good_vision) == 64
    assert _sub_config_head_dim(ok32_vision) == 96
    assert _fa4_vision_needs_fa2(SimpleNamespace(vision_config=bad_vision)) is True
    assert _fa4_vision_needs_fa2(SimpleNamespace(vision_config=good_vision)) is False
    assert _fa4_vision_needs_fa2(SimpleNamespace(vision_config=ok32_vision)) is False  # %64!=0 but %32==0: safe
    assert _fa4_vision_needs_fa2(SimpleNamespace()) is False  # plain LLM, no vision tower


@pytest.mark.skipif(not _is_flash_attn_4_available(), reason="FlashAttention-4 is not installed.")
def test_configure_attn_implementation_fa4_vision_fallback():
    r"""A vision tower with head_dim not divisible by 64 is routed to fa2 while the LLM uses fa4."""
    from types import SimpleNamespace

    from llamafactory.extras.constants import AttentionFunction
    from llamafactory.model.model_utils.attention import configure_attn_implementation

    # Qwen3-VL / Qwen3.5-like: vision hidden 1152 / 16 heads = head_dim 72 (unsupported by fa4 backward).
    config = SimpleNamespace(
        model_type="qwen3_vl",
        vision_config=SimpleNamespace(hidden_size=1152, num_heads=16),
        text_config=SimpleNamespace(hidden_size=4096, num_attention_heads=32, head_dim=128),
    )
    model_args = SimpleNamespace(flash_attn=AttentionFunction.FA4)
    configure_attn_implementation(config, model_args)

    assert config._attn_implementation == {
        "": "flash_attention_4",
        "text_config": "flash_attention_4",
        "vision_config": "flash_attention_2",
    }
