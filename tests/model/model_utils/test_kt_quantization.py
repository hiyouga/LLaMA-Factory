# Copyright 2026 the LlamaFactory team.
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

from types import SimpleNamespace

import pytest

from llamafactory.model.model_utils.quantization import configure_quantization


def _fp8_config():
    return SimpleNamespace(quantization_config={"quant_method": "fp8", "bits": 8})


def _model_args(**overrides):
    values = {
        "use_kt": False,
        "kt_non_expert_weight_path": None,
        "quantization_bit": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_kt_weight_cache_skips_source_fp8_dequantizer():
    config = _fp8_config()
    source_quantization = config.quantization_config.copy()
    init_kwargs = {}

    configure_quantization(
        config,
        tokenizer=None,
        model_args=_model_args(use_kt=True, kt_non_expert_weight_path="/weights/nonexpert"),
        is_trainable=True,
        init_kwargs=init_kwargs,
    )

    assert "quantization_config" not in init_kwargs
    assert "ignore_mismatched_sizes" not in init_kwargs
    assert config.quantization_config == source_quantization


def test_kt_weight_cache_rejects_on_the_fly_quantization():
    with pytest.raises(ValueError, match="quantization_bit.*KT weight caches"):
        configure_quantization(
            _fp8_config(),
            tokenizer=None,
            model_args=_model_args(
                use_kt=True,
                kt_non_expert_weight_path="/weights/nonexpert",
                quantization_bit=4,
            ),
            is_trainable=True,
            init_kwargs={},
        )


@pytest.mark.parametrize(
    "model_args",
    (
        _model_args(),
        _model_args(use_kt=True),
    ),
)
def test_source_fp8_loading_keeps_dequantizer_without_kt_cache(model_args):
    init_kwargs = {}

    configure_quantization(
        _fp8_config(),
        tokenizer=None,
        model_args=model_args,
        is_trainable=True,
        init_kwargs=init_kwargs,
    )

    assert init_kwargs["quantization_config"].dequantize is True
    assert init_kwargs["ignore_mismatched_sizes"] is True
