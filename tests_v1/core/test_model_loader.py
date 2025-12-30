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

import torch

from llamafactory.v1.config.model_args import ModelArguments, PluginConfig
from llamafactory.v1.core.model_loader import ModelLoader


def test_tiny_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2TokenizerFast

    model_args = ModelArguments(model="llamafactory/tiny-random-qwen2.5")
    model_loader = ModelLoader(model_args)
    assert isinstance(model_loader.processor, Qwen2TokenizerFast)
    assert isinstance(model_loader.model.config, Qwen2Config)
    assert isinstance(model_loader.model, Qwen2ForCausalLM)
    assert model_loader.model.dtype == torch.bfloat16


def test_tiny_qwen_with_kernel_plugin():
    from transformers import Qwen2ForCausalLM

    from llamafactory.v1.plugins.model_plugins.kernels.ops.rms_norm.npu_rms_norm import npu_rms_norm_forward

    model_args = ModelArguments(
        model="llamafactory/tiny-random-qwen2.5", kernel_config=PluginConfig(name="auto", include_kernels="auto")
    )
    model_loader = ModelLoader(model_args)
    # test enable apply kernel plugin
    if hasattr(torch, "npu"):
        assert model_loader.model.model.layers[0].input_layernorm.forward.__code__ == npu_rms_norm_forward.__code__
    else:
        assert model_loader.model.model.layers[0].input_layernorm.forward.__code__ != npu_rms_norm_forward.__code__
    assert isinstance(model_loader.model, Qwen2ForCausalLM)


if __name__ == "__main__":
    test_tiny_qwen()
    test_tiny_qwen_with_kernel_plugin()
