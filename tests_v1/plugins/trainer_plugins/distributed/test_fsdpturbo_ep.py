from types import SimpleNamespace

import torch

from llamafactory.v1.plugins.trainer_plugins.distributed.fsdp2 import get_transformer_layer_cls
from llamafactory.v1.plugins.trainer_plugins.distributed.mindspeed_fsdp2 import (
    FSDPTurboEPModelSpec,
    _import_fsdpturbo_ep,
)


class _Model(torch.nn.Module):
    def __init__(self, model_type: str):
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type)


def test_fsdpturbo_ep_api_is_imported_from_external_package():
    expert_parallelize, expert_fully_shard, ep_config, fsdp_config, module_match = _import_fsdpturbo_ep()

    assert expert_parallelize.__module__.startswith("fsdp_turbo.")
    assert expert_fully_shard.__module__.startswith("fsdp_turbo.")
    assert ep_config.__module__.startswith("fsdp_turbo.")
    assert fsdp_config.__module__.startswith("fsdp_turbo.")
    assert module_match.__module__.startswith("fsdp_turbo.")


def test_qwen35_conditional_model_uses_language_model_expert_paths():
    spec = FSDPTurboEPModelSpec.get(_Model("qwen3_5_moe"))

    assert spec is not None
    assert spec.ep_modules == ["model.language_model.layers.{*}.mlp.experts"]
    assert spec.ep_fsdp_modules == ["model.language_model.layers.{*}.mlp"]


def test_transformer_layer_prefers_nested_language_model_over_vision_blocks():
    class TextDecoderLayer(torch.nn.Module):
        pass

    class VisionBlock(torch.nn.Module):
        pass

    model = _Model("qwen3_5_moe")
    model.model = torch.nn.Module()
    model.model.visual = torch.nn.ModuleList([VisionBlock()])
    model.model.language_model = torch.nn.Module()
    model.model.language_model.layers = torch.nn.ModuleList([TextDecoderLayer()])
    model._no_split_modules = {"VisionBlock", "TextDecoderLayer"}

    assert get_transformer_layer_cls(model) is TextDecoderLayer
