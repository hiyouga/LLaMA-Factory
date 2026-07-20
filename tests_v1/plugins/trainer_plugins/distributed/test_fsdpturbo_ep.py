from types import SimpleNamespace

import torch

from llamafactory.v1.plugins.trainer_plugins.distributed.fsdp2 import get_transformer_layer_cls
from llamafactory.v1.plugins.trainer_plugins.distributed.fsdpturbo import (
    FSDPTurboEPModelSpec,
    get_fsdpturbo_mesh_specs,
)
from llamafactory.v1.plugins.trainer_plugins.distributed.hub import DistributedPlugin


class _Model(torch.nn.Module):
    def __init__(self, model_type: str):
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type)


def test_qwen35_support_is_not_hardcoded_in_model_registry():
    assert FSDPTurboEPModelSpec.get(_Model("qwen3_5_moe")) is None


def test_fsdpturbo_declares_expert_mesh_topology_through_plugin():
    strategy = SimpleNamespace(dp_size=16, cp_size=1, ep_size=8)

    specs = get_fsdpturbo_mesh_specs(strategy)
    assert len(specs) == 1
    assert specs[0].name == "fsdpturbo_expert"
    assert specs[0].mesh_shape == (1, 2, 8, 1)
    assert specs[0].mesh_dim_names == ("edp", "efsdp", "ep", "expert_cp")
    assert DistributedPlugin("fsdpturbo").has_method("mesh_specs")

    assert get_fsdpturbo_mesh_specs(SimpleNamespace(dp_size=16, cp_size=1, ep_size=1)) == ()


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
