from types import SimpleNamespace

import torch

from llamafactory.v1.plugins.trainer_plugins.distributed import fsdpturbo as fsdpturbo_module
from llamafactory.v1.plugins.trainer_plugins.distributed.fsdp2 import get_transformer_layer_cls
from llamafactory.v1.plugins.trainer_plugins.distributed.fsdpturbo import (
    FSDPTurboEPModelSpec,
    FSDPTurboParallelState,
)


class _Model(torch.nn.Module):
    def __init__(self, model_type: str):
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type)


def test_qwen35_support_is_not_hardcoded_in_model_registry():
    assert FSDPTurboEPModelSpec.get(_Model("qwen3_5_moe")) is None


def test_fsdpturbo_owns_expert_mesh_topology(monkeypatch):
    calls = []

    class _Mesh:
        def __init__(self, name="expert"):
            self.name = name

        def __getitem__(self, name):
            return _Mesh(name)

    def _init_device_mesh(**kwargs):
        calls.append(kwargs)
        return _Mesh()

    class _DistributedInterface:
        current_device = torch.device("cpu")
        strategy = SimpleNamespace(cp_size=1)

        def get_world_size(self, dim):
            return 16

        def get_device_mesh(self, dim):
            return _Mesh("dp")

    monkeypatch.setattr(fsdpturbo_module, "init_device_mesh", _init_device_mesh)
    state = FSDPTurboParallelState()
    state.initialize(_DistributedInterface(), {"ep_size": 8})

    assert calls == [
        {
            "device_type": "cpu",
            "mesh_shape": (1, 2, 8, 1),
            "mesh_dim_names": ("edp", "efsdp", "ep", "expert_cp"),
        }
    ]
    assert state.ep_mesh.name == "ep"
    assert state.efsdp_mesh.name == "efsdp"
    assert state.expert_cp_mesh.name == "expert_cp"


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
