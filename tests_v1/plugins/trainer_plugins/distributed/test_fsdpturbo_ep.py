from types import SimpleNamespace

import torch

from llamafactory.v1.plugins.trainer_plugins.distributed.fsdp2 import get_transformer_layer_cls
from llamafactory.v1.plugins.trainer_plugins.distributed.fsdpturbo import (
    FSDPTurboEPModelSpec,
    FSDPTurboFSDP2Engine,
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


def test_qwen35_support_is_not_hardcoded_in_model_registry():
    assert FSDPTurboEPModelSpec.get(_Model("qwen3_5_moe")) is None


def test_triton_ops_are_forwarded_from_dist_config(monkeypatch):
    import fsdp_turbo.ops.triton as triton_ops

    class LinearAttention(torch.nn.Module):
        pass

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_attn = LinearAttention()

    model = _Model("parameter_driven")
    model.layers = torch.nn.ModuleList([Layer(), Layer()])
    plans = [
        {
            "name": "fla_chunk_gated_delta_rule",
            "apply_modules": ["layers.{*}.linear_attn"],
            "module_attribute": "chunk_gated_delta_rule",
            "kwargs": {"chunk_size": 64},
        }
    ]
    calls = []

    def fake_apply(received_model, received_plans):
        calls.append((received_model, received_plans))
        return 2

    monkeypatch.setattr(triton_ops, "apply_triton_ops", fake_apply)
    engine = object.__new__(FSDPTurboFSDP2Engine)
    engine.dist_config = {"triton_ops": plans}
    engine.rank = 0

    assert engine._apply_triton_ops(model) == 2
    assert calls == [(model, plans)]


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
