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

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from llamafactory.hparams.model_args import ModelArguments
from llamafactory.hparams.parser import _get_kt_runtime_capacity, _parse_eval_args, _parse_infer_args
from llamafactory.model import adapter as adapter_module
from llamafactory.model.model_utils.checkpointing import _get_gradient_checkpointing_kwargs


def _finetuning_args() -> SimpleNamespace:
    return SimpleNamespace(
        stage="sft",
        finetuning_type="lora",
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        create_new_adapter=False,
        use_dora=False,
    )


class _TrainingArgs:
    def __init__(self, kt_config=None, accelerator_kt_config=None, mirror_config=True):
        self.kt_config = kt_config
        mirrored = kt_config if mirror_config else accelerator_kt_config
        self.accelerator_config = SimpleNamespace(kt_config=mirrored)
        self.gradient_checkpointing = False
        self.gradient_checkpointing_kwargs = None
        self.fsdp_config = {}
        self.calls = []

    def update_kt_config(self, config, *, adapter_name_or_path=None):
        self.calls.append((config, adapter_name_or_path))


@pytest.mark.parametrize(
    ("disable_gpu_checkpointing", "cpu_policy", "expected"),
    [
        (False, None, {"cpu": "recompute", "gpu": "recompute"}),
        (False, "retain", {"cpu": "retain", "gpu": "recompute"}),
        (True, None, {"cpu": "retain", "gpu": "retain"}),
        (True, "retain", {"cpu": "retain", "gpu": "retain"}),
    ],
)
def test_kt_activation_policy_uses_existing_lf_switches(disable_gpu_checkpointing, cpu_policy, expected):
    model_args = ModelArguments(
        model_name_or_path="dummy",
        use_kt=True,
        disable_gradient_checkpointing=disable_gpu_checkpointing,
        kt_cpu_activation=cpu_policy,
    )
    assert model_args.get_kt_activation_policy() == expected


def test_kt_rejects_cpu_recompute_without_gpu_checkpointing():
    model_args = ModelArguments(
        model_name_or_path="dummy",
        use_kt=True,
        disable_gradient_checkpointing=True,
        kt_cpu_activation="recompute",
    )
    with pytest.raises(ValueError, match="requires GPU gradient checkpointing"):
        model_args.get_kt_activation_policy()


def test_apply_kt_config_calls_one_public_transformers_api(tmp_path):
    adapter_root = tmp_path / "adapter"
    adapter_dir = adapter_root / "version-1"
    adapter_dir.mkdir(parents=True)
    model_args = ModelArguments(
        model_name_or_path="dummy",
        adapter_name_or_path=str(adapter_root),
        adapter_folder="version-1",
        use_kt=True,
        kt_cpu_activation="retain",
        kt_weight_path="/weights/experts",
        kt_non_expert_weight_path="/weights/nonexpert",
    )
    training_args = _TrainingArgs({"kt_backend": "auto", "kt_model_max_length": 1152})

    model_args.apply_kt_config(_finetuning_args(), training_args, model_max_length=1024)

    assert len(training_args.calls) == 1
    config, adapter_path = training_args.calls[0]
    assert adapter_path == str(adapter_dir)
    assert config["kt_backend"] == "auto"
    assert config["kt_model_max_length"] == 1152
    assert config["kt_activation_policy"] == {"cpu": "retain", "gpu": "recompute"}
    assert config["kt_lora_rank"] == 8
    assert config["kt_lora_alpha"] == 16
    assert config["kt_lora_dropout"] == 0.05
    assert config["kt_non_expert_weight_path"] == "/weights/nonexpert"
    assert training_args.gradient_checkpointing is False


def test_apply_kt_config_rejects_a_second_owner_for_derived_values():
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True)
    training_args = _TrainingArgs({"kt_lora_rank": 4})
    with pytest.raises(ValueError, match="derived from LLaMA-Factory"):
        model_args.apply_kt_config(_finetuning_args(), training_args, model_max_length=1024)


def test_apply_kt_config_rejects_accelerate_only_configuration_owner():
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True)
    training_args = _TrainingArgs(None, accelerator_kt_config={"kt_backend": "auto"}, mirror_config=False)
    with pytest.raises(ValueError, match="remove `kt_config` from the Accelerate config"):
        model_args.apply_kt_config(_finetuning_args(), training_args, model_max_length=1024)


def test_apply_kt_config_rejects_conflicting_accelerate_mirror():
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True)
    training_args = _TrainingArgs(
        {"kt_backend": "auto"},
        accelerator_kt_config={"kt_backend": "AMXBF16"},
        mirror_config=False,
    )
    with pytest.raises(ValueError, match="cannot define different KT settings"):
        model_args.apply_kt_config(_finetuning_args(), training_args, model_max_length=1024)


def test_apply_kt_config_rejects_accelerate_fsdp_checkpointing(monkeypatch):
    monkeypatch.setenv("FSDP_ACTIVATION_CHECKPOINTING", "true")
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True)

    with pytest.raises(ValueError, match="Disable FSDP activation checkpointing"):
        model_args.apply_kt_config(_finetuning_args(), _TrainingArgs({}), model_max_length=1024)


def test_apply_kt_config_rejects_adapter_folder_symlink_escape(tmp_path):
    adapter_root = tmp_path / "adapter"
    outside = tmp_path / "outside"
    adapter_root.mkdir()
    outside.mkdir()
    (adapter_root / "escaped").symlink_to(outside, target_is_directory=True)
    model_args = ModelArguments(
        model_name_or_path="dummy",
        adapter_name_or_path=str(adapter_root),
        adapter_folder="escaped",
        use_kt=True,
    )

    with pytest.raises(ValueError, match="must stay inside"):
        model_args.apply_kt_config(_finetuning_args(), _TrainingArgs({}), model_max_length=1024)


def test_apply_kt_config_rejects_missing_adapter_subfolder(tmp_path):
    adapter_root = tmp_path / "adapter"
    adapter_root.mkdir()
    model_args = ModelArguments(
        model_name_or_path="dummy",
        adapter_name_or_path=str(adapter_root),
        adapter_folder="missing",
        use_kt=True,
    )

    with pytest.raises(ValueError, match="training requires a local adapter directory"):
        model_args.apply_kt_config(_finetuning_args(), _TrainingArgs({}), model_max_length=1024)


@pytest.mark.parametrize("parse_args", (_parse_infer_args, _parse_eval_args))
def test_inference_parsers_accept_the_training_yaml_kt_config(parse_args):
    arguments = {
        "model_name_or_path": "dummy",
        "use_kt": True,
        "kt_config": {"kt_backend": "AMXBF16"},
    }
    if parse_args is _parse_eval_args:
        arguments["task"] = "dummy"

    parsed = parse_args(arguments)

    assert parsed[0]._kt_inference_config == {"kt_backend": "AMXBF16"}


def test_configure_kt_loading_resolves_local_adapter_subfolder(tmp_path, monkeypatch):
    adapter_root = tmp_path / "adapter"
    adapter_path = adapter_root / "version-1"
    adapter_path.mkdir(parents=True)
    handle = object()
    captured = []
    import transformers.integrations.kt as kt_integration

    monkeypatch.setattr(
        kt_integration, "configure_kt", lambda config: (captured.append(config), handle)[1], raising=False
    )
    model_args = ModelArguments(
        model_name_or_path="dummy",
        adapter_name_or_path=str(adapter_root),
        adapter_folder="version-1",
        use_kt=True,
        kt_cpu_activation="retain",
        kt_weight_path="/weights/experts",
        kt_non_expert_weight_path="/weights/nonexpert",
    )
    model_args._kt_inference_config = {"kt_backend": "AMXBF16", "kt_model_max_length": 1152}

    model_args.configure_kt_loading(_finetuning_args(), model_max_length=1024)

    assert model_args._kt_adapter_artifact_path == str(adapter_path)
    assert model_args._kt_config_handle is handle
    assert captured[0]["kt_backend"] == "AMXBF16"
    assert captured[0]["kt_lora_rank"] == 8
    assert captured[0]["kt_lora_alpha"] == 16
    assert captured[0]["kt_model_max_length"] == 1152
    assert captured[0]["kt_weight_path"] == "/weights/experts"
    assert captured[0]["kt_non_expert_weight_path"] == "/weights/nonexpert"


def test_configure_kt_loading_rejects_adapter_folder_symlink_escape(tmp_path):
    adapter_root = tmp_path / "adapter"
    outside = tmp_path / "outside"
    adapter_root.mkdir()
    outside.mkdir()
    (adapter_root / "escaped").symlink_to(outside, target_is_directory=True)
    model_args = ModelArguments(
        model_name_or_path="dummy",
        adapter_name_or_path=str(adapter_root),
        adapter_folder="escaped",
        use_kt=True,
    )

    with pytest.raises(ValueError, match="must stay inside"):
        model_args.configure_kt_loading(_finetuning_args(), model_max_length=1024)


def test_configure_kt_loading_rejects_nonlocal_adapter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_args = ModelArguments(
        model_name_or_path="dummy",
        adapter_name_or_path="organization/remote-adapter",
        use_kt=True,
    )

    with pytest.raises(ValueError, match="local adapter directory"):
        model_args.configure_kt_loading(_finetuning_args(), model_max_length=1024)


@pytest.mark.parametrize("is_trainable", (False, True))
def test_kt_artifacts_load_after_standard_peft_only_for_inference(is_trainable):
    events = []
    peft_model = torch.nn.Linear(2, 2)
    model_args = ModelArguments(
        model_name_or_path="dummy",
        adapter_name_or_path="adapter",
        use_kt=True,
    )
    model_args._kt_adapter_artifact_path = "/abs/adapter"

    with (
        patch.object(
            adapter_module.PeftModel,
            "from_pretrained",
            side_effect=lambda *_args, **_kwargs: (events.append("standard"), peft_model)[1],
        ),
        patch.object(
            adapter_module,
            "_load_kt_inference_adapter_artifacts",
            side_effect=lambda *_args: events.append("kt"),
        ),
        patch.object(adapter_module, "is_deepspeed_zero3_enabled", return_value=False),
    ):
        adapter_module._setup_lora_tuning(
            SimpleNamespace(),
            torch.nn.Linear(2, 2),
            model_args,
            _finetuning_args(),
            is_trainable=is_trainable,
            cast_trainable_params_to_fp32=False,
        )

    assert events == (["standard"] if is_trainable else ["standard", "kt"])


def test_kt_runtime_capacity_uses_local_physical_token_batch():
    data_args = SimpleNamespace(cutoff_len=1024, packing=True)
    training_args = SimpleNamespace(
        do_train=True,
        do_eval=False,
        do_predict=False,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
    )
    assert _get_kt_runtime_capacity(data_args, training_args, _finetuning_args()) == 2064


def test_checkpointing_uses_kt_public_context_provider(monkeypatch):
    context_fn = object()
    kt_sft = SimpleNamespace(get_activation_checkpoint_context_fn=lambda: context_fn)
    monkeypatch.setitem(sys.modules, "kt_kernel", SimpleNamespace(sft=kt_sft))
    monkeypatch.setitem(sys.modules, "kt_kernel.sft", kt_sft)
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True, kt_cpu_activation="retain")

    kwargs = _get_gradient_checkpointing_kwargs(model_args)

    assert kwargs == {"use_reentrant": False, "context_fn": context_fn}


def test_lf_kt_integration_does_not_own_runtime_or_private_state():
    source_root = Path(__file__).parents[2] / "src" / "llamafactory"
    integration_files = (
        source_root / "hparams" / "model_args.py",
        source_root / "hparams" / "parser.py",
        source_root / "model" / "adapter.py",
        source_root / "model" / "loader.py",
        source_root / "model" / "model_utils" / "checkpointing.py",
        source_root / "train" / "sft" / "trainer.py",
        source_root / "train" / "sft" / "workflow.py",
    )
    production = "\n".join(path.read_text() for path in integration_files)
    for forbidden in (
        "hf_kt_config._kt_config",
        "_kt_adapter_path",
        "torch.distributed.checkpoint",
        "kt_adapter_manifest.json",
        "kt_non_expert_manifest.json",
    ):
        assert forbidden not in production


def test_kt_examples_have_one_configuration_owner():
    repo_root = Path(__file__).parents[2]
    accelerate_dir = repo_root / "examples" / "ktransformers" / "accelerate"
    for path in accelerate_dir.glob("*.yaml"):
        assert "kt_config:" not in path.read_text()

    train_dir = repo_root / "examples" / "ktransformers" / "train_lora"
    for name in ("qwen3_5moe_lora_sft_kt.yaml", "deepseek_v3_int8_lora_sft_kt.yaml"):
        config = (train_dir / name).read_text()
        assert "use_kt: true" in config
        assert "kt_cpu_activation:" in config
        assert "kt_config:" in config
