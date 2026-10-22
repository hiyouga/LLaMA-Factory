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

from llamafactory.webui import common as webui_common
from llamafactory.webui.control import get_checkpoint_dir, list_checkpoints


def test_list_checkpoints_finds_nested_and_top_level(tmp_path, monkeypatch):
    r"""The checkpoint dropdown must list both top-level and nested (intermediate) checkpoints (see #9766).

    A training run that finished normally saves its final adapter directly under the run dir
    (`<run_dir>/adapter_model.safetensors`), while a run that was interrupted before completion only
    has its intermediate `save_steps` checkpoints nested one level deeper as
    `<run_dir>/checkpoint-<step>/adapter_model.safetensors`. Both must show up in the dropdown.
    """
    monkeypatch.setattr(webui_common, "DEFAULT_SAVE_DIR", str(tmp_path))

    save_dir = tmp_path / "mymodel" / "lora"
    (save_dir / "train_completed").mkdir(parents=True)
    (save_dir / "train_completed" / "adapter_model.safetensors").touch()

    (save_dir / "train_interrupted" / "checkpoint-100").mkdir(parents=True)
    (save_dir / "train_interrupted" / "checkpoint-100" / "adapter_model.safetensors").touch()
    (save_dir / "train_interrupted" / "checkpoint-200").mkdir(parents=True)
    (save_dir / "train_interrupted" / "checkpoint-200" / "adapter_model.safetensors").touch()

    result = list_checkpoints("mymodel", "lora")
    choices = [value for _, value in result.choices]
    assert set(choices) == {
        "train_completed",
        os.path.join("train_interrupted", "checkpoint-100"),
        os.path.join("train_interrupted", "checkpoint-200"),
    }


def test_list_checkpoints_no_save_dir_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(webui_common, "DEFAULT_SAVE_DIR", str(tmp_path))
    result = list_checkpoints("nonexistent_model", "lora")
    assert result.choices == []


def test_get_checkpoint_dir_resolves_nested_checkpoint(tmp_path, monkeypatch):
    r"""A nested checkpoint picked from the dropdown must resolve under the save dir (see #9766)."""
    monkeypatch.setattr(webui_common, "DEFAULT_SAVE_DIR", str(tmp_path))
    checkpoint_dir = tmp_path / "mymodel" / "lora" / "train_interrupted" / "checkpoint-100"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "adapter_model.safetensors").touch()

    nested = os.path.join("train_interrupted", "checkpoint-100")
    assert get_checkpoint_dir("mymodel", "lora", nested) == str(checkpoint_dir)


def test_get_save_dir_still_bypasses_for_absolute_custom_path(monkeypatch):
    r"""A user-typed absolute custom path (see #4292) must still bypass the save dir entirely."""
    monkeypatch.setattr(webui_common, "DEFAULT_SAVE_DIR", "saves")
    absolute_path = os.path.abspath(os.path.join(os.sep, "abs", "custom", "path"))
    assert webui_common.get_save_dir("mymodel", "lora", absolute_path) == absolute_path


def test_get_save_dir_keeps_relative_custom_path_against_cwd(monkeypatch):
    r"""A user-typed relative output_dir such as my/custom/dir keeps resolving against the cwd, as it always has."""
    monkeypatch.setattr(webui_common, "DEFAULT_SAVE_DIR", "saves")
    custom_path = os.path.join("my", "custom", "dir")
    assert webui_common.get_save_dir("mymodel", "lora", custom_path) == custom_path


def test_get_checkpoint_dir_keeps_typed_relative_path_against_cwd(tmp_path, monkeypatch):
    r"""A relative path typed into the checkpoint dropdown keeps resolving against the cwd unless it names a checkpoint."""
    monkeypatch.setattr(webui_common, "DEFAULT_SAVE_DIR", str(tmp_path))
    (tmp_path / "mymodel" / "lora" / "train_run" / "notes").mkdir(parents=True)

    typed = os.path.join("my", "custom", "adapter")
    assert get_checkpoint_dir("mymodel", "lora", typed) == typed

    # exists under the save dir, but holds no checkpoint, so it is not a dropdown value
    not_a_checkpoint = os.path.join("train_run", "notes")
    assert get_checkpoint_dir("mymodel", "lora", not_a_checkpoint) == not_a_checkpoint


def test_get_checkpoint_dir_still_bypasses_for_absolute_path(tmp_path, monkeypatch):
    r"""An absolute path typed into the checkpoint dropdown still bypasses the save dir."""
    monkeypatch.setattr(webui_common, "DEFAULT_SAVE_DIR", str(tmp_path))
    absolute_path = os.path.abspath(os.path.join(os.sep, "abs", "custom", "path"))
    assert get_checkpoint_dir("mymodel", "lora", absolute_path) == absolute_path


def test_get_checkpoint_dir_resolves_every_listed_choice(tmp_path, monkeypatch):
    r"""Every value list_checkpoints offers must resolve to a checkpoint that exists."""
    monkeypatch.setattr(webui_common, "DEFAULT_SAVE_DIR", str(tmp_path))

    save_dir = tmp_path / "mymodel" / "lora"
    (save_dir / "train_completed").mkdir(parents=True)
    (save_dir / "train_completed" / "adapter_model.safetensors").touch()
    (save_dir / "train_interrupted" / "checkpoint-100").mkdir(parents=True)
    (save_dir / "train_interrupted" / "checkpoint-100" / "adapter_model.safetensors").touch()

    choices = [value for _, value in list_checkpoints("mymodel", "lora").choices]
    assert len(choices) == 2
    for value in choices:
        resolved = get_checkpoint_dir("mymodel", "lora", value)
        assert os.path.isfile(os.path.join(resolved, "adapter_model.safetensors"))


def test_list_checkpoints_sorts_by_step_number(tmp_path, monkeypatch):
    r"""checkpoint-200 must come before checkpoint-1000, which a lexicographic sort gets wrong."""
    monkeypatch.setattr(webui_common, "DEFAULT_SAVE_DIR", str(tmp_path))

    run_dir = tmp_path / "mymodel" / "lora" / "train_run"
    for step in (100, 200, 1000):
        (run_dir / f"checkpoint-{step}").mkdir(parents=True)
        (run_dir / f"checkpoint-{step}" / "adapter_model.safetensors").touch()

    result = list_checkpoints("mymodel", "lora")
    assert [value for _, value in result.choices] == [
        os.path.join("train_run", "checkpoint-100"),
        os.path.join("train_run", "checkpoint-200"),
        os.path.join("train_run", "checkpoint-1000"),
    ]


def test_list_checkpoints_skips_the_save_dir_itself(tmp_path, monkeypatch):
    r"""A checkpoint sitting in the save dir itself must not surface as a "." dropdown entry."""
    monkeypatch.setattr(webui_common, "DEFAULT_SAVE_DIR", str(tmp_path))

    save_dir = tmp_path / "mymodel" / "lora"
    save_dir.mkdir(parents=True)
    (save_dir / "adapter_model.safetensors").touch()

    result = list_checkpoints("mymodel", "lora")
    assert [value for _, value in result.choices] == []


def test_list_checkpoints_ignores_unrelated_dirs(tmp_path, monkeypatch):
    r"""Logging and export dirs under the save dir hold no checkpoint files and must not be listed."""
    monkeypatch.setattr(webui_common, "DEFAULT_SAVE_DIR", str(tmp_path))

    save_dir = tmp_path / "mymodel" / "lora"
    (save_dir / "runs" / "Jan01_00-00-00").mkdir(parents=True)
    (save_dir / "runs" / "Jan01_00-00-00" / "events.out.tfevents.1").touch()
    (save_dir / "wandb" / "latest-run").mkdir(parents=True)
    (save_dir / "train_run").mkdir(parents=True)
    (save_dir / "train_run" / "adapter_model.safetensors").touch()

    result = list_checkpoints("mymodel", "lora")
    assert [value for _, value in result.choices] == ["train_run"]


def test_list_checkpoints_lists_intermediate_saves_of_a_finished_run(tmp_path, monkeypatch):
    r"""A finished run keeps its intermediate checkpoints selectable alongside its final save."""
    monkeypatch.setattr(webui_common, "DEFAULT_SAVE_DIR", str(tmp_path))

    run_dir = tmp_path / "mymodel" / "lora" / "train_run"
    run_dir.mkdir(parents=True)
    (run_dir / "adapter_model.safetensors").touch()
    (run_dir / "checkpoint-100").mkdir()
    (run_dir / "checkpoint-100" / "adapter_model.safetensors").touch()

    result = list_checkpoints("mymodel", "lora")
    assert [value for _, value in result.choices] == [
        "train_run",
        os.path.join("train_run", "checkpoint-100"),
    ]
