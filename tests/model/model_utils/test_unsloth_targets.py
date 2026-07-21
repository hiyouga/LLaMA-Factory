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

import importlib.util
import sys
import types
from pathlib import Path


def _load_unsloth_module():
    """Load unsloth.py without importing the full llamafactory package (torch)."""
    extras = types.ModuleType("llamafactory.extras")
    logging = types.ModuleType("llamafactory.extras.logging")

    class _Logger:
        def info_rank0(self, *args, **kwargs):
            return None

        def warning_rank0(self, *args, **kwargs):
            return None

    logging.get_logger = lambda name: _Logger()
    extras.logging = logging
    misc = types.ModuleType("llamafactory.extras.misc")
    misc.get_current_device = lambda: "cpu"
    extras.misc = misc
    sys.modules.setdefault("llamafactory", types.ModuleType("llamafactory"))
    sys.modules["llamafactory.extras"] = extras
    sys.modules["llamafactory.extras.logging"] = logging
    sys.modules["llamafactory.extras.misc"] = misc
    sys.modules.setdefault("llamafactory.hparams", types.ModuleType("llamafactory.hparams"))

    path = Path(__file__).resolve().parents[3] / "src/llamafactory/model/model_utils/unsloth.py"
    # parents: test_unsloth_targets -> model_utils -> model -> tests -> repo root? 
    # __file__ = tests/model/model_utils/test_unsloth_targets.py
    # parents[0]=model_utils, [1]=model, [2]=tests, [3]=repo root. Yes.
    src = path.read_text(encoding="utf-8")
    src = src.replace("from ...extras import logging", "from llamafactory.extras import logging")
    src = src.replace("from ...extras.misc import get_current_device", "from llamafactory.extras.misc import get_current_device")
    ns: dict = {}
    exec(compile(src, str(path), "exec"), ns)
    return ns


def test_leaf_target_modules_collapses_expanded_vlm_paths() -> None:
    leaf = _load_unsloth_module()["_leaf_target_modules"]
    expanded = [
        "model.language_model.layers.0.linear_attn.out_proj",
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.1.self_attn.q_proj",
        "out_proj",
    ]
    assert leaf(expanded) == ["out_proj", "q_proj"]


def test_leaf_target_modules_preserves_plain_names_and_strings() -> None:
    leaf = _load_unsloth_module()["_leaf_target_modules"]
    assert leaf(["q_proj", "v_proj"]) == ["q_proj", "v_proj"]
    assert leaf("all-linear") == "all-linear"
    assert leaf(None) is None
