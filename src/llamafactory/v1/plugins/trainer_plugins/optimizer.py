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

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...utils.plugin import BasePlugin


if TYPE_CHECKING:
    from ...config.arg_utils import PluginConfig
    from ...utils.types import HFModel


class OptimizerPlugin(BasePlugin):
    def __call__(self, model: HFModel, optim_config: PluginConfig, **kwargs) -> torch.optim.Optimizer:
        return super().__call__(model, optim_config, **kwargs)


@OptimizerPlugin("adamw").register()
def build_adamw(model: HFModel, optim_config: PluginConfig, **kwargs) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params,
        lr=optim_config.get("lr", kwargs.get("learning_rate", 1e-4)),
        betas=tuple(optim_config.get("betas", (0.9, 0.999))),
        eps=optim_config.get("eps", 1e-8),
        weight_decay=optim_config.get("weight_decay", 0.0),
        foreach=optim_config.get("foreach", None),
        fused=optim_config.get("fused", None),
    )

