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

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ....config.arg_utils import PluginConfig
from ....utils.plugin import BasePlugin


if TYPE_CHECKING:
    from ....utils.types import HFModel, Processor


class DistributedPlugin(BasePlugin):
    def __call__(self, model: HFModel, dist_config: PluginConfig, **kwargs) -> HFModel:
        return super().__call__(model, dist_config, **kwargs)


@DistributedPlugin("fsdp2").register()
def shard_model_fsdp2(model: HFModel, dist_config: PluginConfig, **kwargs) -> HFModel:
    from .fsdp2 import FSDP2Engine

    return FSDP2Engine(dist_config).shard_model(model)


@DistributedPlugin("fsdp2").register("save_model")
def save_model_fsdp2(model: HFModel, output_dir: str, processor: Processor) -> None:
    from .fsdp2 import save_model

    return save_model(model, output_dir, processor)


@DistributedPlugin("fsdp2").register("save_checkpoint")
def save_checkpoint_fsdp2(model: HFModel, optimizer: torch.optim.Optimizer, ckpt_dir: str, **kwargs) -> None:
    from .fsdp2 import save_checkpoint

    return save_checkpoint(model, optimizer, ckpt_dir, **kwargs)


@DistributedPlugin("fsdp2").register("load_checkpoint")
def load_checkpoint_fsdp2(model: HFModel, optimizer: torch.optim.Optimizer, ckpt_dir: str, **kwargs) -> None:
    from .fsdp2 import load_checkpoint

    return load_checkpoint(model, optimizer, ckpt_dir, **kwargs)


@DistributedPlugin("deepspeed").register()
def shard_model_deepspeed(model: HFModel, dist_config: PluginConfig, **kwargs) -> HFModel:
    from .deepspeed import DeepSpeedEngine

    return DeepSpeedEngine(
        dist_config,
        num_micro_batch=kwargs.get("num_micro_batch"),
        micro_batch_size=kwargs.get("micro_batch_size"),
    ).shard_model(model)


@DistributedPlugin("deepspeed").register("save_model")
def save_model_deepspeed(model: HFModel, output_dir: str, processor: Processor) -> None:
    from .deepspeed import save_model

    return save_model(model, output_dir, processor)


@DistributedPlugin("deepspeed").register("save_checkpoint")
def save_checkpoint_deepspeed(model: HFModel, optimizer: torch.optim.Optimizer, ckpt_dir: str) -> None:
    from .deepspeed import save_checkpoint

    return save_checkpoint(model, optimizer, ckpt_dir)


@DistributedPlugin("deepspeed").register("load_checkpoint")
def load_checkpoint_deepspeed(model: HFModel, optimizer: torch.optim.Optimizer, ckpt_dir: str) -> None:
    from .deepspeed import load_checkpoint

    return load_checkpoint(model, optimizer, ckpt_dir)
