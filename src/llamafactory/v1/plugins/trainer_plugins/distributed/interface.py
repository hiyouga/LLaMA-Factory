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

"""Interface for distributed backend plugins.

A distributed backend is an ordinary plugin family: ``dist_config`` selects one backend
via ``name`` and carries only that backend's private params. Mesh topology
(``dp_size``/``cp_size``/``mp_*``) is NOT a backend concern — it lives as flat
``TrainingArguments`` fields and is consumed by the accelerator to build the device mesh.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

from ....utils.plugin import BasePlugin
from .base import BaseDistributed


if TYPE_CHECKING:
    import torch

    from ....config.arg_utils import PluginConfig
    from ....utils.types import HFModel, Processor


@dataclass
class FSDP2Params:
    name: Literal["fsdp2"] = "fsdp2"
    reshard_after_forward: bool = True
    offload_params: bool = False
    pin_memory: bool = True
    dcp_path: str | None = None


@dataclass
class DeepSpeedParams:
    name: Literal["deepspeed"] = "deepspeed"
    config_file: str = ""

    def __post_init__(self) -> None:
        if not self.config_file:
            raise ValueError("DeepSpeedParams.config_file is required.")


class DistributedPlugin(BasePlugin):
    """Plugin family for distributed training backends (ordinary name + params)."""


@DistributedPlugin("fsdp2").register(params=FSDP2Params)
class FSDP2Distributed(BaseDistributed):
    @staticmethod
    def shard_model(model: HFModel, dist_config: PluginConfig | FSDP2Params, **kwargs) -> HFModel:
        params = DistributedPlugin.parse_params("fsdp2", dist_config)
        from .fsdp2 import FSDP2Engine

        return FSDP2Engine(asdict(params), bf16=bool(kwargs.get("bf16"))).shard_model(model)

    @staticmethod
    def save_model(model: HFModel, output_dir: str, processor: Processor) -> None:
        from .fsdp2 import save_model

        return save_model(model, output_dir, processor)

    @staticmethod
    def save_checkpoint(model: HFModel, optimizer: torch.optim.Optimizer, ckpt_dir: str, **kwargs) -> None:
        from .fsdp2 import save_checkpoint

        return save_checkpoint(model, optimizer, ckpt_dir, **kwargs)

    @staticmethod
    def load_checkpoint(model: HFModel, optimizer: torch.optim.Optimizer, ckpt_dir: str, **kwargs) -> None:
        from .fsdp2 import load_checkpoint

        return load_checkpoint(model, optimizer, ckpt_dir, **kwargs)


@DistributedPlugin("deepspeed").register(params=DeepSpeedParams)
class DeepSpeedDistributed(BaseDistributed):
    @staticmethod
    def shard_model(model: HFModel, dist_config: PluginConfig | DeepSpeedParams, **kwargs) -> object:
        params = DistributedPlugin.parse_params("deepspeed", dist_config)
        from .deepspeed import DeepSpeedEngine

        return DeepSpeedEngine(
            asdict(params),
            num_micro_batch=kwargs.get("num_micro_batch"),
            micro_batch_size=kwargs.get("micro_batch_size"),
        ).shard_model(model)

    @staticmethod
    def save_model(model: HFModel, output_dir: str, processor: Processor) -> None:
        from .deepspeed import save_model

        return save_model(model, output_dir, processor)

    @staticmethod
    def save_checkpoint(model: HFModel, optimizer: torch.optim.Optimizer, ckpt_dir: str, **kwargs) -> None:
        from .deepspeed import save_checkpoint

        return save_checkpoint(model, optimizer, ckpt_dir, **kwargs)

    @staticmethod
    def load_checkpoint(model: HFModel, optimizer: torch.optim.Optimizer, ckpt_dir: str, **kwargs) -> None:
        from .deepspeed import load_checkpoint

        return load_checkpoint(model, optimizer, ckpt_dir, **kwargs)
