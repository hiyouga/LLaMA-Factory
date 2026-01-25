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

from ....config.arg_utils import PluginConfig
from ....utils.plugin import BasePlugin
from ....utils.types import HFModel


class DistributedPlugin(BasePlugin):
    def __call__(self, model: HFModel, dist_config: PluginConfig, **kwargs) -> HFModel:
        return super().__call__(model, dist_config, **kwargs)


@DistributedPlugin("fsdp2").register()
def shard_model_fsdp2(model: HFModel, dist_config: PluginConfig) -> HFModel:
    from .fsdp2 import FSDP2Engine

    return FSDP2Engine(dist_config).shard_model(model)


@DistributedPlugin("deepspeed").register()
def shard_model_deepspeed(model: HFModel, dist_config: PluginConfig) -> HFModel:
    return model
