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

from ....utils.plugin import BasePlugin


class SequenceParallelModelPlugin(BasePlugin):
    def __call__(self, model, sp_config):
        return super().__call__(model, sp_config)


class SequenceParallelLossPlugin(BasePlugin):
    def __call__(self, model, inputs, *args, **kwargs):
        return super().__call__(model, inputs, *args, **kwargs)


@SequenceParallelModelPlugin("ulysses").register()
def apply_sequence_parallel(model, sp_config):
    from .ulysses import apply_sequence_parallel as apply_ulysses_sequence_parallel

    return apply_ulysses_sequence_parallel(model, sp_config)


@SequenceParallelLossPlugin("sequence_parallel_loss").register()
def sequence_parallel_loss(model, model_inputs):
    from .loss import sequence_parallel_loss as compute_sequence_parallel_loss

    return compute_sequence_parallel_loss(model, model_inputs)
