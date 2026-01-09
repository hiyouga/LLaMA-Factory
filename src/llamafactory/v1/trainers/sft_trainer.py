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


from ..accelerator.interface import DistributedInterface
from ..config import InputArgument, get_args
from ..core.base_trainer import BaseTrainer
from ..core.data_engine import DataEngine
from ..core.model_engine import ModelEngine


class SFTTrainer(BaseTrainer):
    pass


def run_sft(args: InputArgument = None):
    model_args, data_args, training_args, _ = get_args(args)
    DistributedInterface(training_args.dist_config)
    data_engine = DataEngine(data_args)
    model_engine = ModelEngine(model_args)
    trainer = SFTTrainer(
        args=training_args,
        model=model_engine.model,
        renderer=model_engine.renderer,
        dataset=data_engine,
    )
    trainer.fit()
