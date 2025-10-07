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


from ..config.parser import get_args
from ..core.base_trainer import BaseTrainer
from ..core.data_engine import DataEngine
from ..core.model_engine import ModelEngine


class SFTTrainer(BaseTrainer):
    pass


def run_sft():
    model_args, data_args, training_args, _ = get_args()
    model_engine = ModelEngine(model_args)
    data_engine = DataEngine(data_args)
    model = model_engine.get_model()
    processor = model_engine.get_processor()
    data_loader = data_engine.get_data_loader(processor)
    trainer = SFTTrainer(training_args, model, processor, data_loader)
    trainer.fit()
