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


from ..config import InputArgument, SampleBackend, get_args
from ..core.base_sampler import BaseSampler
from ..core.model_loader import ModelLoader


def run_chat(args: InputArgument = None):
    data_args, model_args, _, sample_args = get_args(args)
    if sample_args.sample_backend != SampleBackend.HF:
        model_args.init_plugin = {"name": "init_on_meta"}

    model_loader = ModelLoader(model_args)
    sampler = BaseSampler(sample_args, model_args, model_loader.model, model_loader.processor)
    if data_args.dataset is not None:
        sampler.batch_infer()
    else:
        sampler.generate()


if __name__ == "__main__":
    run_chat()
