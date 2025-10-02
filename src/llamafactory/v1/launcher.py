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


def run_train():
    raise NotImplementedError("Please use `llamafactory-cli sft` or `llamafactory-cli rm`.")


def run_chat():
    from llamafactory.v1.core.chat_sampler import Sampler

    Sampler().cli()


def run_sft():
    from llamafactory.v1.train.sft import SFTTrainer

    SFTTrainer().run()


if __name__ == "__main__":
    run_train()
