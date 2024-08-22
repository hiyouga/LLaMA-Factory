# Copyright 2024 the LlamaFactory team.
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

r"""
Efficient fine-tuning of large language models.

Level:
  api, webui > chat, eval, train > data, model > hparams > extras

Dependency graph:
  main:
    transformers>=4.41.2,<=4.43.4
    datasets>=2.16.0,<=2.20.0
    accelerate>=0.30.1,<=0.32.0
    peft>=0.11.1,<=0.12.0
    trl>=0.8.6,<=0.9.6
  attention:
    transformers>=4.42.4 (gemma+fa2)
  longlora:
    transformers>=4.41.2,<=4.43.4
  packing:
    transformers>=4.41.2,<=4.43.4
"""

from .cli import VERSION


__version__ = VERSION
