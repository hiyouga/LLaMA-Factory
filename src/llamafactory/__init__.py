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

r"""
Efficient fine-tuning of large language models.

Level:
  api, webui > chat, eval, train > data, model > hparams > extras

Dependency graph:
  main:
    transformers>=4.41.2,<=4.49.0,!=4.46.*,!=4.47.*,!=4.48.0
    datasets>=2.16.0,<=3.2.0
    accelerate>=0.34.0,<=1.2.1
    peft>=0.11.1,<=0.12.0
    trl>=0.8.6,<=0.9.6
  attention:
    transformers>=4.42.4 (gemma+fa2)
  longlora:
    transformers>=4.41.2,<4.48.0
  packing:
    transformers>=4.43.0

Disable version checking: DISABLE_VERSION_CHECK=1
Enable VRAM recording: RECORD_VRAM=1
Force check imports: FORCE_CHECK_IMPORTS=1
Force using torchrun: FORCE_TORCHRUN=1
Set logging verbosity: LLAMAFACTORY_VERBOSITY=WARN
Use modelscope: USE_MODELSCOPE_HUB=1
Use openmind: USE_OPENMIND_HUB=1
"""

from .extras.env import VERSION


__version__ = VERSION
