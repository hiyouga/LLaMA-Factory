#!/bin/bash

# Copyright 2025 the PEFTFactory team.
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

export DISABLE_VERSION_CHECK=1 # installed peft library from PR https://github.com/huggingface/peft/pull/2458

# datasets=(mnli qqp qnli sst2 stsb mrpc rte cola)
datasets=(record multirc boolq wic wsc cb copa)
peft_methods=(ia3 prompt-tuning lora lntuning)
models=(gemma-3-1b-it llama-3-8b-instruct mistral-7b-instruct)


for d in ${datasets[@]};
do
    for m in ${models[@]};
    do
        for pm in ${peft_methods[@]};
        do
            saves=(saves/${pm}/${m}/eval_${d}_*)

            EVAL_DIR="${saves[-1]}"

            python scipts/peftfactory/compute_metrics.py ${EVAL_DIR} ${d}
        done
    done
done
