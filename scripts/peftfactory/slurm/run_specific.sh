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

# configs=(prompt-tuning_llama-3-8b-instruct_mnli_1744902585 lntuning_gemma-3-1b-it_mnli_1744902583 prompt-tuning_mistral-7b-instruct_mnli_1744902589 lora_llama-3-8b-instruct_mnli_1744902586 lntuning_llama-3-8b-instruct_mnli_1744902587 ia3_mistral-7b-instruct_mnli_1744902588)
configs=(prompt-tuning_llama-3-8b-instruct_train_stsb_1745333591)

for c in ${configs[@]};
do
     sbatch --job-name ${c} -o logs/${c}.out -e logs/${c}.err scipts/peftfactory/slurm/run_single.sh experiments/first_runs/${c}.yaml

     sleep 1
done