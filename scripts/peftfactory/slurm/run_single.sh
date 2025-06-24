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

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -o logs/peft-factory-stdout.%J.out
#SBATCH -e logs/peft-factory-stderr.%J.out
#SBATCH --time=2-00:00
#SBATCH --account=p904-24-3

eval "$(conda shell.bash hook)"
conda activate peft-factory
module load libsndfile

export HF_HOME="/projects/${PROJECT}/cache"
export DISABLE_VERSION_CHECK=1 # installed peft library from PR https://github.com/huggingface/peft/pull/2458

llamafactory-cli train $1