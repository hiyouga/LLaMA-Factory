#!/bin/bash

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