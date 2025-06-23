#!/bin/bash

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
