# datasets=(mnli qqp qnli sst2 stsb mrpc rte cola)
# datasets=(mnli)
# peft_methods=(prompt-tuning)
# models=(gemma-3-1b-it)

# datasets=(record multirc boolq wic wsc cb copa)
# peft_methods=(ia3 prompt-tuning lora lntuning)
# models=(gemma-3-1b-it llama-3-8b-instruct mistral-7b-instruct)

datasets=(copa)
peft_methods=(ia3)
models=(gemma-3-1b-it)

export DISABLE_VERSION_CHECK=1

for d in ${datasets[@]};
do
    for m in ${models[@]};
    do
        for pm in ${peft_methods[@]};
        do
            OUTPUT_DIR="saves/${pm}/${m}/train_${d}_`date +%s`"
            DATASET="${d}"
            SEED=123

            export OUTPUT_DIR DATASET SEED

            envsubst < examples/peft/${pm}/${m}/train.yaml > ${pm}_${m}_train_${d}_${TIMESTAMP}.yaml

            llamafactory-cli train ${pm}_${m}_train_${d}_${TIMESTAMP}.yaml
        done
    done
done


