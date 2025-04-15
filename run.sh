datasets=(mnli)
peft_methods=(ia3)
models=(gemma-3-1b-it)


for d in ${datasets[@]};
do
    for m in ${models[@]};
    do
        for pm in ${peft_methods[@]};
        do
            OUTPUT_DIR="saves/${pm}/${m}/train_${d}_`date +%s`/"
            TRAIN_DATASET="${d}_train"
            EVAL_DATASET="${d}_eval"

            export OUTPUT_DIR TRAIN_DATASET EVAL_DATASET

            envsubst < examples/peft/${pm}/${m}/train.yaml > tmp.yaml
        done
    done
done


llamafactory-cli train tmp.yaml && rm tml.yaml