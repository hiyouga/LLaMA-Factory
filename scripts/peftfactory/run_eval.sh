# datasets=(mnli qqp qnli sst2 stsb mrpc rte cola)
# peft_methods=(ia3 prompt-tuning lora lntuning)
# models=(gemma-3-1b-it llama-3-8b-instruct mistral-7b-instruct)

datasets=(record multirc boolq wic wsc cb copa)
peft_methods=(ia3 prompt-tuning lora lntuning)
models=(gemma-3-1b-it llama-3-8b-instruct mistral-7b-instruct)


for d in ${datasets[@]};
do
    for m in ${models[@]};
    do
        for pm in ${peft_methods[@]};
        do
            saves=(saves/${pm}/${m}/train_${d}_*)
            echo $saves

            TIMESTAMP=`date +%s`
            OUTPUT_DIR="saves/${pm}/${m}/eval_${d}_${TIMESTAMP}"
            ADAPTER="${saves[-1]}"
            DATASET="${d}_eval"
            SEED=123
            WANDB_PROJECT="peft-factory-eval"

            export OUTPUT_DIR DATASET SEED ADAPTER WANDB_PROJECT

            envsubst < examples/peft/${pm}/${m}/eval.yaml > ${pm}_${m}_eval_${d}.yaml

            llamafactory-cli train ${pm}_${m}_eval_${d}.yaml

            python scipts/peftfactory/compute_metrics.py ${OUTPUT_DIR} ${d}
        done
    done
done
