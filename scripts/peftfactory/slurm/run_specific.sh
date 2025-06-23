# configs=(prompt-tuning_llama-3-8b-instruct_mnli_1744902585 lntuning_gemma-3-1b-it_mnli_1744902583 prompt-tuning_mistral-7b-instruct_mnli_1744902589 lora_llama-3-8b-instruct_mnli_1744902586 lntuning_llama-3-8b-instruct_mnli_1744902587 ia3_mistral-7b-instruct_mnli_1744902588)
configs=(prompt-tuning_llama-3-8b-instruct_train_stsb_1745333591)

for c in ${configs[@]};
do
     sbatch --job-name ${c} -o logs/${c}.out -e logs/${c}.err scipts/peftfactory/slurm/run_single.sh experiments/first_runs/${c}.yaml

     sleep 1
done