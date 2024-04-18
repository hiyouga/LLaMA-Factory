import torch
import json
import argparse
from ..data import get_template_and_fix_tokenizer
from ..model import load_tokenizer
from ..hparams import get_eval_args
from vllm import LLM, SamplingParams
from typing import Any, Dict, List, Optional
'''

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ../../src/evaluate.py \
    --model_name_or_path /mnt/data/shesj/Trained/CommonAlign/DPO/QwenDPOSFT \
    --temperature 0.5 \
    --max_new_tokens 2048 \
    --template qwen \
    --finetuning_type full \
    --task mMTbench \
    --n_shot 0 \
    --n_iter 10 \
    --save_dir /mnt/data/shesj/EvalOut/mMTbench
'''


# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('--model_path', type=str, default="/mnt/data/shesj/PLM/MetaMath-7B-V1.0")
# parser.add_argument('--testset',  type=str, default="/mnt/data/shesj/Data/RL4CoTData/gsm8k_test.json")
# parser.add_argument('--iter',  type=int, default=1)
# parser.add_argument('--temp',  type=float, default=0.3)

class Evaluator:
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        model_args, data_args, eval_args, finetuning_args = get_eval_args(args)
        print(model_args)
        print(data_args)
        print(eval_args)
        print(finetuning_args)

        print("===================== Starting =====================" )

        task_data = {
            "mMTbench" : "/mnt/data/shesj/Data/RL4CoTData/Benchmark/SimiGen/mMTbenchSixLang.json"
        }

        data_file = task_data[eval_args.task]

        f = open(data_file,'r')
        data = json.load(f)

        input_prompt = [i['instruction'].split("### Instruction:\n")[1].split("\n\n### Response:")[0] for i in data]

        tokenizer = load_tokenizer(model_args)
        tokenizer.padding_side = "left"
        template = get_template_and_fix_tokenizer(tokenizer, data_args.template)

        sampling_params = SamplingParams(n=eval_args.n_iter,temperature=eval_args.temperature,max_tokens=eval_args.max_new_tokens)
        llm = LLM(model=model_args.model_name_or_path,dtype=torch.bfloat16,tensor_parallel_size=8)


        input_prompt_ids = [template.encode_oneturn(tokenizer=tokenizer, messages=[{"role": "user", "content": query}] + [{"role": "assistant", "content": ""}]) for query in input_prompt]

        generations = llm.generate(prompt=None, sampling_params=sampling_params, prompt_token_ids=input_prompt_ids)

        generated_ = []
        for output in generations:
            prompt = output.prompt
            generate_text = [o.text for o in output.outputs]
            generated_.append(generate_text)

        assert len(data) == len(generated_)
        for i,g in zip(data,generated_):
            i['answers'] = []
            for _ in g:
                i['answers'].append({"generated":_})

        f = open(eval_args.save_dir + "/" + eval_args.model_name + "_" + eval_args.task + ".json",'w')
        json.dump(data,f,indent=2,ensure_ascii=False)
