from vllm import LLM, SamplingParams
import torch
import json
import argparse
from load_data import load_MMLU,load_bbh,load_mMMLU,load_bbh_mc,load_mSafe,load_mCOPA
from enginee_utils import return_preprocess_function
import torch
from ..data import get_template_and_fix_tokenizer


settings_map = {}



test_datasets  = {
    'bbh' : load_bbh,
    'mmlu' : load_MMLU,
    "m_mmlu" : load_mMMLU,
    'bbh_mc' : load_bbh_mc,
    'm_safe':load_mSafe,
    'xcopa' : load_mCOPA
} 

parser = argparse.ArgumentParser(description='Run a model with specified settings and test data.')
parser.add_argument('--setting', type=str, required=True, help='The setting key to use.')
parser.add_argument('--test_data', type=str, required=True, help='The test data key to use.')
parser.add_argument('--preprocess', type=str, required=False, default='', help='The preprocess steps to apply.')
parser.add_argument('--template', type=str, required=False, default='NanGPT', help='The preprocess steps to apply.')

args = parser.parse_args()
setting = args.setting
test_data = args.test_data
preprocess = args.preprocess
template = args.template


# Check if the provided setting and test_data are valid
if setting not in settings_map:
    raise ValueError(f"Invalid setting. Available options are: {list(settings_map.keys())}")
if test_data not in test_datasets:
    raise ValueError(f"Invalid test_data. Available options are: {list(test_datasets.keys())}")

template = get_template_and_fix_tokenizer(tokenizer, setemplate)

process_func = return_preprocess_function(preprocess)
model_path = settings_map[setting]

load_data_func = test_datasets[test_data]
data = load_data_func(process_func,template)


sampling_params = SamplingParams(n=1,temperature=0.0,max_tokens=1024, stop=["<|assistant_end|>","\n\nQuestion:","\n\nQ:"],skip_special_tokens=False)


gpu_count = torch.cuda.device_count()
gpu_count = gpu_count - (gpu_count % 4)
llm = LLM(model=model_path,dtype=torch.bfloat16,tensor_parallel_size=gpu_count,gpu_memory_utilization=0.9)

input_prompt = [i['prompted'] for i in data]

result_generator = self.model.generate(
    prompt=None, sampling_params=sampling_params, request_id=request_id, prompt_token_ids=prompt_ids
)

generations = llm.generate(input_prompt,sampling_params)
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

f = open("/mnt/data/shesj/EvalOut/{}/{}-{}-{}.json".format(test_data,setting,preprocess,template),'w')
json.dump(data,f,indent=2,ensure_ascii=False)
