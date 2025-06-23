
import glob, json

import pandas as pd

models = ["gemma-3-1b-it", "llama-3-8b-instruct", "mistral-7b-instruct"]
methods = ["ia3", "prompt-tuning", "lora", "lntuning"]
# datasets = ["mnli", "qqp", "qnli", "sst2", "stsb", "mrpc", "rte", "cola"]
datasets=["record", "multirc", "boolq", "wic", "wsc", "cb", "copa"]


def get_single_result(results):
    if "macro_f1" in results:
        return results["macro_f1"]
    elif "pearsonr" in results:
        return results["pearsonr"]
    else:
        return results["f1"]


def get_results_from_jsonl(eval_dir):
    results = {}
    with open(f"{eval_dir}/results.jsonl") as json_file:
        for line in json_file:
            results.update(json.loads(line))

    return results

for m in models:
    print(f"Model {m}")

    results = {}
    for pm in methods:
        print(f"Method {pm}")
        results[pm] = {}
        for d in datasets:
            print(f"Dataset {d}")
            glob_res = glob.glob(f"saves/{pm}/{m}/eval_{d}*")

            if not glob_res: 
                continue

            results[pm][d] = get_single_result(get_results_from_jsonl(sorted(glob_res)[-1]))*100

    results_df = pd.DataFrame(results).T
    print(results_df.to_latex(float_format="%.1f", caption="Performance across tasks and tuning methods", label="tab:results"))
            

            