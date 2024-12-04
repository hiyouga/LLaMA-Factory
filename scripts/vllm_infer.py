# Copyright 2024 the LlamaFactory team.
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


import json
import os
import sys
from typing import List

import fire
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer


max_tokens = 2048


def vllm_infer(
    model_name_or_path: str = None,
    adapter_name_or_path: str = None,
    dataset_dir: str = "data",
    eval_dataset: str = None,
    template: str = "default",
    max_sample: int = None,
    preprocessing_num_workers: int = 16,
    predict_with_generate: bool = True,
    do_predict: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.7,
    top_k: float = 50,
    output_dir: str = "output",
):

    if len(sys.argv) == 1:
        model_args, data_args, training_args, finetuning_args, generating_args = (
            get_train_args(
                dict(
                    model_name_or_path=model_name_or_path,
                    adapter_name_or_path=adapter_name_or_path,
                    dataset_dir=dataset_dir,
                    eval_dataset=eval_dataset,
                    template=template,
                    max_sample=max_sample,
                    preprocessing_num_workers=preprocessing_num_workers,
                    predict_with_generate=predict_with_generate,
                    do_predict=do_predict,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    output_dir=output_dir,
                )
            )
        )
    else:
        model_args, data_args, training_args, finetuning_args, generating_args = (
            get_train_args()
        )

    tokenizer = load_tokenizer(model_args)["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    eval_dataset = get_dataset(
        template, model_args, data_args, training_args, finetuning_args.stage, tokenizer
    )["eval_dataset"]

    prompts = [item["input_ids"] for item in eval_dataset]
    prompts = tokenizer.batch_decode(prompts, skip_special_tokens=False)

    labels = [
        list(filter(lambda x: x != IGNORE_INDEX, item["labels"]))
        for item in eval_dataset
    ]
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    sampling_params = SamplingParams(
        temperature=generating_args.temperature,
        top_k=generating_args.top_k,
        top_p=generating_args.top_p,
        max_tokens=max_tokens,
    )

    if model_args.adapter_name_or_path:
        if isinstance(model_args.adapter_name_or_path, list):
            lora_path = model_args.adapter_name_or_path[0]
        else:
            lora_path = model_args.adapter_name_or_path

        lora_requests = LoRARequest("lora_adapter_0", 0, lora_path=lora_path)
        enable_lora = True
    else:
        lora_requests = None
        enable_lora = False

    llm = LLM(
        model=model_args.model_name_or_path,
        trust_remote_code=True,
        tokenizer=model_args.model_name_or_path,
        enable_lora=enable_lora,
    )

    outputs = llm.generate(prompts, sampling_params, lora_request=lora_requests)

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)

    output_prediction_file = os.path.join(
        training_args.output_dir, "generated_predictions.jsonl"
    )

    with open(output_prediction_file, "w", encoding="utf-8") as writer:
        res: List[str] = []
        for text, pred, label in zip(prompts, outputs, labels):
            res.append(
                json.dumps(
                    {"prompt": text, "predict": pred.outputs[0].text, "label": label},
                    ensure_ascii=False,
                )
            )
        writer.write("\n".join(res))


if __name__ == "__main__":
    fire.Fire(vllm_infer)
