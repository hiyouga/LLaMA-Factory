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

import fire
from transformers import Seq2SeqTrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import get_device_count
from llamafactory.extras.packages import is_pillow_available, is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


if is_pillow_available():
    from PIL import Image
    from PIL.Image import Image as ImageObject


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest


def vllm_infer(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: int = None,
    vllm_config: str = "{}",
    save_name: str = "generated_predictions.jsonl",
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.0,
):
    r"""
    Performs batch generation using vLLM engine, which supports tensor parallelism.
    Usage: python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --dataset alpaca_en_demo
    """
    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            vllm_config=vllm_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)

    inputs, prompts, labels = [], [], []
    for sample in dataset_module["train_dataset"]:
        if sample["images"]:
            multi_modal_data = {"image": []}
            for image in sample["images"]:
                if not isinstance(image, (str, ImageObject)):
                    raise ValueError(f"Expected image input is a path or PIL.Image, but got {type(image)}.")

                if isinstance(image, str):
                    image = Image.open(image).convert("RGB")

                multi_modal_data["image"].append(image)
        else:
            multi_modal_data = None

        inputs.append({"prompt_token_ids": sample["input_ids"], "multi_modal_data": multi_modal_data})
        prompts.append(tokenizer.decode(sample["input_ids"], skip_special_tokens=False))
        labels.append(
            tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, sample["labels"])), skip_special_tokens=False)
        )

    sampling_params = SamplingParams(
        repetition_penalty=generating_args.repetition_penalty or 1.0,  # repetition_penalty must > 0
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,  # top_p must > 0
        top_k=generating_args.top_k,
        stop_token_ids=[tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids,
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=False,
    )
    if model_args.adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])
    else:
        lora_request = None

    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "tensor_parallel_size": get_device_count() or 1,
        "disable_log_stats": True,
        "enable_lora": model_args.adapter_name_or_path is not None,
    }
    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"image": 4, "video": 2}

    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)

    results = LLM(**engine_args).generate(inputs, sampling_params, lora_request=lora_request)
    preds = [result.outputs[0].text for result in results]
    with open(save_name, "w", encoding="utf-8") as f:
        for text, pred, label in zip(prompts, preds, labels):
            f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

    print("*" * 70)
    print(f"{len(prompts)} generated results have been saved at {save_name}.")
    print("*" * 70)


if __name__ == "__main__":
    fire.Fire(vllm_infer)
