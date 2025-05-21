# Copyright 2025 the LlamaFactory team.
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

import gc
import os
import json
from typing import Optional, Union, List, Dict, Any, Tuple
import re

import fire
from transformers import Seq2SeqTrainingArguments
from tqdm import tqdm

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras import logging
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer
from vllm_infer import vllm_infer


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

logger = logging.get_logger(__name__)  
def extract_json(texts: Union[List[str] | str], *keys: str) -> List[List[str]]:
    pattern = r'"([^"\\]*(\\.[^"\\]*)*)"'
    escape_map = {
        '\n': '\\n', '\t': '\\t', '\b': '\\b',
        '\f': '\\f', '\r': '\\r', '\\': '\\\\',
        '"': '\\"'
    }

    def escape_string(match):
        s = match.group(1)
        for old, new in escape_map.items():
            s = s.replace(old, new)
        return f'"{s}"'

    if isinstance(texts, str):
        texts = [texts]
    results = {key: [] for key in keys}
    for text in texts:
        st = text.find('[')
        ed = text.rfind(']') + 1
        text = text[st:ed]
        text = re.sub(pattern, escape_string, text)
        try:
            data = json.loads(text)
            for item in data:
                for key in keys:
                    if key in item:
                        results[key].append(item[key])
            return [results[key] for key in keys]
        except json.JSONDecodeError as e:
            return [[] for key in keys]

SYSTEM_PROMPT = """Please act as an impartial evaluator and assess the quality of AI assistant responses to user's question. You will be provided with: 
1. The original user prompt
2. Ground truth information contains information that is directly relevant to the user's question
3. The AI assistant's response
Your task is to conduct a thorough evaluation focusing on two key dimensions:
### 1. CORRECTNESS (Score 1-5)
Assess whether the factual claims in the response align with verified information given in the ground truth:
- 5: All factual information is correct and consistent with ground truth
- 4: Most factual information is correct with only minor inaccuracies that don't affect overall understanding
- 3: Response contains a mix of correct and incorrect information, with significant facts being accurate but some notable errors
- 2: Response contains major factual errors that significantly undermine its reliability
- 1: Key facts are incorrect or the response demonstrates fundamental misunderstanding of the topic
### 2. FAITHFULNESS (Score 1-5)
Evaluate whether the response strictly adheres to information contained in the ground truth without adding unsupported claims or fabricated details.
5: All information in the response are either directly supported by the ground truth or correct based on your knowledge.
4: Most information in the response is either directly supported by the ground truth or verifiably correct based on your knowledge; however, some additional content is included that is inaccurate or ambiguous.
3: A significant portion of the response is supported by the ground truth, but includes some additional content that is either incorrect or unclear.
2: Only some parts of the response are supported by the ground truth. The response contains numerous additions that are incorrect or ambiguous.
1: The response is largely or entirely consist of fabricated or misleading information.
### Evaluation Methodology:
1. Carefully read the User Prompt, Assistant's Response and the Ground Truth.
2. Extract and list all key factual statements from the ground truth.
3. For Correctness: For each fact in the ground truth, assess whether it is Correct, Incorrect, or Missing from the assistant’s response.
4. For Faithfulness: Carefully Investigate the Assistant's Response:
   - Identify all factual claims made in the response
   - For each claim, carefully investigate whether:
        - Correct: It is supported by the ground truth or can be verified.
        - Unclear: It is not directly supported by the ground truth, but can be reasonably inferred from the ground truth
        - Incorrect: It's fabricated, not supported by the ground truth.   
5. Give an overall score based on the correctness and faithfulness of the response.
Please first carefully analyze the correctness and faithfulness of the response.
Assign a label(Correct/Incorrect/Missing) to each <response, fact> pair to better assess the correctness of the response.
Assign a label(Correct/Incorrect/Unclear) to each <response, fact> pair to better assess the faithfulness of the response.
At the end of your analysis, please provide the score in the following format:
[
    {
        "correctness": "3",
        "faithfulness": "3"
    }
]
"""


def vllm_judge(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    reward_model: str = None,
    reward_model_adapter: str = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = None,
    cutoff_len: int = 2048,
    max_samples: Optional[int] = None,
    vllm_config: str = "{}",
    save_name: str = "generated_file.jsonl",
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    seed: Optional[int] = None,
    pipeline_parallel_size: int = 1,
    image_max_pixels: int = 768 * 768,
    image_min_pixels: int = 32 * 32,
    video_fps: float = 2.0,
    video_maxlen: int = 128,
    batch_size: int = 1024,
    prompt: str = None,
):
    r"""Perform batch generation using vLLM engine, which supports tensor parallelism.

    Usage: python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --dataset alpaca_en_demo
    """
    if pipeline_parallel_size > get_device_count():
        raise ValueError("Pipeline parallel size should be smaller than the number of gpus.")
    if reward_model is None:
        reward_model = model_name_or_path
    if len(template) == 2:
        base_template,reward_template = template[0],template[1]
    else:
        base_template,reward_template = template,template
    
    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=base_template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
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
    default_system_prompt = template_obj.default_system    
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate

    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "max_model_len": cutoff_len + max_new_tokens,
        "tensor_parallel_size": (get_device_count() // pipeline_parallel_size) or 1,
        "pipeline_parallel_size": pipeline_parallel_size,
        "disable_log_stats": True,
        "enable_lora": model_args.adapter_name_or_path is not None,
    }
    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"image": 4, "video": 2, "audio": 2}

    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)

    llm = LLM(**engine_args)

    # load datasets
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)
    train_dataset = dataset_module["train_dataset"]

    sampling_params = SamplingParams(
        repetition_penalty=generating_args.repetition_penalty or 1.0,  # repetition_penalty must > 0
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,  # top_p must > 0
        top_k=generating_args.top_k or -1,  # top_k must > 0
        stop_token_ids=template_obj.get_stop_token_ids(tokenizer),
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        seed=seed,
    )
    if model_args.adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])
    else:
        lora_request = None

    # Store all results in these lists
    all_prompts = []
    all_preds = []
    all_labels = []

    # Add batch process to avoid the issue of too many files opened
    for i in tqdm(range(0, len(train_dataset), batch_size), desc="Processing batched inference"):
        vllm_inputs, prompts, labels = [], [], []

        batch = train_dataset[i : min(i + batch_size, len(train_dataset))]

        for j in range(len(batch["input_ids"])):
            if batch["images"][j] is not None:
                image = batch["images"][j]
                multi_modal_data = {
                    "image": template_obj.mm_plugin._regularize_images(
                        image, image_max_pixels=image_max_pixels, image_min_pixels=image_min_pixels
                    )["images"]
                }
            elif batch["videos"][j] is not None:
                video = batch["videos"][j]
                multi_modal_data = {
                    "video": template_obj.mm_plugin._regularize_videos(
                        video,
                        image_max_pixels=image_max_pixels,
                        image_min_pixels=image_min_pixels,
                        video_fps=video_fps,
                        video_maxlen=video_maxlen,
                    )["videos"]
                }
            elif batch["audios"][j] is not None:
                audio = batch["audios"][j]
                audio_data = template_obj.mm_plugin._regularize_audios(
                    audio,
                    sampling_rate=16000,
                )
                multi_modal_data = {"audio": zip(audio_data["audios"], audio_data["sampling_rates"])}
            else:
                multi_modal_data = None

            vllm_inputs.append({"prompt_token_ids": batch["input_ids"][j], "multi_modal_data": multi_modal_data})
            prompts.append(tokenizer.decode(batch["input_ids"][j][len(default_system_prompt):], skip_special_tokens=skip_special_tokens))
            labels.append(
                tokenizer.decode(
                    list(filter(lambda x: x != IGNORE_INDEX, batch["labels"][j])),
                    skip_special_tokens=skip_special_tokens,
                )
            )

        results = llm.generate(vllm_inputs, sampling_params, lora_request=lora_request)

        preds = [result.outputs[0].text for result in results]

        # Accumulate results
        all_prompts.extend(prompts)
        all_preds.extend(preds)
        all_labels.extend(labels)
        del llm
        gc.collect()
    # Prediction End
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("*" * 70)
    print(f"{len(all_prompts)} total generated results have been saved at {save_name}.")
    print("*" * 70)
    
    file_name = save_name.split("/")[-1].split(".")[0]
    prediction_file_name = f"{file_name}_predictions.jsonl"
    dataset_info_path = os.path.join(dataset_dir,"dataset_info.json")
    pred_dataset_name = f"{dataset}_{file_name}"
    if prompt and os.path.isfile(prompt):
        try:
            with open(prompt, "r", encoding="utf-8") as f:
                prompt = f.read()
        except Exception as e:
            raise ValueError(f"Failed to read prompt from file: {e}")
    prompt = prompt or SYSTEM_PROMPT
    with open(os.path.join(dataset_dir,prediction_file_name), "w", encoding="utf-8") as f:
        for text, pred, label in zip(all_prompts, all_preds, all_labels):
            f.write(json.dumps(
                    {
                        "instruction": prompt, 
                        "input": f"Question:\n{text}\nGround truth:\n {label}\nAssistant's Response:\n {pred}", 
                        "output": ""
                    },
                    ensure_ascii=False
                ) + "\n"
            )

    with open(dataset_info_path,"r",encoding="utf-8") as f:
        datas = json.load(f)

    datas[pred_dataset_name] = {
        "file_name":prediction_file_name,
        "columns":{
            "prompt":"instruction",
            "query":"input",
            "response":"output"
        }
    }
    with open(dataset_info_path,"w",encoding='utf-8') as f:
        json.dump(datas,f,indent=2,ensure_ascii=False)
    
    vllm_infer(
        reward_model,
        reward_model_adapter,
        pred_dataset_name,
        dataset_dir,
        reward_template,
        cutoff_len,
        max_samples,
        vllm_config,
        save_name,
        temperature,
        top_p,
        top_k,
        max_new_tokens,
        repetition_penalty,
        skip_special_tokens,
        seed,
        pipeline_parallel_size,
        image_max_pixels,
        image_min_pixels,
        video_fps,
        video_maxlen,
    )
    correctnesses = []
    faithfulnesses = []
    with open(save_name,"r",encoding="utf-8") as f:
        judgement_datas = [json.loads(line) for line in f if line.strip()]
    for i,judgement_data in enumerate(judgement_datas):
        judge = judgement_data["predict"]
        try:
            correctness, faithfulness = extract_json(judge,"correctness","faithfulness")
            judgement_data["judgement"] = {
                "correctness":correctness[0],
                "faithfulness":faithfulness[0]
            }            
            correctnesses.append(correctness[0])
            faithfulnesses.append(faithfulness[0])
        except Exception as e:
            judgement_data["judgement"] = {
                "correctness":None,
                "faithfulness":None
            }    
            correctnesses.append(None)
            faithfulnesses.append(None)
    def to_float(score):
        try:
            return float(score)
        except Exception as e:
            logger.info_rank0(f"Error converting {score} to float: {e}")
            return 0
    none_count = correctnesses.count(None)    
    validated_correctnesses = []
    validated_faithfulnesses = []
    for correctness, faithfulness in zip(correctnesses, faithfulnesses):
        if correctness is None or faithfulness is None:
            continue
        try:
            correctness = to_float(correctness)
            faithfulness = to_float(faithfulness)
            validated_correctnesses.append(correctness)
            validated_faithfulnesses.append(faithfulness)
        except Exception as e:
            continue
        
    if len(validated_correctnesses) == 0 or len(validated_faithfulnesses) == 0:
        avg_correctness = 0
        avg_faithfulness = 0
    else:
        avg_correctness = sum(validated_correctnesses) / len(validated_correctnesses)
        avg_faithfulness = sum(validated_faithfulnesses) / len(validated_faithfulnesses)    
                
    judgement_datas.append({"result":
            {
                "avg_correctness":avg_correctness,
                "avg_faithfulness":avg_faithfulness
            }
        }
    )
    with open(save_name,"w",encoding="utf-8") as f:
        for judgement_data in judgement_datas:
            f.write(json.dumps(judgement_data,ensure_ascii=False)+"\n")        
        
    logger.info_rank0(f"Model: {model_name_or_path}")
    logger.info_rank0(f"Reward Model: {reward_model}")
    logger.info_rank0(f"Adapter: {adapter_name_or_path}")
    logger.info_rank0(f"Average Correctness: {str(avg_correctness)}")
    logger.info_rank0(f"Average Faithfulness: {str(avg_faithfulness)}")
    logger.info_rank0(f"*" * 70)
    logger.info_rank0(f"file saved at {save_name}.")
    logger.info_rank0(f"*" * 70)
if __name__ == "__main__":
    fire.Fire(vllm_judge)
