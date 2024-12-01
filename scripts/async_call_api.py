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

# pip install langchain langchain_openai

import os
import sys
import json
import asyncio


import fire
from tqdm import tqdm
from dataclasses import dataclass
from aiolimiter import AsyncLimiter
from typing import List
import pandas as pd
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from llamafactory.hparams import get_train_args
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.data.loader import _get_merged_dataset

load_dotenv()


class AsyncLLM:
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        base_url: str = "http://localhost:{}/v1/".format(
            os.environ.get("API_PORT", 8000)
        ),
        api_key: str = "{}".format(os.environ.get("API_KEY", "0")),
        num_per_second: int = 6,
        **kwargs,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.num_per_second = num_per_second

        # 创建限速器，每秒最多发出 5 个请求
        self.limiter = AsyncLimiter(self.num_per_second, 1)

        self.llm = ChatOpenAI(
            model=self.model, base_url=self.base_url, api_key=self.api_key, **kwargs
        )

    async def __call__(self, text):
        # 限速
        async with self.limiter:
            return await self.llm.ainvoke([text])


llm = AsyncLLM(
    base_url="http://localhost:{}/v1/".format(os.environ.get("API_PORT", 8000)),
    api_key="{}".format(os.environ.get("API_KEY", "0")),
    num_per_second=10,
)
llms = [llm]


@dataclass
class AsyncAPICall:
    uid: str = "0"

    @staticmethod
    async def _run_task_with_progress(task, pbar):
        result = await task
        pbar.update(1)
        return result

    @staticmethod
    def async_run(
        llms: List[AsyncLLM],
        data: List[str],
        keyword: str = "",
        output_dir: str = "output",
        chunk_size=500,
    ) -> List[str]:

        async def infer_chunk(llms: List[AsyncLLM], data: List):
            """
            逐块进行推理，为避免处理庞大数据时，程序崩溃导致已推理数据丢失
            """
            results = [llms[i % len(llms)](text) for i, text in enumerate(data)]

            with tqdm(total=len(results)) as pbar:
                results = await asyncio.gather(
                    *[
                        AsyncAPICall._run_task_with_progress(task, pbar)
                        for task in results
                    ]
                )
            return results

        idx = 0
        all_df = []
        file_exist_skip = False
        user_confirm = False

        while idx < len(data):
            file_path = os.path.join(output_dir, "tmp", f"{idx}.csv.temp")

            if os.path.exists(file_path):
                if not user_confirm:
                    while True:
                        user_response = input(
                            f"Find {file_path} file already exists. Do you want to skip them forever?\ny or Y to skip, n or N to rerun to overwrite: "
                        )
                        if user_response.lower() == "y":
                            user_confirm = True
                            file_exist_skip = True
                            break
                        elif user_response.lower() == "n":
                            user_confirm = True
                            file_exist_skip = False
                            break

                if file_exist_skip:
                    tmp_df = pd.read_csv(file_path)
                    all_df.append(tmp_df)
                    idx += chunk_size
                    continue

            tmp_data = data[idx : idx + chunk_size]
            loop = asyncio.get_event_loop()
            tmp_result = loop.run_until_complete(infer_chunk(llms=llms, data=tmp_data))
            tmp_result = [item.content for item in tmp_result]

            tmp_df = pd.DataFrame({"infer": tmp_result})

            if not os.path.exists(p := os.path.dirname(file_path)):
                os.makedirs(p, exist_ok=True)

            tmp_df.to_csv(file_path, index=False)
            all_df.append(tmp_df)
            idx += chunk_size

        all_df = pd.concat(all_df)
        return all_df["infer"]


def async_api_infer(
    model_name_or_path: str = "",
    eval_dataset: str = "",
    template: str = "",
    dataset_dir: str = "data",
    do_predict: bool = True,
    predict_with_generate: bool = True,
    max_samples: int = None,
    output_dir: str = "output",
    chunk_size=50,
):

    if len(sys.argv) == 1:
        model_args, data_args, training_args, finetuning_args, generating_args = (
            get_train_args(
                dict(
                    model_name_or_path=model_name_or_path,
                    dataset_dir=dataset_dir,
                    eval_dataset=eval_dataset,
                    template=template,
                    output_dir=output_dir,
                    do_predict=True,
                    predict_with_generate=True,
                    max_samples=max_samples,
                )
            )
        )
    else:
        model_args, data_args, training_args, finetuning_args, generating_args = (
            get_train_args()
        )

    dataset = _get_merged_dataset(
        data_args.eval_dataset, model_args, data_args, training_args, "sft"
    )

    labels = [item[0]["content"] for item in dataset["_response"]]
    prompts = [item[0]["content"] for item in dataset["_prompt"]]

    infers = AsyncAPICall.async_run(
        llms,
        prompts,
        chunk_size=chunk_size,
        output_dir=training_args.output_dir,
    )

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)

    output_prediction_file = os.path.join(
        training_args.output_dir, "generated_predictions.jsonl"
    )

    with open(output_prediction_file, "w", encoding="utf-8") as writer:
        res: List[str] = []
        for text, pred, label in zip(prompts, infers, labels):
            res.append(
                json.dumps(
                    {"prompt": text, "predict": pred, "label": label},
                    ensure_ascii=False,
                )
            )
        writer.write("\n".join(res))


if __name__ == "__main__":
    fire.Fire(async_api_infer)
