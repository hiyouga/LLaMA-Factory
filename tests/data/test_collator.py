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

import os

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForVision2Seq

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq, prepare_4d_attention_mask
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")


def test_base_collator():
    model_args, data_args, *_ = get_infer_args({"model_name_or_path": TINY_LLAMA3, "template": "default"})
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    data_collator = MultiModalDataCollatorForSeq2Seq(
        template=template,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX,
        **tokenizer_module,
    )
    p = tokenizer_module["tokenizer"].pad_token_id
    q = IGNORE_INDEX
    features = [
        {
            "input_ids": [0, 1, 2, 3, 4, 5],
            "attention_mask": [1, 1, 1, 1, 1, 1],
            "labels": [q, q, 2, 3, 4, 5],
        },
        {
            "input_ids": [6, 7],
            "attention_mask": [1, 1],
            "labels": [q, 7],
        },
    ]
    batch_input = data_collator(features)
    expected_input = {
        "input_ids": [
            [0, 1, 2, 3, 4, 5, p, p],
            [6, 7, p, p, p, p, p, p],
        ],
        "attention_mask": [
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ],
        "labels": [
            [q, q, 2, 3, 4, 5, q, q],
            [q, 7, q, q, q, q, q, q],
        ],
    }
    for k in batch_input.keys():
        assert batch_input[k].eq(torch.tensor(expected_input[k])).all()


def test_multimodal_collator():
    model_args, data_args, *_ = get_infer_args(
        {"model_name_or_path": "Qwen/Qwen2-VL-2B-Instruct", "template": "qwen2_vl"}
    )
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    with torch.device("meta"):
        model = AutoModelForVision2Seq.from_config(config)

    data_collator = MultiModalDataCollatorForSeq2Seq(
        template=template,
        model=model,
        pad_to_multiple_of=4,
        label_pad_token_id=IGNORE_INDEX,
        **tokenizer_module,
    )
    p = tokenizer_module["tokenizer"].pad_token_id
    q = IGNORE_INDEX
    s = tokenizer_module["tokenizer"].convert_tokens_to_ids("<|vision_start|>")
    e = tokenizer_module["tokenizer"].convert_tokens_to_ids("<|vision_end|>")
    m = tokenizer_module["tokenizer"].convert_tokens_to_ids("<|image_pad|>")
    fake_image = Image.new("RGB", (64, 64), (255, 255, 255))

    features = [
        {
            "input_ids": [0, 1, 2, 3],
            "attention_mask": [1, 1, 1, 1],
            "labels": [0, 1, 2, 3],
        },
    ]
    batch_input = data_collator(features)
    expected_input = {
        "input_ids": [
            [0, 1, 2, 3, s, m, m, m, m, e, p, p],
        ],
        "attention_mask": [
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "labels": [
            [0, 1, 2, 3, q, q, q, q, q, q, q, q],
        ],
        "position_ids": [
            [[0, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1]],
            [[0, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1]],
            [[0, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1]],
        ],
        "rope_deltas": [[-8]],
        **tokenizer_module["processor"].image_processor(fake_image),
    }
    assert batch_input.keys() == expected_input.keys()
    for k in batch_input.keys():
        assert batch_input[k].eq(torch.tensor(expected_input[k])).all()


def test_4d_attention_mask():
    o = 0.0
    x = torch.finfo(torch.float16).min
    attention_mask_with_indices = torch.tensor(
        [
            [1, 1, 2, 2, 2, 0],
            [1, 2, 2, 3, 3, 3],
        ]
    )
    attention_mask_computed = prepare_4d_attention_mask(attention_mask_with_indices, torch.float16)
    attention_mask_expected = torch.tensor(
        [
            [
                [
                    [o, x, x, x, x, x],
                    [o, o, x, x, x, x],
                    [x, x, o, x, x, x],
                    [x, x, o, o, x, x],
                    [x, x, o, o, o, x],
                    [x, x, x, x, x, x],
                ]
            ],
            [
                [
                    [o, x, x, x, x, x],
                    [x, o, x, x, x, x],
                    [x, o, o, x, x, x],
                    [x, x, x, o, x, x],
                    [x, x, x, o, o, x],
                    [x, x, x, o, o, o],
                ]
            ],
        ],
        dtype=torch.float16,
    )
    assert list(attention_mask_computed.size()) == [2, 1, 6, 6]
    assert torch.all(attention_mask_computed == attention_mask_expected)


if __name__ == "__main__":
    test_multimodal_collator()
