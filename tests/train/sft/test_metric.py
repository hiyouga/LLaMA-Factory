# Copyright 2026 the LlamaFactory team.
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

import warnings

import numpy as np
from transformers import EvalPrediction

from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.train.sft.metric import ComputeAccuracy


def test_compute_accuracy_ignores_samples_without_valid_labels():
    eval_preds = EvalPrediction(
        predictions=np.array([[1, 2, 3], [2, 3, 0]]),
        label_ids=np.array(
            [
                [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX],
                [IGNORE_INDEX, 2, 3],
            ]
        ),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = ComputeAccuracy()(eval_preds)

    assert result == {"accuracy": 1.0}


def test_compute_accuracy_returns_empty_result_without_valid_labels():
    eval_preds = EvalPrediction(
        predictions=np.array([[1, 2, 3]]),
        label_ids=np.array([[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]]),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = ComputeAccuracy()(eval_preds)

    assert result == {}
