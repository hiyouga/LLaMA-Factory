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
import subprocess
import sys
from pathlib import Path

import pytest

from llamafactory.extras.packages import is_ray_available


@pytest.mark.xfail(reason="CI machines may OOM or hang when heavily loaded.")
@pytest.mark.skipif(not is_ray_available(), reason="Ray is not installed")
@pytest.mark.runs_on(["cuda", "xpu"])
def test_ray_sft_trainer(tmp_path: Path):
    """Test Ray distributed SFT trainer via `USE_RAY=1 llamafactory-cli train config.yaml`."""
    config_yaml = """\
### model
model_name_or_path: Qwen/Qwen3-0.6B
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: identity
dataset_dir: REMOTE:llamafactory/demo_data
template: qwen3_nothink
cutoff_len: 512
max_samples: 10

### output
output_dir: {output_dir}
overwrite_output_dir: true
report_to: none

### ray
ray_num_workers: 2

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 1.0e-4
max_steps: 1
bf16: true
"""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_yaml.format(output_dir=str(output_dir)))

    env = os.environ.copy()
    env["USE_RAY"] = "1"
    env["HF_DATASETS_CACHE"] = str(tmp_path / "hf_datasets")
    env["HF_HOME"] = str(tmp_path / "hf_home")

    result = subprocess.run(
        [sys.executable, "-m", "llamafactory.cli", "train", str(config_file)],
        env=env,
        capture_output=True,
        cwd=str(Path(__file__).parent.parent.parent),
    )

    stderr = result.stderr.decode("utf-8", errors="replace")
    assert result.returncode == 0, f"Ray SFT training failed with return code {result.returncode}\nSTDERR: {stderr}"
