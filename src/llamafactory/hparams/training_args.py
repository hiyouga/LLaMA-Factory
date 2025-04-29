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

import json
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from transformers import Seq2SeqTrainingArguments
from transformers.training_args import _convert_str_dict

from ..extras.misc import use_ray


@dataclass
class RayArguments:
    r"""Arguments pertaining to the Ray training."""

    ray_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "The training results will be saved at `<ray_storage_path>/ray_run_name`."},
    )
    ray_storage_path: str = field(
        default="./saves",
        metadata={"help": "The storage path to save training results to"},
    )
    ray_storage_filesystem: Optional[Literal["s3", "gs", "gcs"]] = field(
        default=None,
        metadata={"help": "The storage filesystem to use. If None specified, local filesystem will be used."},
    )
    ray_num_workers: int = field(
        default=1,
        metadata={"help": "The number of workers for Ray training. Default is 1 worker."},
    )
    resources_per_worker: Union[dict, str] = field(
        default_factory=lambda: {"GPU": 1},
        metadata={"help": "The resources per worker for Ray training. Default is to use 1 GPU per worker."},
    )
    placement_strategy: Literal["SPREAD", "PACK", "STRICT_SPREAD", "STRICT_PACK"] = field(
        default="PACK",
        metadata={"help": "The placement strategy for Ray training. Default is PACK."},
    )
    ray_init_kwargs: Optional[dict] = field(
        default=None,
        metadata={"help": "The arguments to pass to ray.init for Ray training. Default is None."},
    )

    def __post_init__(self):
        self.use_ray = use_ray()
        if isinstance(self.resources_per_worker, str) and self.resources_per_worker.startswith("{"):
            self.resources_per_worker = _convert_str_dict(json.loads(self.resources_per_worker))
        if self.ray_storage_filesystem is not None:
            if self.ray_storage_filesystem not in ["s3", "gs", "gcs"]:
                raise ValueError(
                    f"ray_storage_filesystem must be one of ['s3', 'gs', 'gcs'], got {self.ray_storage_filesystem}"
                )

            import pyarrow.fs as fs

            if self.ray_storage_filesystem == "s3":
                self.ray_storage_filesystem = fs.S3FileSystem()
            elif self.ray_storage_filesystem == "gs" or self.ray_storage_filesystem == "gcs":
                self.ray_storage_filesystem = fs.GcsFileSystem()


@dataclass
class TrainingArguments(RayArguments, Seq2SeqTrainingArguments):
    r"""Arguments pertaining to the trainer."""

    def __post_init__(self):
        Seq2SeqTrainingArguments.__post_init__(self)
        RayArguments.__post_init__(self)
