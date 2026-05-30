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
from dataclasses import dataclass, field
from uuid import uuid4

from .arg_utils import BatchingStrategy, PluginConfig, get_plugin_config


@dataclass
class MeshConfig:
    """Device-mesh topology consumed by the accelerator (`DistributedInterface`).

    These are mesh-path (fsdp2) concepts. deepspeed defines its parallelism in its
    own config_file and ignores this mesh, so it runs with the defaults here.
    """

    mp_replicate_size: int = 1
    """Model parallel replicate size."""
    mp_shard_size: int | None = None
    """Model parallel shard size, default to world_size // mp_replicate_size."""
    dp_size: int | None = None
    """Data parallel size, default to world_size // cp_size."""
    cp_size: int = 1
    """Context (sequence) parallel size."""
    dist_timeout: int = 18000
    """Timeout (seconds) for distributed process group initialization."""


@dataclass
class TrainingArguments:
    output_dir: str = field(
        default=os.path.join("outputs", str(uuid4().hex)),
        metadata={"help": "Path to the output directory."},
    )
    micro_batch_size: int = field(
        default=1,
        metadata={"help": "Micro batch size for training."},
    )
    global_batch_size: int | None = field(
        default=None,
        metadata={"help": "Global batch size for training, default to DP size * micro batch size."},
    )
    cutoff_len: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for training."},
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for training."},
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs."},
    )
    max_steps: int | None = field(
        default=None,
        metadata={"help": "Maximum number of training steps. If set, overrides num_train_epochs."},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm for training."},
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bf16 for training."},
    )
    batching_strategy: BatchingStrategy = field(
        default=BatchingStrategy.NORMAL,
        metadata={"help": "Batching strategy for training."},
    )
    batching_workers: int = field(
        default=16,
        metadata={"help": "Number of workers for batching."},
    )
    enable_activation_checkpointing: bool = field(
        default=True,
        metadata={"help": "Enable activation checkpointing for training."},
    )
    dist_config: PluginConfig | None = field(
        default=None,
        metadata={"help": "Distributed backend plugin config (e.g. {name: fsdp2} or {name: deepspeed})."},
    )
    dp_size: int | None = field(
        default=None,
        metadata={"help": "Data parallel size. Mesh-path (fsdp2) topology; default to world_size // cp_size."},
    )
    cp_size: int = field(
        default=1,
        metadata={"help": "Context (sequence) parallel size. Mesh-path topology; deepspeed only supports 1."},
    )
    cp_mode: str = field(
        default="ulysses",
        metadata={"help": "Sequence parallel implementation to use when cp_size > 1."},
    )
    mp_replicate_size: int = field(
        default=1,
        metadata={"help": "Model parallel replicate size. Mesh-path topology."},
    )
    mp_shard_size: int | None = field(
        default=None,
        metadata={
            "help": "Model parallel shard size. Mesh-path topology; default to world_size // mp_replicate_size."
        },
    )
    dist_timeout: int = field(
        default=18000,
        metadata={"help": "Timeout (seconds) for distributed process group initialization."},
    )
    optim_config: PluginConfig | None = field(
        default=None,
        metadata={"help": "Optimizer configuration for training."},
    )
    lr_scheduler_config: PluginConfig | None = field(
        default=None,
        metadata={"help": "Learning rate scheduler configuration for training."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."},
    )
    resume_from_checkpoint: str | None = field(
        default=None,
        metadata={"help": "Path to a checkpoint directory to resume training from, or 'auto' to find the latest."},
    )
    save_steps: int | None = field(
        default=None,
        metadata={"help": "Save a training checkpoint every N global steps."},
    )
    save_epochs: float | None = field(
        default=None,
        metadata={"help": "Save a training checkpoint every N epochs."},
    )
    save_ckpt_as_hf: bool = field(
        default=False,
        metadata={
            "help": "Save intermediate checkpoints in HuggingFace format instead of distributed format. Warning: doubles memory usage."
        },
    )
    save_total_limit: int | None = field(
        default=None,
        metadata={"help": "Maximum number of checkpoints to keep. Oldest checkpoints are deleted."},
    )
    logging_steps: int = field(
        default=1,
        metadata={"help": "Log metrics every N optimizer steps."},
    )

    def __post_init__(self) -> None:
        self.dist_config = get_plugin_config(self.dist_config)
        self.optim_config = get_plugin_config(self.optim_config)
        self.lr_scheduler_config = get_plugin_config(self.lr_scheduler_config)

        # Device-mesh topology consumed by the accelerator (`DistributedInterface`).
        self.mesh_config = MeshConfig(
            mp_replicate_size=self.mp_replicate_size,
            mp_shard_size=self.mp_shard_size,
            dp_size=self.dp_size,
            cp_size=self.cp_size,
            dist_timeout=self.dist_timeout,
        )

        if str(self.batching_strategy) == str(BatchingStrategy.DYNAMIC_BATCHING):
            if self.max_steps is None or self.max_steps <= 0:
                raise ValueError("`dynamic_batching` requires `max_steps` because it is step-driven.")
            if self.save_epochs is not None:
                raise ValueError("`save_epochs` is not supported with `dynamic_batching`; use `save_steps` instead.")


        # Parse the DeepSpeed ZeRO-3 judgment ahead of time so downstream callers (e.g. ModelEngine)
        # only check `is_deepspeed_zero3_enabled()` and never read the config themselves.
        # The deepspeed plugin is optional; guard the import so removing it degrades gracefully.
        try:
            from ..plugins.model_plugins.deepspeed_utils import register_deepspeed_dist_config

            register_deepspeed_dist_config(self.dist_config)
        except ImportError:
            pass
