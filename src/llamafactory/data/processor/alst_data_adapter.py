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

"""Data adapter for Arctic Long Sequence Training (ALST) with DeepSpeed."""

from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from ...extras import logging
from ..data_utils import preprocess_sp_dataset


if TYPE_CHECKING:
    from ...hparams import ModelArguments
    from ...model.model_utils.alst_config import ALSTConfig


logger = logging.get_logger(__name__)


def check_alst_data_requirements() -> bool:
    """Check if ALST data processing requirements are available."""
    try:
        import deepspeed  # noqa: F401
        from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPDataLoaderAdapter  # noqa: F401

        return True
    except ImportError as e:
        logger.warning(f"ALST data requirements not met: {e}")
        return False


class ALSTDataAdapter:
    """Data adapter for Arctic Long Sequence Training with DeepSpeed."""

    def __init__(
        self, model_args: "ModelArguments", alst_config: "ALSTConfig", sp_group: Optional["dist.ProcessGroup"] = None
    ):
        self.model_args = model_args
        self.alst_config = alst_config
        self.sp_group = sp_group
        self.is_available = check_alst_data_requirements()

        if not self.is_available:
            logger.warning("ALST data adapter not available, falling back to manual processing")

    def should_use_alst_data_adapter(self, sequence_length: int) -> bool:
        """Determine if ALST data adapter should be used."""
        result = (
            self.alst_config.enabled and self.is_available and self.sp_group is not None
            # Remove sequence length requirement - let ALST handle all sequences when enabled
        )

        # Debug logging to identify why ALST adapter is not being used
        logger.info_rank0("ALST adapter conditions check:")
        logger.info_rank0(f"  - alst_config.enabled: {self.alst_config.enabled}")
        logger.info_rank0(f"  - is_available: {self.is_available}")
        logger.info_rank0(f"  - sp_group is not None: {self.sp_group is not None}")
        logger.info_rank0(f"  - sp_group: {self.sp_group}")
        logger.info_rank0(f"  - should_use_alst_data_adapter result: {result}")

        return result

    def wrap_dataloader(self, dataloader: DataLoader, sequence_length: Optional[int] = None) -> DataLoader:
        """Wrap DataLoader with UlyssesSPDataLoaderAdapter if appropriate."""
        if not self.should_use_alst_data_adapter(sequence_length or 0):
            logger.info_rank0("ALST not enabled, using standard DataLoader without sequence parallel processing")
            return dataloader  # Return original dataloader without manual SP processing

        import torch.distributed as dist
        from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPDataLoaderAdapter  # noqa: F401

        logger.info_rank0("Wrapping DataLoader with UlyssesSPDataLoaderAdapter")

        # Get sequence parallel parameters from the process group
        if self.sp_group is None:
            raise RuntimeError("Sequence parallel group is None - cannot create UlyssesSPDataLoaderAdapter")

        sp_rank = dist.get_rank(self.sp_group)
        sp_world_size = dist.get_world_size(self.sp_group)
        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")

        logger.info_rank0(f"Creating UlyssesSPDataLoaderAdapter with sp_rank={sp_rank}, sp_world_size={sp_world_size}")

        # Create ALST-enabled dataloader with keyword arguments (like Oumi)
        alst_dataloader = UlyssesSPDataLoaderAdapter(
            dataloader, sp_rank=sp_rank, sp_group=self.sp_group, sp_world_size=sp_world_size, device=device
        )

        logger.info_rank0("Successfully created ALST-enabled DataLoader")
        return alst_dataloader

    def _manual_sequence_parallel_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """Create sequence parallel dataloader using manual processing."""
        if self.model_args.sequence_parallel_size <= 1:
            return dataloader

        # Use existing manual sequence parallel processing
        dataset = dataloader.dataset

        # Create a wrapper dataset that handles sequence parallel processing
        wrapped_dataset = ManualSequenceParallelDataset(
            dataset, self.model_args.sequence_parallel_size, self.model_args.sequence_parallel_mode
        )

        # Create new dataloader with wrapped dataset
        # Use either batch_sampler OR individual batch/shuffle/sampler/drop_last parameters
        if dataloader.batch_sampler is not None:
            manual_dataloader = DataLoader(
                wrapped_dataset,
                batch_sampler=dataloader.batch_sampler,
                num_workers=dataloader.num_workers,
                collate_fn=dataloader.collate_fn,
                pin_memory=dataloader.pin_memory,
                timeout=dataloader.timeout,
                worker_init_fn=getattr(dataloader, "worker_init_fn", None),
            )
        else:
            manual_dataloader = DataLoader(
                wrapped_dataset,
                batch_size=dataloader.batch_size,
                shuffle=getattr(dataloader, "_shuffle", False),
                sampler=dataloader.sampler,
                num_workers=dataloader.num_workers,
                collate_fn=dataloader.collate_fn,
                pin_memory=dataloader.pin_memory,
                drop_last=dataloader.drop_last,
                timeout=dataloader.timeout,
                worker_init_fn=getattr(dataloader, "worker_init_fn", None),
            )

        logger.info_rank0("Created manual sequence parallel DataLoader")
        return manual_dataloader

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess a batch for ALST if needed."""
        if not self.alst_config.enabled:
            return batch

        # Add any ALST-specific batch preprocessing here
        # For now, return the batch as-is since UlyssesSPDataLoaderAdapter handles the processing
        return batch

    def get_effective_batch_size(self, original_batch_size: int) -> int:
        """Get the effective batch size considering sequence parallelism."""
        if not self.alst_config.enabled:
            return original_batch_size

        # With sequence parallelism, the effective batch size per device remains the same
        # but the sequence dimension is split across devices
        return original_batch_size

    def get_sequence_parallel_config(self) -> dict[str, Any]:
        """Get sequence parallel configuration for logging/debugging."""
        return {
            "enabled": self.alst_config.enabled,
            "backend": "deepspeed-alst" if self.is_available else "manual",
            "sequence_parallel_size": self.model_args.sequence_parallel_size,
            "ulysses_degree": self.alst_config.ulysses_degree,
            "sequence_tiling": self.alst_config.sequence_tiling,
        }


class ManualSequenceParallelDataset(Dataset):
    """Dataset wrapper for manual sequence parallel processing."""

    def __init__(self, dataset: Dataset, sequence_parallel_size: int, sequence_parallel_mode: str):
        self.dataset = dataset
        self.sequence_parallel_size = sequence_parallel_size
        self.sequence_parallel_mode = sequence_parallel_mode

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get item with sequence parallel processing applied."""
        item = self.dataset[idx]

        # Apply sequence parallel processing to relevant fields
        processed_item = {}
        for key, value in item.items():
            if isinstance(value, (list, torch.Tensor)) and key.endswith(("input_ids", "labels", "attention_mask")):
                # Apply sequence parallel processing
                if isinstance(value, torch.Tensor):
                    value = value.tolist()

                processed_chunks = preprocess_sp_dataset(
                    value, self.sequence_parallel_size, self.sequence_parallel_mode
                )

                # Get the chunk for current rank
                if dist.is_initialized():
                    rank = dist.get_rank()
                    sp_rank = rank % self.sequence_parallel_size
                    chunk_data = processed_chunks[sp_rank]
                else:
                    # If not distributed, just take the first chunk
                    chunk_data = processed_chunks[0]

                # Convert to tensor with proper dtype and ensure contiguous layout
                if isinstance(chunk_data, (list, tuple)):
                    # Convert list to numpy first for efficiency (addresses performance warning)
                    import numpy as np

                    chunk_array = np.array(chunk_data, dtype=np.int64 if key in ["input_ids", "labels"] else np.int32)
                    tensor = torch.from_numpy(chunk_array).contiguous()

                    # Ensure tensor is dense (not sparse) for distributed operations
                    if tensor.is_sparse:
                        tensor = tensor.to_dense()

                    processed_item[key] = tensor
                else:
                    # Ensure consistent dtype for different keys
                    target_dtype = torch.int64 if key in ["input_ids", "labels"] else torch.int32
                    tensor = torch.tensor(chunk_data, dtype=target_dtype).contiguous()

                    # Ensure tensor is dense (not sparse) for distributed operations
                    if tensor.is_sparse:
                        tensor = tensor.to_dense()

                    processed_item[key] = tensor

                # Debug tensor properties for distributed compatibility (disabled for now)
                # from ...train.debug_utils import DEBUG_TENSORS, debug_tensor_properties
                # if DEBUG_TENSORS:
                #     debug_tensor_properties(processed_item[key], f"processed_{key}")

                # Note: ALST expects CPU tensors from dataset, they get moved to device later
            else:
                processed_item[key] = value

        return processed_item


def create_alst_data_adapter(
    model_args: "ModelArguments", alst_config: "ALSTConfig", sp_group: Optional["dist.ProcessGroup"] = None
) -> ALSTDataAdapter:
    """Create ALST data adapter."""
    adapter = ALSTDataAdapter(model_args, alst_config, sp_group)

    config = adapter.get_sequence_parallel_config()
    logger.info_rank0("Created ALST data adapter:")
    for key, value in config.items():
        logger.info_rank0(f"  - {key}: {value}")

    return adapter


def wrap_dataloader_for_alst(
    dataloader: DataLoader,
    model_args: "ModelArguments",
    alst_config: "ALSTConfig",
    sp_group: Optional["dist.ProcessGroup"] = None,
    sequence_length: Optional[int] = None,
) -> DataLoader:
    """Convenience function to wrap a DataLoader for ALST."""
    adapter = create_alst_data_adapter(model_args, alst_config, sp_group)
    return adapter.wrap_dataloader(dataloader, sequence_length)
