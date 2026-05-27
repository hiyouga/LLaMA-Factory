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


import torch
from transformers import PreTrainedTokenizer
from transformers import set_seed as hf_set_seed

from ..accelerator.interface import DistributedInterface
from .constants import IGNORE_INDEX
from .types import BatchInput, ModelInput, Processor, Tensor


def set_seed(seed: int) -> None:
    """Set seed for reproducibility.

    Args:
        seed: Random seed.
    """
    hf_set_seed(seed)


def is_tokenizer(processor: Processor) -> bool:
    """Check if processor is tokenizer.

    Args:
        processor: Processor.

    Returns:
        Whether processor is tokenizer.
    """
    return not hasattr(processor, "tokenizer")


def get_tokenizer(processor: Processor) -> PreTrainedTokenizer:
    """Get tokenizer from processor.

    Args:
        processor: Processor.

    Returns:
        Tokenizer.
    """
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def _pad_and_truncate(tensor: Tensor, max_seqlen: int, pad_value: int = 0) -> Tensor:
    if tensor.shape[-1] >= max_seqlen:
        return tensor[..., :max_seqlen]

    pad_shape = list(tensor.shape)
    pad_shape[-1] = max_seqlen - tensor.shape[-1]
    pad_tensor = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad_tensor], dim=-1)


_MULTIMODAL_PASSTHROUGH_KEYS = frozenset(
    {
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
        "second_per_grid_ts",
    }
)


def _find_contiguous_blocks(type_ids: list[int], target: int) -> list[tuple[int, int]]:
    """Find contiguous blocks of a target value. Returns list of (start_pos, length)."""
    blocks = []
    i = 0
    while i < len(type_ids):
        if type_ids[i] == target:
            start = i
            while i < len(type_ids) and type_ids[i] == target:
                i += 1
            blocks.append((start, i - start))
        else:
            i += 1
    return blocks


def _align_multimodal_on_truncation(sample: ModelInput, max_length: int) -> ModelInput:
    """Remove orphaned multimodal data when sequence will be truncated.

    When cutoff_len truncates input_ids, images/videos whose placeholder tokens are
    partially cut lose their token<->pixel correspondence. This function:
    1. Determines which images/videos are fully within max_length
    2. Trims pixel_values and grid_thw to keep only complete ones
    3. Zeros out orphaned vision tokens so the model ignores them
    """
    mm_type_ids = sample.get("mm_token_type_ids")
    if mm_type_ids is None:
        return sample

    sample = dict(sample)

    # --- Image alignment ---
    if "image_grid_thw" in sample and "pixel_values" in sample:
        image_blocks = _find_contiguous_blocks(mm_type_ids, target=1)
        image_grid_thw = sample["image_grid_thw"]

        n_complete_images = 0
        for start, length in image_blocks:
            if start + length <= max_length:
                n_complete_images += 1
            else:
                break

        if n_complete_images < len(image_grid_thw):
            keep_patches = sum(int(image_grid_thw[i].prod()) for i in range(n_complete_images))
            sample["pixel_values"] = sample["pixel_values"][:keep_patches]
            sample["image_grid_thw"] = image_grid_thw[:n_complete_images]

            # Zero out orphaned image tokens (partial image within [0, max_length))
            input_ids = list(sample["input_ids"])
            mm_type_ids = list(mm_type_ids)
            labels = list(sample["labels"]) if "labels" in sample else None
            loss_weights = list(sample["loss_weights"]) if "loss_weights" in sample else None

            for block_idx in range(n_complete_images, len(image_blocks)):
                start, length = image_blocks[block_idx]
                for pos in range(start, min(start + length, max_length)):
                    input_ids[pos] = 0
                    mm_type_ids[pos] = 0
                    if labels is not None:
                        labels[pos] = IGNORE_INDEX
                    if loss_weights is not None:
                        loss_weights[pos] = 0.0

            sample["input_ids"] = input_ids
            sample["mm_token_type_ids"] = mm_type_ids
            if labels is not None:
                sample["labels"] = labels
            if loss_weights is not None:
                sample["loss_weights"] = loss_weights

    # --- Video alignment ---
    if "video_grid_thw" in sample and "pixel_values_videos" in sample:
        video_frame_blocks = _find_contiguous_blocks(sample.get("mm_token_type_ids", mm_type_ids), target=2)
        video_grid_thw = sample["video_grid_thw"]

        # Group frames into videos: video_i has T=video_grid_thw[i][0] frames
        n_complete_videos = 0
        frame_idx = 0
        for vid_i in range(len(video_grid_thw)):
            T = int(video_grid_thw[vid_i][0])
            all_frames_in = True
            for f in range(T):
                if frame_idx + f >= len(video_frame_blocks):
                    all_frames_in = False
                    break
                start, length = video_frame_blocks[frame_idx + f]
                if start + length > max_length:
                    all_frames_in = False
                    break
            if all_frames_in:
                n_complete_videos += 1
                frame_idx += T
            else:
                break

        if n_complete_videos < len(video_grid_thw):
            keep_patches = sum(int(video_grid_thw[i].prod()) for i in range(n_complete_videos))
            sample["pixel_values_videos"] = sample["pixel_values_videos"][:keep_patches]
            sample["video_grid_thw"] = video_grid_thw[:n_complete_videos]

            # Zero out orphaned video frame tokens
            complete_frame_count = sum(int(video_grid_thw[i][0]) for i in range(n_complete_videos))
            input_ids = list(sample["input_ids"]) if not isinstance(sample["input_ids"], list) else sample["input_ids"]
            cur_mm_type_ids = sample.get("mm_token_type_ids", mm_type_ids)
            if not isinstance(cur_mm_type_ids, list):
                cur_mm_type_ids = list(cur_mm_type_ids)
            labels = (
                list(sample["labels"])
                if "labels" in sample and not isinstance(sample["labels"], list)
                else sample.get("labels")
            )
            loss_weights = (
                list(sample["loss_weights"])
                if "loss_weights" in sample and not isinstance(sample["loss_weights"], list)
                else sample.get("loss_weights")
            )

            for block_idx in range(complete_frame_count, len(video_frame_blocks)):
                start, length = video_frame_blocks[block_idx]
                for pos in range(start, min(start + length, max_length)):
                    input_ids[pos] = 0
                    cur_mm_type_ids[pos] = 0
                    if labels is not None:
                        labels[pos] = IGNORE_INDEX
                    if loss_weights is not None:
                        loss_weights[pos] = 0.0

            sample["input_ids"] = input_ids
            sample["mm_token_type_ids"] = cur_mm_type_ids
            if labels is not None:
                sample["labels"] = labels
            if loss_weights is not None:
                sample["loss_weights"] = loss_weights

    # Remove empty multimodal fields entirely
    if "image_grid_thw" in sample and len(sample["image_grid_thw"]) == 0:
        del sample["pixel_values"]
        del sample["image_grid_thw"]
    if "video_grid_thw" in sample and len(sample["video_grid_thw"]) == 0:
        del sample["pixel_values_videos"]
        del sample["video_grid_thw"]

    return sample


def pad_and_truncate(samples: list[ModelInput], max_seqlen: int) -> list[BatchInput]:
    max_length = min(max(len(sample["input_ids"]) for sample in samples), max_seqlen)
    padded_samples = []
    for sample in samples:
        # Align multimodal fields before truncation: remove images/videos whose
        # placeholder tokens would be partially cut, preventing pixel<->token mismatch.
        if len(sample["input_ids"]) > max_length and any(k in sample for k in _MULTIMODAL_PASSTHROUGH_KEYS):
            sample = _align_multimodal_on_truncation(sample, max_length)

        padded_sample = {}
        for key, value in sample.items():
            if key in _MULTIMODAL_PASSTHROUGH_KEYS:
                padded_sample[key] = value
                continue

            if "label" in key:
                pad_value = IGNORE_INDEX
            else:
                pad_value = 0

            if not isinstance(value, str):
                padded_sample[key] = _pad_and_truncate(torch.tensor(value), max_length, pad_value)
            else:
                padded_sample[key] = value

        padded_samples.append(padded_sample)

    return padded_samples


def compute_valid_tokens(batches: list[BatchInput]) -> int:
    """Compute valid tokens in batches.

    Args:
        batches: Batches.

    Returns:
        Number of valid tokens.
    """
    device = DistributedInterface().current_device
    return sum(
        (batch["labels"].to(device, non_blocking=True) != IGNORE_INDEX).sum().item()
        for batch in batches
        if "labels" in batch
    )
