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

import bisect
from typing import TYPE_CHECKING, List, Sequence, Tuple

import av
import numpy as np

from ...extras.packages import is_pillow_available


if is_pillow_available():
    from PIL import Image


if TYPE_CHECKING:
    from av import Container
    from numpy.typing import NDArray
    from PIL.Image import Image as ImageObject
    from transformers import ProcessorMixin
    from transformers.image_processing_utils import BaseImageProcessor


def search_for_fit(numbers: Sequence[int], capacity: int) -> int:
    r"""
    Finds the index of largest number that fits into the knapsack with the given capacity.
    """
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


def greedy_knapsack(numbers: List[int], capacity: int) -> List[List[int]]:
    r"""
    An efficient greedy algorithm with binary search for the knapsack problem.
    """
    numbers.sort()  # sort numbers in ascending order for binary search
    knapsacks = []

    while numbers:
        current_knapsack = []
        remaining_capacity = capacity

        while True:
            index = search_for_fit(numbers, remaining_capacity)
            if index == -1:
                break  # no more numbers fit in this knapsack

            remaining_capacity -= numbers[index]  # update the remaining capacity
            current_knapsack.append(numbers.pop(index))  # add the number to knapsack

        knapsacks.append(current_knapsack)

    return knapsacks


def get_pixel_values(images: Sequence["ImageObject"], processor: "ProcessorMixin", image_keys: "list[str]" = ["pixel_values"]) -> "dict":
    r"""
    Processes visual inputs. (currently only supports a single image)
    """
    image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
    if len(images) == 0:
        image = Image.new("RGB", (100, 100), (255, 255, 255))
        inputs = image_processor([image], return_tensors="pt")
        return {k: inputs[k] for k in image_keys}
    else:
        inputs = image_processor(images, return_tensors="pt")
        return {k: inputs[k] for k in image_keys}


def preprocess_video(
    video: "str"
) -> "NDArray":
    container = av.open(video)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    clip = read_video_pyav(container, indices)
    return clip


def get_pixel_values_videos(
    videos: Sequence["str"], processor: "ProcessorMixin", video_keys: "list[str]" = ["pixel_values_videos"]
) -> "dict":
    image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
    video_processor = getattr(processor, "video_processor", None)
    clips = []
    for video in videos:
        clip = preprocess_video(video)
        clips.append(clip)
    if len(clips) == 0:
        clips = clips[0]
    if video_processor is not None:
        inputs = video_processor(clips, return_tensors="pt")
        return {k: inputs[k] for k in video_keys}
    else:
        inputs = image_processor(videos=clips, padding=True, return_tensors="pt", images=None)
        return {k: inputs[k] for k in video_keys}


def read_video_pyav(container: "Container", indices: "NDArray"):
    """
    Decode the video with PyAV decoder.

    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def get_paligemma_token_type_ids(input_len: int, processor: "ProcessorMixin") -> List[int]:
    r"""
    Gets paligemma token type ids for computing loss.
    """
    image_seq_length = getattr(processor, "image_seq_length")
    return [0] * image_seq_length + [1] * (input_len - image_seq_length)


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> Tuple[int, int]:
    r"""
    Computes the real sequence length after truncation by the cutoff_len.
    """
    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len
    else:  # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len
