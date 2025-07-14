# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llava/processing_llava.py
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

import inspect
import math
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, BinaryIO, Literal, Optional, TypedDict, Union

import numpy as np
import torch
from transformers.image_utils import get_image_size, is_valid_image, to_numpy_array
from transformers.models.mllama.processing_mllama import (
    convert_sparse_cross_attention_mask_to_dense,
    get_cross_attention_token_mask,
)
from typing_extensions import override

from ..extras.constants import AUDIO_PLACEHOLDER, IGNORE_INDEX, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from ..extras.packages import (
    is_librosa_available,
    is_pillow_available,
    is_pyav_available,
    is_transformers_version_greater_than,
)


if is_librosa_available():
    import librosa


if is_pillow_available():
    from PIL import Image
    from PIL.Image import Image as ImageObject


if is_pyav_available():
    import av


if is_transformers_version_greater_than("4.52.0"):
    from transformers.image_utils import make_flat_list_of_images
    from transformers.video_utils import make_batched_videos
else:
    from transformers.image_utils import make_batched_videos, make_flat_list_of_images


if TYPE_CHECKING:
    from av.stream import Stream
    from numpy.typing import NDArray
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
    from transformers.image_processing_utils import BaseImageProcessor

    class EncodedImage(TypedDict):
        path: Optional[str]
        bytes: Optional[bytes]

    ImageInput = Union[str, bytes, EncodedImage, BinaryIO, ImageObject]
    VideoInput = Union[str, BinaryIO, list[list[ImageInput]]]
    AudioInput = Union[str, BinaryIO, NDArray]

    class MMProcessor(ProcessorMixin):
        patch_size: int
        image_seq_length: int
        num_additional_image_tokens: int
        vision_feature_select_strategy: Literal["default", "full"]

        def _get_number_of_features(self, orig_height: int, orig_width: int, height: int, width: int) -> int:
            pass


def _get_paligemma_token_type_ids(imglens: list[int], seqlens: list[int], processor: "MMProcessor") -> list[list[int]]:
    r"""Get paligemma token type ids for computing loss.

    It is slightly different with the original token type ids where the prompt part is 0.

    Returns:
        batch_token_type_ids: shape (batch_size, seq_length)

    """
    batch_token_type_ids = []
    for imglen, seqlen in zip(imglens, seqlens):
        image_seqlen = imglen * processor.image_seq_length
        batch_token_type_ids.append([0] * image_seqlen + [1] * (seqlen - image_seqlen))

    return batch_token_type_ids


def _get_gemma3_token_type_ids(batch_ids: list[list[int]], processor: "MMProcessor"):
    r"""Get gemma3 token type ids for computing loss.

    Returns:
        batch_token_type_ids: shape (batch_size, seq_length)

    """
    image_token_id: int = getattr(processor, "image_token_id")
    batch_token_type_ids = []
    for token_ids in batch_ids:
        token_ids = np.array(token_ids)
        token_type_ids = np.zeros_like(token_ids)
        token_type_ids[token_ids == image_token_id] = 1
        batch_token_type_ids.append(token_type_ids.tolist())

    return batch_token_type_ids


def _make_batched_images(images: list["ImageObject"], imglens: list[int]) -> list[list["ImageObject"]]:
    r"""Make nested list of images."""
    batch_images = []
    for imglen in imglens:
        batch_images.append(images[:imglen])
        images = images[imglen:]

    return batch_images


def _check_video_is_nested_images(video: "VideoInput") -> bool:
    r"""Check if the video is nested images."""
    return isinstance(video, list) and all(isinstance(frame, (str, BinaryIO, dict)) for frame in video)


@dataclass
class MMPluginMixin:
    image_token: Optional[str]
    video_token: Optional[str]
    audio_token: Optional[str]
    expand_mm_tokens: bool = True

    def _validate_input(
        self,
        processor: Optional["MMProcessor"],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> None:
        r"""Validate if this model accepts the input modalities."""
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        video_processor: BaseImageProcessor = getattr(
            processor, "video_processor", getattr(processor, "image_processor", None)
        )
        feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None)
        if len(images) != 0 and self.image_token is None:
            raise ValueError(
                "This model does not support image input. Please check whether the correct `template` is used."
            )

        if len(videos) != 0 and self.video_token is None:
            raise ValueError(
                "This model does not support video input. Please check whether the correct `template` is used."
            )

        if len(audios) != 0 and self.audio_token is None:
            raise ValueError(
                "This model does not support audio input. Please check whether the correct `template` is used."
            )

        if self.image_token is not None and processor is None:
            raise ValueError("Processor was not found, please check and update your model file.")

        if self.image_token is not None and image_processor is None:
            raise ValueError("Image processor was not found, please check and update your model file.")

        if self.video_token is not None and video_processor is None:
            raise ValueError("Video processor was not found, please check and update your model file.")

        if self.audio_token is not None and feature_extractor is None:
            raise ValueError("Audio feature extractor was not found, please check and update your model file.")

    def _validate_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ):
        r"""Validate if the number of images, videos and audios match the number of placeholders in messages."""
        num_image_tokens, num_video_tokens, num_audio_tokens = 0, 0, 0
        for message in messages:
            num_image_tokens += message["content"].count(IMAGE_PLACEHOLDER)
            num_video_tokens += message["content"].count(VIDEO_PLACEHOLDER)
            num_audio_tokens += message["content"].count(AUDIO_PLACEHOLDER)

        if len(images) != num_image_tokens:
            raise ValueError(
                f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens in {messages}."
            )

        if len(videos) != num_video_tokens:
            raise ValueError(
                f"The number of videos does not match the number of {VIDEO_PLACEHOLDER} tokens in {messages}."
            )

        if len(audios) != num_audio_tokens:
            raise ValueError(
                f"The number of audios does not match the number of {AUDIO_PLACEHOLDER} tokens in {messages}."
            )

    def _preprocess_image(
        self, image: "ImageObject", image_max_pixels: int, image_min_pixels: int, **kwargs
    ) -> "ImageObject":
        r"""Pre-process a single image."""
        if (image.width * image.height) > image_max_pixels:
            resize_factor = math.sqrt(image_max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < image_min_pixels:
            resize_factor = math.sqrt(image_min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _get_video_sample_indices(
        self, video_stream: "Stream", video_fps: float, video_maxlen: int, **kwargs
    ) -> list[int]:
        r"""Compute video sample indices according to fps."""
        total_frames = video_stream.frames
        if total_frames == 0:  # infinite video
            return np.linspace(0, video_maxlen - 1, video_maxlen).astype(np.int32)

        sample_frames = max(1, math.floor(float(video_stream.duration * video_stream.time_base) * video_fps))
        sample_frames = min(total_frames, video_maxlen, sample_frames)
        return np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)

    def _regularize_images(self, images: list["ImageInput"], **kwargs) -> dict[str, list["ImageObject"]]:
        r"""Regularize images to avoid error. Including reading and pre-processing."""
        results = []
        for image in images:
            if isinstance(image, (str, BinaryIO)):
                image = Image.open(image)
            elif isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            elif isinstance(image, dict):
                if image["bytes"] is not None:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    image = Image.open(image["path"])

            if not isinstance(image, ImageObject):
                raise ValueError(f"Expect input is a list of images, but got {type(image)}.")

            results.append(self._preprocess_image(image, **kwargs))

        return {"images": results}

    def _regularize_videos(self, videos: list["VideoInput"], **kwargs) -> dict[str, list[list["ImageObject"]]]:
        r"""Regularizes videos to avoid error. Including reading, resizing and converting."""
        results = []
        for video in videos:
            frames: list[ImageObject] = []
            if _check_video_is_nested_images(video):
                for frame in video:
                    if not is_valid_image(frame) and not isinstance(frame, dict) and not os.path.exists(frame):
                        raise ValueError("Invalid image found in video frames.")
                frames = video
            else:
                container = av.open(video, "r")
                video_stream = next(stream for stream in container.streams if stream.type == "video")
                sample_indices = self._get_video_sample_indices(video_stream, **kwargs)
                container.seek(0)
                for frame_idx, frame in enumerate(container.decode(video_stream)):
                    if frame_idx in sample_indices:
                        frames.append(frame.to_image())

            frames = self._regularize_images(frames, **kwargs)["images"]
            results.append(frames)

        return {"videos": results}

    def _regularize_audios(
        self, audios: list["AudioInput"], sampling_rate: float, **kwargs
    ) -> dict[str, Union[list["NDArray"], list[float]]]:
        r"""Regularizes audios to avoid error. Including reading and resampling."""
        results, sampling_rates = [], []
        for audio in audios:
            if not isinstance(audio, np.ndarray):
                audio, sampling_rate = librosa.load(audio, sr=sampling_rate)

            results.append(audio)
            sampling_rates.append(sampling_rate)

        return {"audios": results, "sampling_rates": sampling_rates}

    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
        imglens: Optional[list[int]] = None,
    ) -> dict[str, "torch.Tensor"]:
        r"""Process visual inputs.

        Returns: (llava and paligemma)
            pixel_values: tensor with shape (B, C, H, W)

        Returns: (qwen2-vl)
            pixel_values: tensor with shape (num_patches, patch_dim)
            image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height
                            where num_patches == torch.prod(image_grid_thw)

        Returns: (mllama)
            pixel_values: tensor with shape
                          (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width)
                          For example, (2, 1, 4, 3, 560, 560).
            aspect_ratio_ids: tensor with shape (batch_size, max_num_images). For example, (2, 1).
            aspect_ratio_mask: tensor with shape (batch_size, max_num_images, max_image_tiles). For example, (2, 1, 4).
            num_tiles: List[List[int]] with shape (batch_size, num_images_in_batch). For example, (2, 1).

        """
        mm_inputs = {}
        if len(images) != 0:
            image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            if imglens is not None:  # if imglens are provided, make batched images
                images = _make_batched_images(images, imglens)

            image_processor_kwargs = {}
            if getattr(processor, "image_do_pan_and_scan", False):  # gemma3 image processor
                image_processor_kwargs.update(
                    {
                        "do_pan_and_scan": True,
                        "pan_and_scan_min_crop_size": 256,
                        "pan_and_scan_max_num_crops": 4,
                        "pan_and_scan_min_ratio_to_activate": 1.2,
                    }
                )

            mm_inputs.update(image_processor(images, return_tensors="pt", **image_processor_kwargs))

        if len(videos) != 0:
            video_processor: BaseImageProcessor = getattr(
                processor, "video_processor", getattr(processor, "image_processor", None)
            )
            videos = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )["videos"]
            if "videos" in inspect.signature(video_processor.preprocess).parameters:  # for qwen2_vl and video_llava
                mm_inputs.update(video_processor(images=None, videos=videos, return_tensors="pt"))
            else:  # for llava_next_video
                mm_inputs.update(video_processor(videos, return_tensors="pt"))

        if len(audios) != 0:
            feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None)
            audios = self._regularize_audios(
                audios,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )["audios"]
            mm_inputs.update(
                feature_extractor(
                    audios,
                    sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
                    return_attention_mask=True,
                    padding="max_length",
                    return_tensors="pt",
                )
            )
            mm_inputs["feature_attention_mask"] = mm_inputs.pop("attention_mask", None)  # prevent conflicts

        return mm_inputs


@dataclass
class BasePlugin(MMPluginMixin):
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        r"""Pre-process input messages before tokenization for VLMs."""
        self._validate_input(processor, images, videos, audios)
        return messages

    def process_token_ids(
        self,
        input_ids: list[int],
        labels: Optional[list[int]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["MMProcessor"],
    ) -> tuple[list[int], Optional[list[int]]]:
        r"""Pre-process token ids after tokenization for VLMs."""
        self._validate_input(processor, images, videos, audios)
        return input_ids, labels

    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        r"""Build batched multimodal inputs for VLMs.

        Arguments:
            images: a list of image inputs, shape (num_images,)
            videos: a list of video inputs, shape (num_videos,)
            audios: a list of audio inputs, shape (num_audios,)
            imglens: number of images in each sample, shape (batch_size,)
            vidlens: number of videos in each sample, shape (batch_size,)
            audlens: number of audios in each sample, shape (batch_size,)
            batch_ids: token ids of input samples, shape (batch_size, seq_len)
            processor: a processor for pre-processing images and videos

        """
        self._validate_input(processor, images, videos, audios)
        return self._get_mm_inputs(images, videos, audios, processor)


@dataclass
class Gemma3Plugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        boi_token: str = getattr(processor, "boi_token")
        full_image_sequence: str = getattr(processor, "full_image_sequence")
        image_str = full_image_sequence if self.expand_mm_tokens else boi_token

        do_pan_and_scan: bool = getattr(processor, "image_do_pan_and_scan", False)
        if do_pan_and_scan:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if do_pan_and_scan:
                    image_placeholder_str = (
                        "Here is the original image {{image}} and here are some crops to help you see better "
                        + " ".join(["{{image}}"] * mm_inputs["num_crops"][0][num_image_tokens])
                    )
                else:
                    image_placeholder_str = "{{image}}"

                content = content.replace(IMAGE_PLACEHOLDER, image_placeholder_str, 1)
                num_image_tokens += 1

            message["content"] = content.replace("{{image}}", image_str)

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs.pop("num_crops", None)
        mm_inputs["token_type_ids"] = _get_gemma3_token_type_ids(batch_ids, processor)
        return mm_inputs


class Gemma3nPlugin(Gemma3Plugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)
        boi_token: str = getattr(processor, "boi_token")
        boa_token: str = getattr(processor, "boa_token")
        full_image_sequence: str = getattr(processor, "full_image_sequence")
        full_audio_sequence: str = getattr(processor, "full_audio_sequence")
        image_str = full_image_sequence if self.expand_mm_tokens else boi_token
        audio_str = full_audio_sequence if self.expand_mm_tokens else boa_token

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, image_str, 1)

            while AUDIO_PLACEHOLDER in content:
                content = content.replace(AUDIO_PLACEHOLDER, audio_str, 1)

            message["content"] = content

        return messages


@dataclass
class InternVLPlugin(BasePlugin):
    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "ProcessorMixin",
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")
        image_processor_kwargs = {}
        if getattr(processor, "crop_to_patches", False):
            image_processor_kwargs.update(
                {
                    "crop_to_patches": True,
                    "max_patches": 12,
                    "min_patches": 1,
                }
            )

        mm_inputs = {}
        image_video_patches = []

        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 1024 * 1024),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]

        if len(videos) != 0:
            videos = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )["videos"]

        if len(images) != 0:
            images = make_flat_list_of_images(images)
            image_inputs = image_processor(images=images, return_tensors="pt", **image_processor_kwargs)
            image_num_patches = image_inputs.pop("num_patches")
            image_pixel_values = image_inputs.pop("pixel_values")
            image_num_patches_indices = np.cumsum(image_num_patches)

        if len(videos) != 0:
            videos = make_batched_videos(videos)
            num_frames_per_video = [len(video) for video in videos]
            patch_indices = np.cumsum(num_frames_per_video)
            image_processor_kwargs["crop_to_patches"] = False
            video_inputs = image_processor(images=videos, return_tensors="pt", **image_processor_kwargs)
            video_num_patches = video_inputs.pop("num_patches")
            video_pixel_values = video_inputs.pop("pixel_values")
            video_num_patches_indices = np.cumsum(video_num_patches)

        # NOT SUPPORT IMAGE VIDEO INTERLEAVED
        if len(images) != 0 and image_pixel_values is not None:
            for i in range(len(images)):
                start_index = image_num_patches_indices[i - 1] if i > 0 else 0
                end_index = image_num_patches_indices[i]
                image_video_patches.append(image_pixel_values[start_index:end_index])

        if len(videos) != 0 and video_pixel_values is not None:
            patch_indices_with_prefix = [0] + list(patch_indices)
            for i in range(len(videos)):
                current_patch_index = patch_indices_with_prefix[i]
                end_patch_index = patch_indices_with_prefix[i + 1]
                start_index = video_num_patches_indices[current_patch_index - 1] if i > 0 else 0
                end_index = video_num_patches_indices[end_patch_index - 1]
                image_video_patches.append(video_pixel_values[start_index:end_index])

        if len(images) != 0 or len(videos) != 0:
            mm_inputs["pixel_values"] = torch.cat(image_video_patches, dim=0)

        if len(images) != 0:
            mm_inputs.update({"image_num_patches": image_num_patches})

        if len(videos) != 0:
            mm_inputs.update({"video_patch_indices": patch_indices})
            mm_inputs.update({"video_num_patches": video_num_patches})

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["ProcessorMixin"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        image_seqlen = getattr(processor, "image_seq_length") if self.expand_mm_tokens else 1
        messages = deepcopy(messages)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)

        image_pixel_patch_list = mm_inputs.get("image_num_patches")  # pathes of images
        video_num_patches = mm_inputs.get("video_num_patches")  # all patches for frames of videos
        video_patch_indices = mm_inputs.get("video_patch_indices")  # num frames of per video

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"<img>{'<IMG_CONTEXT>' * image_seqlen * image_pixel_patch_list[num_image_tokens]}</img>",
                    1,
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                current_patch_index = video_patch_indices[num_video_tokens - 1] if num_video_tokens > 0 else 0
                end_patch_index = video_patch_indices[num_video_tokens]
                num_patches = list(video_num_patches[current_patch_index:end_patch_index])
                video_replaced_prompt = "\n".join(
                    f"Frame{i + 1}: <img>{'<IMG_CONTEXT>' * image_seqlen * num_patches[i]}</img>"
                    for i in range(len(num_patches))
                )
                content = content.replace(VIDEO_PLACEHOLDER, video_replaced_prompt, 1)
                num_video_tokens += 1

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["ProcessorMixin"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs.pop("image_num_patches", None)
        mm_inputs.pop("video_patch_indices", None)
        mm_inputs.pop("video_num_patches", None)
        return mm_inputs


class KimiVLPlugin(BasePlugin):
    @override
    def process_messages(self, messages, images, videos, audios, processor):
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            image_grid_hws = mm_inputs.get("image_grid_hws", [])
        else:
            image_grid_hws = [None] * len(images)

        num_image_tokens = 0
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")
        merge_length = math.prod(image_processor.merge_kernel_size)
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_hws[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"<|media_start|>image<|media_content|>{self.image_token * image_seqlen}<|media_end|>",
                    1,
                )
                num_image_tokens += 1

            message["content"] = content

        return messages


@dataclass
class Llama4Plugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                image_height, image_width = mm_inputs["pixel_values"][0].shape[-2:]
                num_patches_per_chunk = int(
                    (image_height // processor.patch_size)
                    * (image_width // processor.patch_size)
                    // processor.downsample_ratio
                )
                aspect_ratios = mm_inputs.pop("aspect_ratios")

        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            if self.expand_mm_tokens:
                placeholder_count = content.count(IMAGE_PLACEHOLDER)
                prompt_splits = content.split(IMAGE_PLACEHOLDER)
                new_content = []
                for local_image_index, split_part in enumerate(prompt_splits):
                    new_content.append(split_part)
                    if local_image_index < placeholder_count:
                        tokens_for_this_image = processor._prompt_split_image(
                            aspect_ratios[num_image_tokens], num_patches_per_chunk
                        )
                        num_image_tokens += 1
                        new_content.append(tokens_for_this_image)

                content = "".join(new_content)
            else:
                content = content.replace(IMAGE_PLACEHOLDER, self.image_token)

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs.pop("aspect_ratios", None)
        return mm_inputs


@dataclass
class LlavaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0]))
                image_seqlen = (height // processor.patch_size) * (
                    width // processor.patch_size
                ) + processor.num_additional_image_tokens
                if processor.vision_feature_select_strategy == "default":
                    image_seqlen -= 1
        else:
            image_seqlen = 1

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)

            message["content"] = content.replace("{{image}}", self.image_token)

        return messages


@dataclass
class LlavaNextPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                image_sizes = iter(mm_inputs["image_sizes"].tolist())
                height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0][0]))

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    orig_height, orig_width = next(image_sizes)
                    image_seqlen = processor._get_number_of_features(orig_height, orig_width, height, width)
                    if processor.vision_feature_select_strategy == "default":
                        image_seqlen -= 1
                else:
                    image_seqlen = 1

                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)
                num_image_tokens += 1

            message["content"] = content.replace("{{image}}", self.image_token)

        return messages


@dataclass
class LlavaNextVideoPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                image_sizes = iter(mm_inputs["image_sizes"].tolist())
                height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0][0]))

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    orig_height, orig_width = next(image_sizes)
                    image_seqlen = processor._get_number_of_features(orig_height, orig_width, height, width)
                    if processor.vision_feature_select_strategy == "default":
                        image_seqlen -= 1
                else:
                    image_seqlen = 1

                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)

            message["content"] = content.replace("{{image}}", self.image_token)

        if self.expand_mm_tokens:
            if "pixel_values_videos" in mm_inputs:
                one_video = to_numpy_array(mm_inputs.get("pixel_values_videos")[0])
                height, width = get_image_size(one_video[0])
                num_frames = one_video.shape[0]  # frame dim is always after batch dim
                image_seqlen = (height // processor.patch_size) * (width // processor.patch_size)
                video_seqlen = image_seqlen // 4 * num_frames  # divide by 4 needed for avg pooling layer
        else:
            video_seqlen = 1

        for message in messages:
            content = message["content"]
            while VIDEO_PLACEHOLDER in content:
                content = content.replace(VIDEO_PLACEHOLDER, "{{video}}" * video_seqlen, 1)

            message["content"] = content.replace("{{video}}", self.video_token)

        return messages


@dataclass
class MiniCPMVPlugin(BasePlugin):
    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            if "valid_image_nums_ls" in kwargs:
                valid_image_nums_ls = kwargs["valid_image_nums_ls"]
                new_images = []
                idx = 0
                for valid_image_nums in valid_image_nums_ls:
                    new_images.append(images[idx : idx + valid_image_nums])
                    idx += valid_image_nums

                images = new_images

            image_inputs = image_processor(
                images, do_pad=True, max_slice_nums=image_processor.max_slice_nums, return_tensors="pt"
            )
            mm_inputs.update(image_inputs)

        if len(videos) != 0:
            videos = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )["videos"]
            video_inputs = image_processor(videos, do_pad=True, max_slice_nums=2, return_tensors="pt")
            mm_inputs.update(video_inputs)

        if len(audios) != 0:
            audios = self._regularize_audios(
                audios,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )["audios"]
            if "valid_audio_nums_ls" in kwargs:
                valid_audio_nums_ls = kwargs["valid_audio_nums_ls"]
                audios_ls = []
                idx = 0
                for valid_audio_nums in valid_audio_nums_ls:
                    audios_ls.append(audios[idx : idx + valid_audio_nums])
                    idx += valid_audio_nums
            else:
                audios_ls = [audios]

            audio_features, audio_feature_lens, audio_phs = processor.audio_feature_extract(
                audios_ls,
                chunk_input=True,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )
            audio_feature_lens = [torch.tensor(audio_feature_len) for audio_feature_len in audio_feature_lens]
            mm_inputs.update({"audio_features": audio_features, "audio_feature_lens": audio_feature_lens})
            if kwargs.get("ret_phs", False):
                mm_inputs.update({"audio_phs": audio_phs})

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens, num_audio_tokens = 0, 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")
        mm_inputs, audio_inputs = {}, {}
        if len(images) != 0 and len(videos) != 0:
            raise ValueError("MiniCPM-V model does not support input images and videos at the same time.")

        if len(videos) != 0:
            max_slice_nums = 2
            use_image_id = False
            mm_inputs = self._get_mm_inputs([], videos, [], processor)
        else:
            max_slice_nums = image_processor.max_slice_nums
            use_image_id = image_processor.use_image_id

        for i, message in enumerate(messages):
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}", 1)
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                video_seqlen = len(mm_inputs["pixel_values"][num_video_tokens]) if self.expand_mm_tokens else 1
                content = content.replace(VIDEO_PLACEHOLDER, "{{image}}" * video_seqlen, 1)
                num_video_tokens += 1

            while AUDIO_PLACEHOLDER in content:
                content = content.replace(AUDIO_PLACEHOLDER, "{{audio}}", 1)
                num_audio_tokens += 1

            message["content"] = content.replace("{{image}}", "(<image>./</image>)").replace(
                "{{audio}}", "(<audio>./</audio>)"
            )

        if len(images):
            mm_inputs = self._get_mm_inputs(images, [], [], processor)

        if len(audios):
            audio_inputs = self._get_mm_inputs([], [], audios, processor, ret_phs=True)

        if self.expand_mm_tokens and mm_inputs:
            pattern = "(<image>./</image>)"
            image_sizes = mm_inputs["image_sizes"]
            idx = 0
            for index, message in enumerate(messages):
                text = message["content"]
                image_tags = re.findall(pattern, text)
                text_chunks = text.split(pattern)
                final_text = ""
                for i in range(len(image_tags)):
                    final_text = (
                        final_text
                        + text_chunks[i]
                        + image_processor.get_slice_image_placeholder(
                            image_sizes[0][idx], idx, max_slice_nums, use_image_id
                        )
                    )
                    idx += 1

                final_text += text_chunks[-1]
                messages[index]["content"] = final_text

        if self.expand_mm_tokens and audio_inputs:
            pattern = "(<audio>./</audio>)"
            idx = 0
            for index, message in enumerate(messages):
                text = message["content"]
                audio_tags = re.findall(pattern, text)
                text_chunks = text.split(pattern)
                final_text = ""
                for i in range(len(audio_tags)):
                    audio_placeholder = audio_inputs["audio_phs"][0][idx]
                    final_text = final_text + text_chunks[i] + audio_placeholder
                    idx += 1

                final_text += text_chunks[-1]
                messages[index]["content"] = final_text

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        # image bound
        image_bounds_list = []
        valid_image_nums_ls = []
        for i, input_ids in enumerate(batch_ids):
            input_ids_ = torch.tensor(input_ids)
            start_cond = (input_ids_ == processor.tokenizer.im_start_id) | (
                input_ids_ == processor.tokenizer.slice_start_id
            )
            end_cond = (input_ids_ == processor.tokenizer.im_end_id) | (input_ids_ == processor.tokenizer.slice_end_id)
            image_start_tokens = torch.where(start_cond)[0]
            image_start_tokens += 1
            image_end_tokens = torch.where(end_cond)[0]
            valid_image_nums_ls.append(imglens[i])
            image_bounds = torch.hstack(
                [
                    image_start_tokens.unsqueeze(-1),
                    image_end_tokens.unsqueeze(-1),
                ]
            )
            image_bounds_list.append(image_bounds)

        mm_inputs = self._get_mm_inputs(images, videos, [], processor, valid_image_nums_ls=valid_image_nums_ls)
        if "tgt_sizes" not in mm_inputs:
            dummy_data = [torch.empty(0) for _ in range(len(batch_ids))]
            mm_inputs.update({"tgt_sizes": dummy_data, "pixel_values": dummy_data, "image_sizes": dummy_data})

        mm_inputs.update({"image_bound": image_bounds_list})

        if len(audios) > 0:
            # audio bound
            audio_bounds_ls = []
            spk_bounds_ls = []
            valid_audio_nums_ls = []

            for input_ids, audiolen in zip(batch_ids, audlens):
                input_ids_ = torch.tensor(input_ids)
                audio_start_idx = torch.where(input_ids_ == processor.tokenizer.audio_start_id)[0]
                audio_end_idx = torch.where(input_ids_ == processor.tokenizer.audio_end_id)[0]
                assert len(audio_start_idx) == len(audio_end_idx)
                audio_bounds = torch.hstack([(audio_start_idx + 1).unsqueeze(-1), audio_end_idx.unsqueeze(-1)])
                audio_bounds_ls.append(audio_bounds)
                valid_audio_nums_ls.append(audiolen)

                spk_start_idx = torch.where(input_ids_ == processor.tokenizer.spk_start_id)[0]
                spk_end_idx = torch.where(input_ids_ == processor.tokenizer.spk_end_id)[0]
                assert len(spk_start_idx) == len(spk_end_idx)
                spk_bounds = torch.hstack([(spk_start_idx + 1).unsqueeze(-1), spk_end_idx.unsqueeze(-1)])
                spk_bounds_ls.append(spk_bounds)

            audio_inputs = self._get_mm_inputs([], [], audios, processor, valid_audio_nums_ls=valid_audio_nums_ls)
            mm_inputs.update(audio_inputs)
            mm_inputs.update({"audio_bounds": audio_bounds_ls, "spk_bounds": spk_bounds_ls})

        return mm_inputs


@dataclass
class MllamaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            num_image_tokens += content.count(IMAGE_PLACEHOLDER)
            message["content"] = content.replace(IMAGE_PLACEHOLDER, self.image_token)

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor, imglens)
        if mm_inputs:
            num_tiles = mm_inputs.pop("num_tiles")
            image_token_id: int = getattr(processor, "image_token_id")
            max_image_tiles: int = getattr(processor.image_processor, "max_image_tiles")
            cross_attention_token_mask = [
                get_cross_attention_token_mask(input_ids, image_token_id) for input_ids in batch_ids
            ]
            mm_inputs["cross_attention_mask"] = torch.from_numpy(
                convert_sparse_cross_attention_mask_to_dense(
                    cross_attention_token_mask,
                    num_tiles=num_tiles,
                    max_num_tiles=max_image_tiles,
                    length=max(len(input_ids) for input_ids in batch_ids),
                )
            )  # shape: (batch_size, length, max_num_images, max_num_tiles)

        return mm_inputs


@dataclass
class PaliGemmaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, "", 1)
                num_image_tokens += 1

            message["content"] = content

        return messages

    @override
    def process_token_ids(
        self,
        input_ids: list[int],
        labels: Optional[list[int]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["MMProcessor"],
    ) -> tuple[list[int], Optional[list[int]]]:
        self._validate_input(processor, images, videos, audios)
        num_images = len(images)
        image_seqlen = processor.image_seq_length if self.expand_mm_tokens else 0  # skip mm token
        image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        input_ids = [image_token_id] * num_images * image_seqlen + input_ids
        if labels is not None:
            labels = [IGNORE_INDEX] * num_images * image_seqlen + labels

        return input_ids, labels

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        seqlens = [len(input_ids) for input_ids in batch_ids]
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs["token_type_ids"] = _get_paligemma_token_type_ids(imglens, seqlens, processor)
        return mm_inputs


@dataclass
class PixtralPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                # BC for transformers < 4.49.0
                if isinstance(mm_inputs["image_sizes"], list):
                    image_sizes = iter(mm_inputs["image_sizes"][0])
                else:
                    image_sizes = iter(mm_inputs["image_sizes"].tolist())

                image_break_token: str = getattr(processor, "image_break_token")
                image_end_token: str = getattr(processor, "image_end_token")

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    patch_size = processor.patch_size * getattr(processor, "spatial_merge_size", 1)
                    height, width = next(image_sizes)
                    num_height_tokens = height // patch_size
                    num_width_tokens = width // patch_size
                    replace_tokens = [[self.image_token] * num_width_tokens + [image_break_token]] * num_height_tokens
                    replace_tokens = [item for sublist in replace_tokens for item in sublist]  # flatten list
                    replace_tokens[-1] = image_end_token
                    replace_str = "".join(replace_tokens)
                else:
                    replace_str = self.image_token

                content = content.replace(IMAGE_PLACEHOLDER, replace_str, 1)

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        # ref to this commit https://github.com/huggingface/transformers/pull/35122
        # after transformers 4.49.0, the `image_sizes` is mandatory as an input parameter for Pixtral VisionEncoder forwarding.
        # it can be passed into `LlavaConditionalGeneration` as a parameter.
        if not is_transformers_version_greater_than("4.49.0"):
            mm_inputs.pop("image_sizes", None)
        return mm_inputs


@dataclass
class Qwen2AudioPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        bos_token: str = getattr(processor, "audio_bos_token")
        eos_token: str = getattr(processor, "audio_eos_token")
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs([], [], audios, processor)
            if "feature_attention_mask" in mm_inputs:
                audio_lengths = mm_inputs["feature_attention_mask"].sum(-1).tolist()

        for message in messages:
            content = message["content"]
            while AUDIO_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    audio_length = audio_lengths.pop(0)
                    input_length = (audio_length - 1) // 2 + 1
                    audio_seqlen = (input_length - 2) // 2 + 1
                else:
                    audio_seqlen = 1

                content = content.replace(
                    AUDIO_PLACEHOLDER, f"{bos_token}{self.audio_token * audio_seqlen}{eos_token}", 1
                )

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        return self._get_mm_inputs(images, videos, audios, processor)


@dataclass
class Qwen2VLPlugin(BasePlugin):
    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height))

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height))

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height))

        return image

    @override
    def _regularize_videos(
        self, videos: list["VideoInput"], **kwargs
    ) -> dict[str, Union[list[list["ImageObject"]], list[float]]]:
        results, fps_per_video = [], []
        for video in videos:
            frames: list[ImageObject] = []
            if _check_video_is_nested_images(video):
                for frame in video:
                    if not is_valid_image(frame) and not isinstance(frame, dict) and not os.path.exists(frame):
                        raise ValueError("Invalid image found in video frames.")

                frames = video
                fps_per_video.append(kwargs.get("video_fps", 2.0))
            else:
                container = av.open(video, "r")
                video_stream = next(stream for stream in container.streams if stream.type == "video")
                sample_indices = self._get_video_sample_indices(video_stream, **kwargs)
                container.seek(0)
                for frame_idx, frame in enumerate(container.decode(video_stream)):
                    if frame_idx in sample_indices:
                        frames.append(frame.to_image())

                if video_stream.duration is None:
                    fps_per_video.append(kwargs.get("video_fps", 2.0))
                else:
                    fps_per_video.append(len(sample_indices) / float(video_stream.duration * video_stream.time_base))

            if len(frames) % 2 != 0:
                frames.append(frames[-1])

            frames = self._regularize_images(frames, **kwargs)["images"]
            results.append(frames)

        return {"videos": results, "fps_per_video": fps_per_video}

    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
    ) -> dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))

        if len(videos) != 0:
            video_data = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            mm_inputs.update(image_processor(images=None, videos=video_data["videos"], return_tensors="pt"))
            temporal_patch_size: int = getattr(image_processor, "temporal_patch_size", 2)
            if "second_per_grid_ts" in processor.model_input_names:
                mm_inputs["second_per_grid_ts"] = [temporal_patch_size / fps for fps in video_data["fps_per_video"]]

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")

        merge_length: int = getattr(image_processor, "merge_size") ** 2
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
        else:
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER, f"<|vision_start|>{self.image_token * image_seqlen}<|vision_end|>", 1
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                video_seqlen = video_grid_thw[num_video_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    VIDEO_PLACEHOLDER, f"<|vision_start|>{self.video_token * video_seqlen}<|vision_end|>", 1
                )
                num_video_tokens += 1

            message["content"] = content

        return messages


@dataclass
class GLM4VPlugin(Qwen2VLPlugin):
    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
    ) -> dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        video_processor: BaseImageProcessor = getattr(processor, "video_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))

        if len(videos) != 0:
            video_data = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            # prepare video metadata
            video_metadata = [
                {"fps": 2, "duration": len(video), "total_frames": len(video)} for video in video_data["videos"]
            ]
            mm_inputs.update(video_processor(images=None, videos=video_data["videos"], video_metadata=video_metadata))

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")

        merge_length: int = getattr(image_processor, "merge_size") ** 2
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
            num_frames = video_grid_thw[0][0] if len(video_grid_thw) > 0 else 0  # hard code for now
            timestamps = mm_inputs.get("timestamps", [])

            if hasattr(timestamps, "tolist"):
                timestamps = timestamps.tolist()

            if not timestamps:
                timestamps_list = []
            elif isinstance(timestamps[0], list):
                timestamps_list = timestamps[0]
            else:
                timestamps_list = timestamps

            unique_timestamps = timestamps_list.copy()
            selected_timestamps = unique_timestamps[:num_frames]
            while len(selected_timestamps) < num_frames:
                selected_timestamps.append(selected_timestamps[-1] if selected_timestamps else 0)

        else:
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)
            num_frames = 0
            selected_timestamps = [0]

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER, f"<|begin_of_image|>{self.image_token * image_seqlen}<|end_of_image|>", 1
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                video_structure = ""
                for frame_index in range(num_frames):
                    video_seqlen = (
                        video_grid_thw[num_video_tokens][1:].prod() // merge_length if self.expand_mm_tokens else 1
                    )
                    timestamp_sec = selected_timestamps[frame_index]
                    frame_structure = (
                        f"<|begin_of_image|>{self.image_token * video_seqlen}<|end_of_image|>{timestamp_sec}"
                    )
                    video_structure += frame_structure

                content = content.replace(VIDEO_PLACEHOLDER, f"<|begin_of_video|>{video_structure}<|end_of_video|>", 1)
                num_video_tokens += 1

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["ProcessorMixin"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs.pop("timestamps", None)
        return mm_inputs


class Qwen2OmniPlugin(Qwen2VLPlugin):
    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
    ) -> dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))

        if len(videos) != 0:
            video_dict = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            mm_inputs.update(image_processor(images=None, videos=video_dict["videos"], return_tensors="pt"))
            temporal_patch_size: int = getattr(image_processor, "temporal_patch_size", 2)
            mm_inputs["video_second_per_grid"] = torch.tensor(
                [temporal_patch_size / fps for fps in video_dict["fps_per_video"]]
            )

        if len(audios) != 0:
            audios = self._regularize_audios(
                audios,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )["audios"]
            mm_inputs.update(
                feature_extractor(
                    audios,
                    sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
                    return_attention_mask=True,
                    padding="max_length",
                    return_tensors="pt",
                )
            )
            mm_inputs["feature_attention_mask"] = mm_inputs.pop("attention_mask")  # prevent conflicts

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens, num_audio_tokens = 0, 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)

        merge_length = processor.image_processor.merge_size**2
        use_audio_in_video = getattr(processor, "use_audio_in_video", False)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
            if "feature_attention_mask" in mm_inputs:
                input_lengths = (mm_inputs["feature_attention_mask"].sum(-1).numpy() - 1) // 2 + 1
                audio_lengths = (input_lengths - 2) // 2 + 1
        else:
            mm_inputs = {}
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)
            audio_lengths = [None] * len(audios)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER, f"<|vision_bos|>{self.image_token * image_seqlen}<|vision_eos|>", 1
                )
                num_image_tokens += 1

            if (
                use_audio_in_video and len(audios) and len(videos)
            ):  # if use the audio of video # deal video token and audio token togather
                if len(videos) != len(audios):
                    raise ValueError(
                        f"Number of videos ({len(videos)}) must match number of audios ({len(audios)}) when using audio in video."
                    )

                while VIDEO_PLACEHOLDER in content:
                    video_pos = content.find(VIDEO_PLACEHOLDER)
                    audio_pos = content.find(AUDIO_PLACEHOLDER, video_pos)
                    if audio_pos == -1 or audio_pos < video_pos:
                        raise ValueError(
                            f"Each {VIDEO_PLACEHOLDER} must be followed by an {AUDIO_PLACEHOLDER} when using audio in video."
                        )

                    audio_t_index = torch.arange(audio_lengths[num_audio_tokens])
                    video_t_index = (
                        torch.arange(video_grid_thw[num_video_tokens][0])
                        .view(-1, 1, 1)
                        .expand(
                            -1,
                            video_grid_thw[num_video_tokens][1] // image_processor.merge_size,
                            video_grid_thw[num_video_tokens][2] // image_processor.merge_size,
                        )
                        .flatten()
                        * mm_inputs["video_second_per_grid"][num_video_tokens]
                        * 25  # FIXME hardcode of position_id_per_seconds=25
                    ).long()
                    t_ntoken_per_chunk = 50  # FIXME hardcode: [25 * 2]
                    video_chunk_indices = processor.get_chunked_index(video_t_index, t_ntoken_per_chunk)
                    audio_chunk_indices = processor.get_chunked_index(audio_t_index, t_ntoken_per_chunk)
                    placeholder_string = ""
                    placeholder_string += "<|vision_bos|>" + "<|audio_bos|>"
                    for j in range(max(len(video_chunk_indices), len(audio_chunk_indices))):
                        video_chunk_index = video_chunk_indices[j] if j < len(video_chunk_indices) else None
                        audio_chunk_index = audio_chunk_indices[j] if j < len(audio_chunk_indices) else None
                        if video_chunk_index is not None:
                            placeholder_string += self.video_token * (video_chunk_index[1] - video_chunk_index[0])

                        if audio_chunk_index is not None:
                            placeholder_string += self.audio_token * (audio_chunk_index[1] - audio_chunk_index[0])

                    placeholder_string += "<|audio_eos|>" + "<|vision_eos|>"
                    content = content.replace(VIDEO_PLACEHOLDER, placeholder_string, 1)
                    content = content.replace(AUDIO_PLACEHOLDER, "", 1)
                    num_audio_tokens += 1
                    num_video_tokens += 1
            else:
                while AUDIO_PLACEHOLDER in content:
                    audio_seqlen = audio_lengths[num_audio_tokens] if self.expand_mm_tokens else 1
                    content = content.replace(
                        AUDIO_PLACEHOLDER, f"<|audio_bos|>{self.audio_token * audio_seqlen}<|audio_eos|>", 1
                    )
                    num_audio_tokens += 1

                while VIDEO_PLACEHOLDER in content:
                    video_seqlen = (
                        video_grid_thw[num_video_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                    )
                    content = content.replace(
                        VIDEO_PLACEHOLDER, f"<|vision_bos|>{self.video_token * video_seqlen}<|vision_eos|>", 1
                    )
                    num_video_tokens += 1

            message["content"] = content

        return messages


@dataclass
class VideoLlavaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        num_frames = 0
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values_images" in mm_inputs:
                height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values_images"][0]))
                num_frames = 1

            if "pixel_values_videos" in mm_inputs:
                one_video = to_numpy_array(mm_inputs["pixel_values_videos"][0])
                height, width = get_image_size(one_video[0])
                num_frames = one_video.shape[0]  # frame dim is always after batch dim

            if "pixel_values_images" in mm_inputs or "pixel_values_videos" in mm_inputs:
                image_seqlen = (height // processor.patch_size) * (
                    width // processor.patch_size
                ) + processor.num_additional_image_tokens
                video_seqlen = image_seqlen * num_frames
                if processor.vision_feature_select_strategy == "default":
                    image_seqlen -= 1
        else:
            image_seqlen, video_seqlen = 1, 1

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                content = content.replace(VIDEO_PLACEHOLDER, "{{video}}" * video_seqlen, 1)
                num_video_tokens += 1

            content = content.replace("{{image}}", self.image_token)
            message["content"] = content.replace("{{video}}", self.video_token)

        return messages


PLUGINS = {
    "base": BasePlugin,
    "gemma3": Gemma3Plugin,
    "glm4v": GLM4VPlugin,
    "gemma3n": Gemma3nPlugin,
    "intern_vl": InternVLPlugin,
    "kimi_vl": KimiVLPlugin,
    "llama4": Llama4Plugin,
    "llava": LlavaPlugin,
    "llava_next": LlavaNextPlugin,
    "llava_next_video": LlavaNextVideoPlugin,
    "minicpm_v": MiniCPMVPlugin,
    "mllama": MllamaPlugin,
    "paligemma": PaliGemmaPlugin,
    "pixtral": PixtralPlugin,
    "qwen2_audio": Qwen2AudioPlugin,
    "qwen2_omni": Qwen2OmniPlugin,
    "qwen2_vl": Qwen2VLPlugin,
    "video_llava": VideoLlavaPlugin,
}


def register_mm_plugin(name: str, plugin_class: type["BasePlugin"]) -> None:
    r"""Register a multimodal plugin."""
    if name in PLUGINS:
        raise ValueError(f"Multimodal plugin {name} already exists.")

    PLUGINS[name] = plugin_class


def get_mm_plugin(
    name: str,
    image_token: Optional[str] = None,
    video_token: Optional[str] = None,
    audio_token: Optional[str] = None,
) -> "BasePlugin":
    r"""Get plugin for multimodal inputs."""
    if name not in PLUGINS:
        raise ValueError(f"Multimodal plugin `{name}` not found.")

    return PLUGINS[name](image_token, video_token, audio_token)
