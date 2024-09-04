from copy import deepcopy
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np

from ..extras.constants import IGNORE_INDEX, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from ..extras.packages import is_pillow_available, is_pyav_available


if is_pillow_available():
    from PIL import Image
    from PIL.Image import Image as ImageObject


if is_pyav_available():
    import av


if TYPE_CHECKING:
    import torch
    from numpy.typing import NDArray
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.image_processing_utils import BaseImageProcessor

    class EncodedImage(TypedDict):
        path: Optional[str]
        bytes: Optional[bytes]

    ImageInput = Union[str, EncodedImage, ImageObject]
    VideoInput = str


def _regularize_images(images: Sequence["ImageInput"], processor: "ProcessorMixin") -> List["ImageObject"]:
    r"""
    Regularizes images to avoid error. Including reading, resizing and converting.
    """
    image_resolution: int = getattr(processor, "image_resolution", 512)
    results = []
    for image in images:
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, dict):
            if image["bytes"] is not None:
                image = Image.open(BytesIO(image["bytes"]))
            else:
                image = Image.open(image["path"])

        if not isinstance(image, ImageObject):
            raise ValueError("Expect input is a list of Images, but got {}.".format(type(image)))

        if max(image.width, image.height) > image_resolution:
            factor = image_resolution / max(image.width, image.height)
            image = image.resize((int(image.width * factor), int(image.height * factor)))

        if image.mode != "RGB":
            image = image.convert("RGB")

        results.append(image)

    return results


def _regularize_videos(videos: Sequence["VideoInput"], processor: "ProcessorMixin") -> List["NDArray"]:
    r"""
    Regularizes videos to avoid error. Including reading, resizing and converting.
    """
    video_fps: float = getattr(processor, "video_fps", 1.0)
    video_factor: int = getattr(processor, "video_factor", 1)
    results = []
    for video in videos:
        container = av.open(video, "r")
        video_stream = next(stream for stream in container.streams if stream.type == "video")
        total_frames = video_stream.frames
        sample_frames = float(video_stream.duration * video_stream.time_base) * video_fps
        sample_frames = round(sample_frames / video_factor) * video_factor  # for qwen2_vl
        sample_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)
        frames: List["ImageObject"] = []
        container.seek(0)
        for frame_idx, frame in enumerate(container.decode(video_stream)):
            if frame_idx in sample_indices:
                frames.append(frame.to_image())

        frames = _regularize_images(frames, processor)
        results.append(frames)

    return results


def _get_mm_inputs(
    images: Sequence["ImageInput"],
    videos: Sequence["VideoInput"],
    processor: "ProcessorMixin",
) -> Dict[str, "torch.Tensor"]:
    r"""
    Processes visual inputs.

    Returns: (llava and paligemma)
        pixel_values: tensor with shape (B, C, H, W)

    Returns: (qwen2-vl)
        pixel_values: tensor with shape (num_patches, patch_dim)
        image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height

    It holds num_patches == torch.prod(image_grid_thw)
    """
    image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
    input_dict = {"images": None, "videos": None}
    if len(images) != 0:
        images = _regularize_images(images, processor)
        input_dict["images"] = images

    if len(videos) != 0:
        videos = _regularize_videos(videos, processor)
        input_dict["videos"] = videos

    if input_dict["images"] is not None or input_dict["videos"] is not None:
        return image_processor(**input_dict, return_tensors="pt")
    else:
        return {}


def _get_paligemma_token_type_ids(
    imglens: Sequence[int], seqlens: Sequence[int], processor: "ProcessorMixin"
) -> List[List[int]]:
    r"""
    Gets paligemma token type ids for computing loss.

    Returns:
        batch_token_type_ids: shape (batch_size, sequence_length)
    """
    batch_token_type_ids = []
    for imglen, seqlen in zip(imglens, seqlens):
        image_seqlen = imglen * getattr(processor, "image_seqlen")
        batch_token_type_ids.append([0] * image_seqlen + [1] * (seqlen - image_seqlen))

    return batch_token_type_ids


class BasePlugin:
    def __init__(self, image_token: Optional[str], video_token: Optional[str]) -> None:
        self.image_token = image_token
        self.video_token = video_token

    def _validate_input(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
    ) -> None:
        if len(images) != 0 and self.image_token is None:
            raise ValueError("This model does not support image input.")

        if len(videos) != 0 and self.video_token is None:
            raise ValueError("This model does not support video input.")

    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        r"""
        Pre-processes input messages before tokenization for VLMs.
        """
        self._validate_input(images, videos)
        return messages

    def process_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
    ) -> Tuple[List[int], Optional[List[int]]]:
        r"""
        Pre-processes token ids after tokenization for VLMs.
        """
        self._validate_input(images, videos)
        return input_ids, labels

    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        r"""
        Builds batched multimodal inputs for VLMs.
        """
        self._validate_input(images, videos)
        return {}


class LlavaPlugin(BasePlugin):
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens = 0
        image_seqlen = getattr(processor, "image_seqlen")
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                num_image_tokens += 1
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}", 1)

            message["content"] = content.replace("{{image}}", self.image_token * image_seqlen)

        if len(images) != num_image_tokens:
            raise ValueError("The number of images does not match the number of {} tokens".format(IMAGE_PLACEHOLDER))

        return messages

    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return _get_mm_inputs(images, videos, processor)


class PaliGemmaPlugin(BasePlugin):
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                num_image_tokens += 1
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}", 1)

            message["content"] = content.replace("{{image}}", "")

        if len(images) != num_image_tokens:
            raise ValueError("The number of images does not match the number of {} tokens".format(IMAGE_PLACEHOLDER))

        return messages

    def process_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
    ) -> Tuple[List[int], Optional[List[int]]]:
        self._validate_input(images, videos)
        num_images = len(images)
        image_seqlen = num_images * getattr(processor, "image_seqlen")
        image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        input_ids = [image_token_id] * image_seqlen + input_ids
        if labels is not None:
            labels = [IGNORE_INDEX] * image_seqlen + labels

        return input_ids, labels

    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        mm_inputs = _get_mm_inputs(images, videos, processor)
        mm_inputs["token_type_ids"] = _get_paligemma_token_type_ids(imglens, seqlens, processor)
        return mm_inputs


class Qwen2vlPlugin(BasePlugin):
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        merge_length: int = getattr(image_processor, "merge_size") ** 2
        mm_inputs = _get_mm_inputs(images, videos, processor)
        image_grid_thw = mm_inputs.get("image_grid_thw", [])
        video_grid_thw = mm_inputs.get("video_grid_thw", [])

        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if num_image_tokens >= len(image_grid_thw):
                    raise ValueError("`len(images)` is less than the number of {} tokens.".format(IMAGE_PLACEHOLDER))

                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    "<|vision_start|>{}<|vision_end|>".format(
                        self.image_token * (image_grid_thw[num_image_tokens].prod() // merge_length)
                    ),
                    1,
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                if num_video_tokens >= len(video_grid_thw):
                    raise ValueError("`len(videos)` is less than the number of {} tokens.".format(VIDEO_PLACEHOLDER))

                content = content.replace(
                    VIDEO_PLACEHOLDER,
                    "<|vision_start|>{}<|vision_end|>".format(
                        self.video_token * (video_grid_thw[num_video_tokens].prod() // merge_length)
                    ),
                    1,
                )
                num_video_tokens += 1

            message["content"] = content

        if len(images) != num_image_tokens:
            raise ValueError("The number of images does not match the number of {} tokens".format(IMAGE_PLACEHOLDER))

        if len(videos) != num_video_tokens:
            raise ValueError("The number of videos does not match the number of {} tokens".format(VIDEO_PLACEHOLDER))

        return messages

    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return _get_mm_inputs(images, videos, processor)


PLUGINS = {
    "base": BasePlugin,
    "llava": LlavaPlugin,
    "paligemma": PaliGemmaPlugin,
    "qwen2_vl": Qwen2vlPlugin,
}


def get_mm_plugin(
    name: str,
    image_token: Optional[str] = None,
    video_token: Optional[str] = None,
) -> "BasePlugin":
    plugin_class = PLUGINS.get(name, None)
    if plugin_class is None:
        raise ValueError("Multimodal plugin `{}` not found.".format(name))

    return plugin_class(image_token, video_token)
