from copy import deepcopy
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

from PIL.Image import Image
from transformers import ProcessorMixin

from ..extras.constants import IGNORE_INDEX, IMAGE_PLACEHOLDER
from ..extras.packages import is_pillow_available


if is_pillow_available():
    from PIL import Image
    from PIL.Image import Image as ImageObject


if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.image_processing_utils import BaseImageProcessor

    class EncodedImage(TypedDict):
        path: Optional[str]
        bytes: Optional[bytes]

    ImageInput = Union[str, EncodedImage, ImageObject]


def _regularize_images(images: Sequence["ImageInput"], processor: "ProcessorMixin") -> List["ImageObject"]:
    r"""
    Regularizes images to avoid error. Including reading, resizing and converting.
    """
    image_resolution = getattr(processor, "image_resolution", 512)
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


def _get_mm_inputs(images: Sequence["ImageInput"], processor: "ProcessorMixin") -> Dict[str, "torch.Tensor"]:
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
    if len(images) != 0:
        images = _regularize_images(images, processor)
        image_inputs = image_processor(images=images, return_tensors="pt")
    else:
        image_inputs = {}

    return image_inputs


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
    def __init__(self, image_token: str) -> None:
        self.image_token = image_token

    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        r"""
        Pre-processes input messages before tokenization for VLMs.
        """
        return messages

    def process_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]],
        images: Sequence["ImageInput"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
    ) -> Tuple[List[int], Optional[List[int]]]:
        r"""
        Pre-processes token ids after tokenization for VLMs.
        """
        return input_ids, labels

    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        imglens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        r"""
        Builds batched multimodal inputs for VLMs.
        """
        return {}


class LlavaPlugin(BasePlugin):
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
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
        imglens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        return _get_mm_inputs(images, processor)


class PaliGemmaPlugin(BasePlugin):
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
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
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
    ) -> Tuple[List[int], Optional[List[int]]]:
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
        imglens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        mm_inputs = _get_mm_inputs(images, processor)
        mm_inputs["token_type_ids"] = _get_paligemma_token_type_ids(imglens, seqlens, processor)
        return mm_inputs


class Qwen2vlPlugin(BasePlugin):
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        merge_length: int = getattr(image_processor, "merge_size") ** 2
        if len(images) != 0:
            image_grid_thw = _get_mm_inputs(images, processor)["image_grid_thw"]
        else:
            image_grid_thw = []

        num_image_tokens = 0
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

            message["content"] = content

        if len(images) != num_image_tokens:
            raise ValueError("The number of images does not match the number of {} tokens".format(IMAGE_PLACEHOLDER))

        return messages

    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        imglens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        return _get_mm_inputs(images, processor)


PLUGINS = {
    "base": BasePlugin,
    "llava": LlavaPlugin,
    "paligemma": PaliGemmaPlugin,
    "qwen2_vl": Qwen2vlPlugin,
}


def get_mm_plugin(name: str, image_token: str) -> "BasePlugin":
    plugin_class = PLUGINS.get(name, None)
    if plugin_class is None:
        raise ValueError("Multimodal plugin `{}` not found.".format(name))

    return plugin_class(image_token)
