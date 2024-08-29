from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from PIL.Image import Image
from transformers import ProcessorMixin

from ..extras.constants import IGNORE_INDEX, IMAGE_PLACEHOLDER
from ..extras.packages import is_pillow_available


if is_pillow_available():
    import torch
    from PIL import Image


if TYPE_CHECKING:
    from PIL.Image import Image as ImageObject
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.image_processing_utils import BaseImageProcessor


def get_pixel_values(images: Sequence["ImageObject"], processor: "ProcessorMixin") -> "torch.Tensor":
    r"""
    Processes visual inputs. (currently only supports a single image)

    Returns:
        pixel_values: tensor with shape (B, C, H, W)
    """
    image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
    image = images[0] if len(images) != 0 else Image.new("RGB", (100, 100), (255, 255, 255))
    return image_processor([image], return_tensors="pt")["pixel_values"]


def get_paligemma_token_type_ids(input_len: int, processor: "ProcessorMixin") -> List[List[int]]:
    r"""
    Gets paligemma token type ids for computing loss.

    Returns:
        token_type_ids: shape (1, seq_len)
    """
    image_seq_length = getattr(processor, "image_seq_length")
    return [[0] * image_seq_length + [1] * (input_len - image_seq_length)]


def get_qwen2vl_image_inputs(
    images: Sequence["ImageObject"], processor: "ProcessorMixin"
) -> Dict[str, "torch.Tensor"]:
    r"""
    Processes qwen2-vl visual inputs. Supports multiple images.

    Returns:
        pixel_values: tensor with shape (num_patches, patch_dim)
        image_grid_thw: tensot with shape (num_images, 3), where the three numbers are time, width, height

    It holds num_patches == torch.prod(image_grid_thw)
    """
    image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
    if len(images) != 0:
        image_inputs = image_processor(images=images, return_tensors="pt")
    else:
        image = Image.new("RGB", (56, 56), (255, 255, 255))
        image_inputs = image_processor(images=[image], return_tensors="pt")
        image_inputs["image_grid_thw"][0][0] = 0  # fake image

    return {"pixel_values": image_inputs["pixel_values"], "image_grid_thw": image_inputs["image_grid_thw"]}


class BasePlugin:
    def __init__(self, image_token: str) -> None:
        self.image_token = image_token

    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageObject"],
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
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
    ) -> Tuple[List[int], Optional[List[int]]]:
        r"""
        Pre-processes token ids after tokenization for VLMs.
        """
        return input_ids, labels

    def get_mm_inputs(
        self,
        images: Sequence["ImageObject"],
        feature_seqlens: Dict[str, int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Any]:
        r"""
        Builds batched multimodal inputs for VLMs.
        """
        return {}

    def process_model_inputs(
        self,
        model_inputs: Dict[str, List[Any]],
        images: Sequence["ImageObject"],
        feature_seqlens: Dict[str, int],
        processor: Optional["ProcessorMixin"],
    ) -> None:
        r"""
        Appends multimodal inputs to model inputs for VLMs.
        """
        return


class LlavaPlugin(BasePlugin):
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageObject"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        image_count = 0
        new_messages = []
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_count += 1
                if image_count > 1:
                    raise ValueError("Llava model only accepts one image per sample.")

                content = content.replace(IMAGE_PLACEHOLDER, self.image_token, 1)

            new_messages.append({"role": message["role"], "content": content})

        return new_messages

    def get_mm_inputs(
        self,
        images: Sequence["ImageObject"],
        feature_seqlens: Dict[str, int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Any]:
        return {"pixel_values": get_pixel_values(images, processor)}

    def process_model_inputs(
        self,
        model_inputs: Dict[str, List[Any]],
        images: Sequence["ImageObject"],
        feature_seqlens: Dict[str, int],
        processor: Optional["ProcessorMixin"],
    ) -> None:
        mm_inputs = self.get_mm_inputs(images, feature_seqlens, processor)
        model_inputs["pixel_values"].append(mm_inputs["pixel_values"][0])


class PaliGemmaPlugin(BasePlugin):
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageObject"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        image_count = 0
        new_messages = []
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_count += 1
                if image_count > 1:
                    raise ValueError("PaliGemma model only accepts one image per sample.")

                content = content.replace(IMAGE_PLACEHOLDER, "", 1)

            new_messages.append({"role": message["role"], "content": content})

        return new_messages

    def process_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
    ) -> Tuple[List[int], Optional[List[int]]]:
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        image_seq_length: int = getattr(image_processor, "image_seq_length")
        image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        input_ids = [image_token_id] * image_seq_length + input_ids
        if labels is not None:
            labels = [IGNORE_INDEX] * image_seq_length + labels

        return input_ids, labels

    def get_mm_inputs(
        self,
        images: Sequence["ImageObject"],
        feature_seqlens: Dict[str, int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Any]:
        mm_inputs = {"pixel_values": get_pixel_values(images, processor)}
        for feature_name, feature_length in feature_seqlens.items():
            mm_inputs[feature_name] = get_paligemma_token_type_ids(feature_length, processor)

        return mm_inputs

    def process_model_inputs(
        self,
        model_inputs: Dict[str, List[Any]],
        images: Sequence["ImageObject"],
        feature_seqlens: Dict[str, int],
        processor: Optional["ProcessorMixin"],
    ) -> None:
        mm_inputs = self.get_mm_inputs(images, feature_seqlens, processor)
        model_inputs["pixel_values"].append(mm_inputs["pixel_values"][0])
        for feature_name in feature_seqlens.keys():
            model_inputs[feature_name].append(mm_inputs[feature_name][0])


class Qwen2vlPlugin(BasePlugin):
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageObject"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        merge_length: int = getattr(image_processor, "merge_size") ** 2
        if len(images) > 0:
            image_grid_thw = get_qwen2vl_image_inputs(images, processor)["image_grid_thw"]

        index = 0
        new_messages = []
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    "<|vision_start|>{}<|vision_end|>".format(
                        self.image_token * (image_grid_thw[index].prod() // merge_length)
                    ),
                    1,
                )
                index += 1

            new_messages.append({"role": message["role"], "content": content})

        return new_messages

    def get_mm_inputs(
        self,
        images: Sequence["ImageObject"],
        feature_seqlens: Dict[str, int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Any]:
        return get_qwen2vl_image_inputs(images, processor)

    def process_model_inputs(
        self,
        model_inputs: Dict[str, List[Any]],
        images: Sequence["ImageObject"],
        feature_seqlens: Dict[str, int],
        processor: Optional["ProcessorMixin"],
    ) -> None:
        mm_inputs = self.get_mm_inputs(images, feature_seqlens, processor)
        model_inputs["pixel_values"].append(mm_inputs["pixel_values"])
        model_inputs["image_grid_thw"].append(mm_inputs["image_grid_thw"])


PLUGINS = {
    "llava": LlavaPlugin,
    "paligemma": PaliGemmaPlugin,
    "qwen2_vl": Qwen2vlPlugin,
}


def get_mm_plugin(name: str, image_token: str) -> "BasePlugin":
    if name not in PLUGINS:
        raise ValueError("{} not found.".format(name))

    return PLUGINS[name](image_token)
