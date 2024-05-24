from typing import TYPE_CHECKING, List, Sequence

from ...extras.packages import is_pillow_available


if is_pillow_available():
    from PIL import Image


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from PIL.Image import Image as ImageObject
    from transformers import ProcessorMixin
    from transformers.image_processing_utils import BaseImageProcessor


def get_pixel_values(images: Sequence["ImageObject"], processor: "ProcessorMixin") -> "NDArray":
    # process visual inputs (currently only supports a single image)
    image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
    image = images[0] if len(images) != 0 else Image.new("RGB", (100, 100), (255, 255, 255))
    return image_processor(image, return_tensors="pt")["pixel_values"][0]  # shape (C, H, W)


def get_paligemma_token_type_ids(input_len: int, processor: "ProcessorMixin") -> List[int]:
    # get paligemma token type ids for computing loss
    image_seq_length = getattr(processor, "image_seq_length")
    return [0] * image_seq_length + [1] * (input_len - image_seq_length)
