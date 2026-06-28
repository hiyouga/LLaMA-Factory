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

"""Label / verify the rendered token stream.

Locate assistant content regions by scanning for the marker token-id subsequences, assign
``labels``/``loss_weights`` from each assistant message, and run cheap structural guards
(media-placeholder count, one-region-per-assistant-message).
"""

from ...utils.constants import IGNORE_INDEX
from ...utils.helper import get_tokenizer
from ...utils.types import Message, Processor
from .format import _find_subseq


def _check_placeholder_counts(
    processor: Processor, full_text: str, n_images: int, n_videos: int, n_audios: int = 0
) -> None:
    """Guard: every media placeholder in the rendered text must originate from a media block.

    After escaping, literal placeholder strings in user text are broken, so any remaining
    placeholder must come from an image_url/video_url/audio_url block. A mismatch means the data
    encodes media some other way (e.g. inline tags) -- raise rather than crash inside the processor.

    Note: the count is taken on the pre-expansion text, where each media item contributes exactly
    one placeholder token (e.g. Qwen2-Audio emits a single ``<|AUDIO|>`` per audio that the
    processor later expands to many).
    """
    tokenizer = get_tokenizer(processor)
    for attr, count, kind in (
        ("image_token_id", n_images, "image"),
        ("video_token_id", n_videos, "video"),
        ("audio_token_id", n_audios, "audio"),
    ):
        tid = getattr(processor, attr, None)
        if tid is None:
            tid = getattr(tokenizer, attr, None)
        if tid is None:
            continue
        placeholder = tokenizer.convert_ids_to_tokens(tid)
        seen = full_text.count(placeholder)
        if seen != count:
            raise ValueError(
                f"{kind} placeholder count ({seen}) != number of {kind} blocks ({count}); "
                "media must be provided via image_url/video_url content blocks."
            )


def _label_assistant_regions(
    input_ids: list[int], start_ids: list[int], end_ids: list[int], assistant_messages: list[Message]
) -> tuple[list[int], list[float], int]:
    """Label assistant content by scanning for the marker token-id subsequences.

    Each properly-closed ``start_ids ... end_ids`` span is one assistant region; the closing
    ``end_ids`` is included (parity with the previous char-based renderer). Regions map in order
    to assistant messages and take that message's ``loss_weight`` (weight 0 leaves the region
    unlabeled). An unterminated trailing start marker (the generation prompt under
    ``is_generate``) yields no region.
    """
    labels = [IGNORE_INDEX] * len(input_ids)
    loss_weights = [0.0] * len(input_ids)
    regions: list[tuple[int, int]] = []
    n = len(input_ids)
    i = 0
    while i <= n - len(start_ids):
        if input_ids[i : i + len(start_ids)] == start_ids:
            content_start = i + len(start_ids)
            end = _find_subseq(input_ids, end_ids, content_start)
            if end == -1:
                break  # unterminated (generation prompt) -> not labeled
            region_end = end + len(end_ids)
            regions.append((content_start, region_end))
            i = region_end
        else:
            i += 1

    for idx, (start, end) in enumerate(regions):
        weight = assistant_messages[idx].get("loss_weight", 1.0) if idx < len(assistant_messages) else 1.0
        if weight > 1e-6:
            for t in range(start, min(end, n)):
                labels[t] = input_ids[t]
                loss_weights[t] = weight
    return labels, loss_weights, len(regions)


def _verify_render(regions_count: int, assistant_messages: list[Message]) -> None:
    """Cheap structural invariant: exactly one region per assistant message.

    A mismatch signals marker injection that slipped through, a tools-text false marker, or
    marker-detection failure -- fail loud rather than train on corrupted labels.
    """
    n_assistant = len(assistant_messages)
    if regions_count != n_assistant:
        raise ValueError(
            f"assistant region count ({regions_count}) != assistant messages ({n_assistant}); "
            "possible marker collision or detection failure."
        )
