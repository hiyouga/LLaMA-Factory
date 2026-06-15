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

"""Assistant role markers for supported models (explicit whitelist).

These are NOT full chat templates -- only the auxiliary metadata needed to locate assistant
content spans in the stream produced by the model's own ``apply_chat_template``. We keep an
explicit per-``model_type`` whitelist rather than probing the template at runtime: every entry
here has been verified to encode identically standalone and in-context, so the marker strings can
be tokenized directly (see ``rendering.py``). Add a new entry when adding model support.
"""

# ChatML assistant markers, shared by the Qwen3 / Qwen3.5 family. The start marker is the
# role-opening run ``<|im_start|>assistant\n``; the end marker is the turn terminator ``<|im_end|>``.
_CHATML = ("<|im_start|>assistant\n", "<|im_end|>")

# model_type (transformers ``config.model_type``) -> (assistant_start_marker, assistant_end_marker)
_ASSISTANT_MARKERS: dict[str, tuple[str, str]] = {
    "qwen3": _CHATML,
    "qwen3_moe": _CHATML,
    "qwen3_vl": _CHATML,
    "qwen3_vl_moe": _CHATML,
    "qwen3_5": _CHATML,
}


def resolve_assistant_markers(model_type: str | None) -> tuple[str, str]:
    """Return the (start, end) assistant marker strings for a supported ``model_type``.

    Raises ``ValueError`` for an unknown ``model_type`` -- v1 rendering only supports models that
    are explicitly listed here, by design (no generic template probing).
    """
    if model_type not in _ASSISTANT_MARKERS:
        raise ValueError(
            f"Unsupported model_type {model_type!r} for v1 rendering; "
            f"supported: {sorted(_ASSISTANT_MARKERS)}. "
            "Add an entry to llamafactory.v1.core.rendering.markers to support a new model."
        )
    return _ASSISTANT_MARKERS[model_type]
