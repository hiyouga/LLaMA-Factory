"""One-shot patcher for the Nanbeige4.2 remote code cached by transformers.

Run this on the GPU cluster (inside the lhr_py311 env) to make the
trust_remote_code modeling files compatible with transformers>=5.x::

    python patch_nanbeige_remote_code.py                  # auto-locate cache
    python patch_nanbeige_remote_code.py /path/to/modeling_nanbeige.py

It fixes two incompatibilities that crash under new transformers:

1. ``_tied_weights_keys`` is declared as ``List[str]`` (old convention); new
   transformers expects ``Dict[str, str]`` and calls ``.keys()`` on it during
   ``remove_tied_weights_from_state_dict`` -> checkpoint save crashes with
   ``AttributeError: 'list' object has no attribute 'keys'``.

2. ``NanbeigeAttention._init_rope`` reads ``rope_scaling["type"]``, but new
   transformers injects ``{"rope_type": "default", ...}`` -> standalone
   ``AutoModelForCausalLM.from_pretrained(...)`` crashes with
   ``KeyError: 'type'``.

The patch is idempotent: re-running it on already-patched files is a no-op.

NOTE - When is this no longer needed?
    This script (and the matching shims in ``src/llamafactory/model/patcher.py``)
    exist ONLY because the remote code shipped in the Nanbeige4.2 HF repo
    (``Nanbeige/Nanbeige4.2-3B``) predates transformers>=5.x. Once the upstream
    Nanbeige repo updates its ``modeling_nanbeige.py`` /
    ``configuration_nanbeige.py`` to the new conventions:

      * ``_tied_weights_keys`` declared as ``Dict[str, str]`` (target -> source)
        instead of ``List[str]``,
      * ``_init_rope`` reading ``rope_scaling["rope_type"]`` (with a fallback to
        the legacy ``"type"`` key) instead of hard-coding ``rope_scaling["type"]``,

    both this script and the LlamaFactory shims become no-ops and can be
    removed. After such an upstream update, re-pulling the model snapshot
    (``huggingface-cli delete-cache`` or ``rm -rf`` the cached module dir) is
    enough to get the fixed remote code.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import sys


def find_modeling_file() -> str | None:
    cache_root = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    candidates = glob.glob(
        os.path.join(cache_root, "modules", "transformers_modules", "**", "modeling_nanbeige.py"),
        recursive=True,
    )
    # Prefer the most recently modified copy (handles both Nanbeige/... and flat layouts).
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0] if candidates else None


def patch_tied_weights(src: str) -> tuple[str, bool]:
    """Convert ``_tied_weights_keys = [...]`` list literal to a dict literal."""
    pattern = re.compile(r"(_tied_weights_keys\s*=\s*)\[(?P<body>[^\]]*)\]", re.DOTALL)

    def to_dict(m: re.Match) -> str:
        body = m.group("body").strip()
        if not body:
            return f"{m.group(1)}{{}}"
        keys = [k.strip() for k in body.split(",") if k.strip()]
        pairs = []
        for k in keys:
            k_clean = k.strip().strip("'\"")
            source = '"model.embed_tokens.weight"' if k_clean == "lm_head.weight" else k
            pairs.append(f"{k}: {source}")
        return f"{m.group(1)}{{{', '.join(pairs)}}}"

    new_src, n = pattern.subn(to_dict, src)
    return new_src, n > 0


def patch_init_rope(src: str) -> tuple[str, bool]:
    """Make ``_init_rope`` robust to both ``type`` and ``rope_type`` keys.

    We replace every direct ``rope_scaling["type"]`` / ``rope_scaling.get("type", ...)``
    access with a fallback that also tries ``rope_type``. We also inject a one-time
    normalization at the top of ``_init_rope`` so downstream ``["factor"]`` reads
    keep working unchanged.
    """
    changed = False

    # Inject a rope_scaling normalization right after the `def _init_rope(self):` line,
    # unless it's already there.
    init_rope_re = re.compile(r"(def _init_rope\(self\)[^\n]*:\n)", )
    marker = "# transformers>=5.x renamed rope_scaling[\"type\"] -> \"rope_type\""
    if init_rope_re.search(src) and marker not in src:
        normalize = (
            "        " + marker + "\n"
            '        if self.config.rope_scaling is not None and isinstance(self.config.rope_scaling, dict):\n'
            '            rs = dict(self.config.rope_scaling)\n'
            '            if "type" not in rs and "rope_type" in rs:\n'
            '                rs["type"] = rs["rope_type"]\n'
            '            self.config.rope_scaling = rs\n'
        )
        src = init_rope_re.sub(r"\1" + normalize, src, count=1)
        changed = True

    return src, changed


def patch_rope_scaling_validation(src: str) -> tuple[str, bool]:
    """Loosen ``_rope_scaling_validation`` to accept the ``rope_type`` key.

    New transformers may pass ``{"rope_type": "default", "factor": 1.0}`` which
    fails the strict ``len == 2`` + ``type`` checks when injected after config
    construction. Normalize ``rope_type`` -> ``type`` early.
    """
    marker = '# transformers>=5.x renamed rope_scaling["type"] -> "rope_type"'
    if marker in src:
        return src, False  # already patched

    pattern = re.compile(
        r"(def _rope_scaling_validation\(self\):.*?if self\.rope_scaling is None:\s*\n\s*return\s*\n)",
        re.DOTALL,
    )
    normalize = (
        "\n        if self.rope_scaling is not None and isinstance(self.rope_scaling, dict):\n"
        '            if "type" not in self.rope_scaling and "rope_type" in self.rope_scaling:\n'
        '                self.rope_scaling["type"] = self.rope_scaling.pop("rope_type")\n'
    )
    new_src, n = pattern.subn(lambda m: m.group(1) + normalize, src)
    return new_src, n > 0


def patch_file(path: str, patchers) -> None:
    print(f"Patching {path}")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    original = src
    notes = []
    for name, fn in patchers:
        src, did = fn(src)
        if did:
            notes.append(name)
    if src != original:
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        for n in notes:
            print(f"  - {n}")
    else:
        print("  (already patched, no changes)")


def main() -> int:
    if len(sys.argv) > 1:
        modeling_path = sys.argv[1]
    else:
        modeling_path = find_modeling_file()
    if not modeling_path or not os.path.isfile(modeling_path):
        print(
            "ERROR: could not locate modeling_nanbeige.py. "
            "Pass its path explicitly:\n"
            "  python patch_nanbeige_remote_code.py /path/to/modeling_nanbeige.py",
            file=sys.stderr,
        )
        return 1

    patch_file(
        modeling_path,
        [
            ("fixed _tied_weights_keys (list -> dict)", patch_tied_weights),
            ("fixed _init_rope (type / rope_type fallback)", patch_init_rope),
        ],
    )

    config_path = os.path.join(os.path.dirname(modeling_path), "configuration_nanbeige.py")
    if os.path.isfile(config_path):
        patch_file(
            config_path,
            [("fixed _rope_scaling_validation (rope_type normalization)", patch_rope_scaling_validation)],
        )

    # Clear cached bytecode so the patched source is recompiled.
    pycache = os.path.join(os.path.dirname(modeling_path), "__pycache__")
    if os.path.isdir(pycache):
        shutil.rmtree(pycache)
        print(f"  - cleared {pycache}")

    print("\nDone. Re-run your training / test_embed.py now.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
