"""Generate a small, offline OCR dataset for testing the SFT pipeline."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/minicpm_ocr_mock"))
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for index, text in enumerate(["HELLO OCR 123", "INVOICE TOTAL 42", "DATE 2026-10-20", "MiniCPM OCR TEST"]):
        image = Image.new("RGB", (224, 112), "white")
        ImageDraw.Draw(image).text((12, 40), text, fill="black", font_size=18)
        path = root / f"ocr_{index}.png"
        image.save(path)
        records.append(
            {
                "messages": [
                    {"role": "user", "content": "<image>\nRead all text in this image."},
                    {"role": "assistant", "content": text},
                ],
                "images": [str(path)],
            }
        )
    (root / "train.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    info = {
        "minicpm_ocr_mock": {
            "file_name": "train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
        }
    }
    (root / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} synthetic OCR examples to {root}")


if __name__ == "__main__":
    main()
