#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把 NeMo manifest (audio_filepath + text) 转成 LLaMA-Factory 可用的
ShareGPT + <audio> + audios 格式，用于 Gemma3n-E2B ASR SFT。

用法示例：

    python tools/convert_nemo_manifest_to_sharegpt_audio.py \
        --input /path/to/nemo_manifest.jsonl \
        --output data/gemma3n_asr_nemo/gemma3n_asr_nemo_train.jsonl \
        --max-samples 5000 \
        --prompt "请逐字转写下面这段语音，不要额外说明，只输出文本：<audio>"
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="NeMo manifest jsonl 路径")
    p.add_argument("--output", type=str, required=True, help="输出 ShareGPT+audio jsonl 路径")
    p.add_argument(
        "--prompt",
        type=str,
        default="请逐字转写下面这段语音，不要额外说明，只输出文本：<audio>",
        help="human 侧的提示词，必须包含 <audio>",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最多转换多少条（用于单卡 debug），None 表示全部",
    )
    p.add_argument(
        "--audio-key",
        type=str,
        default="audio_filepath",
        help="NeMo manifest 里表示音频路径的字段名",
    )
    p.add_argument(
        "--text-key",
        type=str,
        default="text",
        help="NeMo manifest 里表示转写文本的字段名",
    )
    return p.parse_args()


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def convert_manifest(
    input_path: str,
    output_path: str,
    prompt: str,
    audio_key: str = "audio_filepath",
    text_key: str = "text",
    max_samples: Optional[int] = None,
) -> None:
    if "<audio>" not in prompt:
        raise ValueError("prompt 中必须包含 <audio> 占位符，否则多模态对不上。")

    ensure_parent_dir(output_path)

    n_in = 0
    n_out = 0

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            n_in += 1
            try:
                obj = json.loads(line)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] 跳过无法解析的行 {n_in}: {e}")
                continue

            audio_path = obj.get(audio_key)
            text = obj.get(text_key)

            # 基本清洗：必须同时有音频 + 文本
            if not audio_path or not text:
                continue

            sample = {
                "conversations": [
                    {
                        "from": "human",
                        "value": prompt,
                    },
                    {
                        "from": "gpt",
                        "value": text,
                    },
                ],
                "audios": [audio_path],
            }

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n_out += 1

            if max_samples is not None and n_out >= max_samples:
                break

    print(f"[OK] 读取 NeMo manifest {n_in} 行，生成 ShareGPT+audio 样本 {n_out} 条 -> {output_path}")


def main() -> None:
    args = parse_args()
    convert_manifest(
        input_path=args.input,
        output_path=args.output,
        prompt=args.prompt,
        audio_key=args.audio_key,
        text_key=args.text_key,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
