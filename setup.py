# Copyright 2024 the LlamaFactory team.
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

import os
import re
from typing import List

from setuptools import find_packages, setup


def get_version() -> str:
    with open(os.path.join("src", "llamafactory", "extras", "env.py"), encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"{}\W*=\W*\"([^\"]+)\"".format("VERSION")
        (version,) = re.findall(pattern, file_content)
        return version


def get_requires() -> List[str]:
    with open("requirements.txt", encoding="utf-8") as f:
        file_content = f.read()
        lines = [line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")]
        return lines


def get_console_scripts() -> List[str]:
    console_scripts = ["llamafactory-cli = llamafactory.cli:main"]
    if os.environ.get("ENABLE_SHORT_CONSOLE", "1").lower() in ["true", "1"]:
        console_scripts.append("lmf = llamafactory.cli:main")

    return console_scripts


extra_require = {
    "torch": ["torch>=1.13.1"],
    "torch-npu": ["torch==2.1.0", "torch-npu==2.1.0.post3", "decorator"],
    "metrics": ["nltk", "jieba", "rouge-chinese"],
    "deepspeed": ["deepspeed>=0.10.0,<=0.14.4"],
    "liger-kernel": ["liger-kernel"],
    "bitsandbytes": ["bitsandbytes>=0.39.0"],
    "hqq": ["hqq"],
    "eetq": ["eetq"],
    "gptq": ["optimum>=1.17.0", "auto-gptq>=0.5.0"],
    "awq": ["autoawq"],
    "aqlm": ["aqlm[gpu]>=1.1.0"],
    "vllm": ["vllm>=0.4.3,<0.6.5"],
    "galore": ["galore-torch"],
    "badam": ["badam>=1.2.1"],
    "adam-mini": ["adam-mini"],
    "qwen": ["transformers_stream_generator"],
    "modelscope": ["modelscope"],
    "openmind": ["openmind"],
    "dev": ["pre-commit", "ruff", "pytest"],
}


def main():
    setup(
        name="llamafactory",
        version=get_version(),
        author="hiyouga",
        author_email="hiyouga" "@" "buaa.edu.cn",
        description="Easy-to-use LLM fine-tuning framework",
        long_description=open("README.md", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        keywords=["LLaMA", "BLOOM", "Falcon", "LLM", "ChatGPT", "transformer", "pytorch", "deep learning"],
        license="Apache 2.0 License",
        url="https://github.com/hiyouga/LLaMA-Factory",
        package_dir={"": "src"},
        packages=find_packages("src"),
        python_requires=">=3.8.0",
        install_requires=get_requires(),
        extras_require=extra_require,
        entry_points={"console_scripts": get_console_scripts()},
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )


if __name__ == "__main__":
    main()
