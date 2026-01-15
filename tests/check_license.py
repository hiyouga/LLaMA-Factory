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

import sys
from pathlib import Path


KEYWORDS = ("Copyright", "LlamaFactory")
VALID_YEARS = ("2025", "2026")


def main():
    path_list: list[Path] = []
    for check_dir in sys.argv[1:]:
        path_list.extend(Path(check_dir).glob("**/*.py"))

    for path in path_list:
        with open(path.absolute(), encoding="utf-8") as f:
            file_content = f.read().strip().split("\n")
            if not file_content[0]:
                continue

            print(f"Check license: {path}")
            first_line = file_content[0]
            assert all(keyword in first_line for keyword in KEYWORDS), f"File {path} does not contain license."
            assert any(year in first_line for year in VALID_YEARS), f"File {path} does not contain valid year."


if __name__ == "__main__":
    main()
