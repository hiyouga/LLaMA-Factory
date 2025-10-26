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


def main():
    from .extras.misc import is_env_enabled

    if is_env_enabled("USE_V1"):
        from .v1 import launcher
    else:
        from . import launcher

    launcher.launch()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
