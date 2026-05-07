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

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProfilerArguments:
    r"""Arguments for torch profiler configuration."""

    enable_torch_profiler: bool = field(
        default=False,
        metadata={"help": "Whether to enable torch profiler for collecting performance traces."},
    )
    profiler_output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to write profiler traces. Defaults to <output_dir>/profiler if not set."},
    )
    profiler_wait_steps: int = field(
        default=1,
        metadata={"help": "Number of steps to skip at the start of each profiling cycle."},
    )
    profiler_warmup_steps: int = field(
        default=1,
        metadata={"help": "Number of profiler warm-up steps per cycle."},
    )
    profiler_active_steps: int = field(
        default=1,
        metadata={"help": "Number of steps to actively record per cycle."},
    )
    profiler_repeat: int = field(
        default=1,
        metadata={"help": "Number of profiling cycles. Set to 0 for continuous profiling."},
    )
    profiler_record_shapes: bool = field(
        default=True,
        metadata={"help": "Whether to record tensor shapes during profiling."},
    )
    profiler_profile_memory: bool = field(
        default=True,
        metadata={"help": "Whether to profile memory usage."},
    )
    profiler_with_stack: bool = field(
        default=True,
        metadata={"help": "Whether to record stack traces during profiling."},
    )
