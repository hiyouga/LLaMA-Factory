# Copyright 2026 the LlamaFactory team.
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

from llamafactory.extras.ploting import gen_loss_plot


def test_gen_loss_plot_preserves_zero_loss():
    figure = gen_loss_plot(
        [
            {"current_steps": 1, "loss": 0.0},
            {"current_steps": 2},
            {"current_steps": 3, "loss": 1.0},
            {"current_steps": 4, "loss": None},
        ]
    )

    original_line = figure.axes[0].lines[0]
    assert original_line.get_xdata().tolist() == [1, 3]
    assert original_line.get_ydata().tolist() == [0.0, 1.0]
