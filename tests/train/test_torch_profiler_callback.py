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
# See the License for the specific language governing permissions and,
# limitations under the License.

from dataclasses import dataclass

import pytest
import torch

from llamafactory.train.callbacks import TorchProfilerCallback


@dataclass
class _FakeArgs:
    profiler_output_dir: str = None
    profiler_wait_steps: int = 0
    profiler_warmup_steps: int = 0
    profiler_active_steps: int = 1
    profiler_repeat: int = 1
    profiler_record_shapes: bool = False
    profiler_profile_memory: bool = False
    profiler_with_stack: bool = False


@pytest.mark.runs_on(["cpu", "mps"])
def test_profiler_callback_warns_missing_npu_activity(monkeypatch, caplog):
    r"""``TorchProfilerCallback.on_train_begin`` should not crash if the installed
    torch build lacks ``ProfilerActivity.NPU`` (older torch shipped without NPU).

    Before the fix, the broad ``except Exception: pass`` silently swallowed the
    ``AttributeError``, leaving the user with no NPU traces and no warning.

    After the fix, the warning ``"... ProfilerActivity.NPU" ...`` is surfaced so
    the user can see why NPU tracing is disabled.
    """
    from llamafactory.train import callbacks as cb_mod

    # Force the code path that thinks NPU is available, then attempt to access the
    # enum value — which we make missing via a fake ProfilerActivity class.
    monkeypatch.setattr(cb_mod, "is_torch_npu_available", lambda: True)
    monkeypatch.setattr(cb_mod, "is_torch_cuda_available", lambda: False)

    class _NoNPUActivity:
        CPU = torch.profiler.ProfilerActivity.CPU
        CUDA = torch.profiler.ProfilerActivity.CUDA
        # deliberately omit NPU

    monkeypatch.setattr(torch.profiler, "ProfilerActivity", _NoNPUActivity())

    callback = TorchProfilerCallback(_FakeArgs())

    import logging

    with caplog.at_level(logging.WARNING, logger="llamafactory"):
        try:
            callback.on_train_begin(
                args=type("A", (), {"output_dir": "/tmp/opencode_test_prof_no_npu"})(),
                state=type("S", (), {})(),
                control=None,
            )
        finally:
            if getattr(callback, "profiler", None) is not None:
                callback.profiler.stop()

    messages = " ".join(record.message for record in caplog.records)
    assert "ProfilerActivity.NPU" in messages


@pytest.mark.runs_on(["cpu", "mps"])
def test_profiler_callback_handles_failed_activity(monkeypatch, caplog):
    r"""If appending a profiler activity raises for any reason, the callback must
    log a warning instead of silently swallowing the error.
    """

    def boom():
        raise RuntimeError("simulated torch issue")

    monkeypatch.setattr(
        torch.profiler,
        "ProfilerActivity",
        type("Boom", (), {"CPU": torch.profiler.ProfilerActivity.CPU, "CUDA": boom, "NPU": boom}),
    )

    from llamafactory.train import callbacks as cb_mod

    monkeypatch.setattr(cb_mod, "is_torch_cuda_available", lambda: True)
    monkeypatch.setattr(cb_mod, "is_torch_npu_available", lambda: False)

    callback = TorchProfilerCallback(_FakeArgs())

    import logging

    with caplog.at_level(logging.WARNING, logger="llamafactory"):
        try:
            callback.on_train_begin(
                args=type("A", (), {"output_dir": "/tmp/opencode_test_prof_boom"})(),
                state=type("S", (), {})(),
                control=None,
            )
        finally:
            if getattr(callback, "profiler", None) is not None:
                callback.profiler.stop()

    messages = " ".join(record.message for record in caplog.records)
    assert "Failed to add device profiler activities" in messages


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
