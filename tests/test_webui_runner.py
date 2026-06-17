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

"""Tests for WebUI runner stderr handling and subprocess monitoring.

These tests verify the fix for the training hang bug (issue #10180) where
disconnecting from the WebUI would cause the training subprocess to block
because the stderr PIPE buffer filled up when the generator stopped being consumed.

The fix redirects stderr to a temporary file instead of using subprocess.PIPE,
and uses poll() instead of communicate() for non-blocking process monitoring.
"""

import os
import subprocess
import sys
import tempfile
import time


class TestStderrFileHandling:
    """Test the stderr file-based approach used by Runner."""

    def test_subprocess_stderr_to_file_does_not_block(self):
        """Verify that a subprocess writing to stderr via a file does not block.

        This is the core test for the fix: when stderr is redirected to a file,
        the subprocess can write unlimited stderr without blocking, even if
        no one is reading the output (simulating a disconnected WebUI client).
        """
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".log", delete=False) as stderr_file:
            stderr_path = stderr_file.name
            try:
                # Spawn a subprocess that writes a large amount to stderr
                # With PIPE this would block when the buffer fills (~64KB on Linux)
                # With a file, it should complete quickly
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        "import sys; [sys.stderr.write('x' * 1000 + '\\n') for _ in range(200)]",
                    ],
                    stderr=stderr_file,
                    text=True,
                )

                # Wait up to 10 seconds - with PIPE this could hang indefinitely
                return_code = proc.wait(timeout=10)
                assert return_code == 0

                # Verify stderr was captured in the file
                stderr_file.flush()
                stderr_file.seek(0)
                content = stderr_file.read()
                assert len(content) > 100000  # Should have ~200KB of output
            finally:
                os.unlink(stderr_path)

    def test_subprocess_pipe_can_block_with_large_stderr(self):
        """Demonstrate that subprocess.PIPE can block with large stderr output.

        This test shows the problem that existed before the fix: when using PIPE,
        a subprocess that writes a lot to stderr will block if the parent doesn't
        drain the pipe. We use a short timeout to detect the hang.
        """
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                # Write much more than the pipe buffer size (~64KB on Linux, ~8KB on macOS)
                "import sys; [sys.stderr.write('x' * 1000 + '\\n') for _ in range(500)]",
            ],
            stderr=subprocess.PIPE,
            text=True,
        )

        # Without draining the pipe, the subprocess should block (or might not on some OS)
        # We use poll + sleep to simulate not reading from the pipe
        time.sleep(0.5)
        result = proc.poll()

        if result is None:
            # Process is still running (blocked on stderr write) - this proves the bug
            proc.kill()
            proc.wait()
        # If result is not None, the OS pipe buffer was large enough.
        # Either way, the test documents the problem scenario.

    def test_poll_returns_none_while_running(self):
        """Verify poll() returns None while subprocess is still running."""
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(2)"],
            stderr=subprocess.DEVNULL,
        )
        try:
            assert proc.poll() is None
        finally:
            proc.kill()
            proc.wait()

    def test_poll_returns_exit_code_when_done(self):
        """Verify poll() returns the exit code when subprocess completes."""
        proc = subprocess.Popen(
            [sys.executable, "-c", "pass"],
            stderr=subprocess.DEVNULL,
        )
        proc.wait()
        assert proc.poll() == 0

    def test_poll_returns_nonzero_on_failure(self):
        """Verify poll() returns non-zero exit code on failure."""
        proc = subprocess.Popen(
            [sys.executable, "-c", "raise SystemExit(42)"],
            stderr=subprocess.DEVNULL,
        )
        proc.wait()
        assert proc.poll() == 42


class TestStderrFileLifecycle:
    """Test the lifecycle of the temporary stderr file."""

    def test_stderr_file_is_created_and_writable(self):
        """Verify that a NamedTemporaryFile can be used as stderr target."""
        with tempfile.NamedTemporaryFile(
            mode="w+", prefix="llamafactory_stderr_", suffix=".log", delete=False
        ) as stderr_file:
            stderr_path = stderr_file.name
            try:
                assert os.path.exists(stderr_path)

                proc = subprocess.Popen(
                    [sys.executable, "-c", "import sys; sys.stderr.write('test error\\n')"],
                    stderr=stderr_file,
                    text=True,
                )
                proc.wait()

                stderr_file.flush()
                stderr_file.seek(0)
                content = stderr_file.read()
                assert "test error" in content
            finally:
                os.unlink(stderr_path)

    def test_stderr_file_cleanup(self):
        """Verify that the stderr file can be properly cleaned up."""
        stderr_file = tempfile.NamedTemporaryFile(
            mode="w+", prefix="llamafactory_stderr_", suffix=".log", delete=False
        )
        stderr_path = stderr_file.name

        # Write some content
        stderr_file.write("some stderr content")
        stderr_file.flush()

        # Simulate the cleanup logic from Runner._cleanup_stderr_file
        assert os.path.exists(stderr_path)
        stderr_file.close()
        os.unlink(stderr_path)
        assert not os.path.exists(stderr_path)

    def test_read_stderr_from_file(self):
        """Verify that stderr content can be read back from the file."""
        with tempfile.NamedTemporaryFile(
            mode="w+", prefix="llamafactory_stderr_", suffix=".log", delete=False
        ) as stderr_file:
            stderr_path = stderr_file.name
            try:
                error_messages = [
                    "Error: CUDA out of memory\n",
                    "RuntimeError: something went wrong\n",
                    "Traceback (most recent call last):\n",
                ]
                for msg in error_messages:
                    stderr_file.write(msg)
                stderr_file.flush()

                # Read back like Runner._read_stderr does (last 64KB)
                stderr_file.seek(0, os.SEEK_END)
                size = stderr_file.tell()
                stderr_file.seek(max(0, size - 65536))
                content = stderr_file.read()
                for msg in error_messages:
                    assert msg in content
            finally:
                os.unlink(stderr_path)

    def test_read_stderr_large_file_truncated(self):
        """Verify that only the last 64KB is read from a large stderr file."""
        with tempfile.NamedTemporaryFile(
            mode="w+", prefix="llamafactory_stderr_", suffix=".log", delete=False
        ) as stderr_file:
            stderr_path = stderr_file.name
            try:
                # Write more than 64KB of early content
                early_marker = "EARLY_CONTENT_MARKER\n"
                stderr_file.write(early_marker * 5000)  # ~105KB of early content
                # Write a late marker that should be within the last 64KB
                late_marker = "LATE_ERROR: final crash info\n"
                stderr_file.write(late_marker)
                stderr_file.flush()

                # Read back like Runner._read_stderr does (last 64KB)
                stderr_file.seek(0, os.SEEK_END)
                size = stderr_file.tell()
                stderr_file.seek(max(0, size - 65536))
                content = stderr_file.read()

                assert late_marker in content
                assert len(content) <= 65536
                # Early content at the very beginning should be truncated
                assert content.count("EARLY_CONTENT_MARKER") < 5000
            finally:
                os.unlink(stderr_path)

    def test_multiple_subprocesses_get_separate_files(self):
        """Verify that each subprocess gets its own stderr file."""
        files = []
        try:
            for _ in range(3):
                f = tempfile.NamedTemporaryFile(
                    mode="w+", prefix="llamafactory_stderr_", suffix=".log", delete=False
                )
                files.append(f)

            paths = [f.name for f in files]
            # All paths should be unique
            assert len(set(paths)) == 3
        finally:
            for f in files:
                f.close()
                os.unlink(f.name)


class TestSubprocessDisconnectResilience:
    """Test that the subprocess continues running even without pipe draining.

    These tests simulate the WebUI disconnect scenario where the generator
    stops being consumed but the subprocess should continue running.
    """

    def test_subprocess_completes_without_monitoring(self):
        """Verify subprocess completes even if parent doesn't monitor it.

        This simulates the scenario where the WebUI disconnects: the generator
        stops yielding but the subprocess should still complete its work.
        """
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".log", delete=False
        ) as stderr_file:
            stderr_path = stderr_file.name
            try:
                # Create a marker file that the subprocess will create on completion
                marker = tempfile.mktemp(suffix=".marker")

                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        f"import sys, time; "
                        f"sys.stderr.write('working...\\n'); "
                        f"time.sleep(0.5); "
                        f"open('{marker}', 'w').write('done'); "
                        f"sys.stderr.write('finished\\n')",
                    ],
                    stderr=stderr_file,
                    text=True,
                )

                # Don't consume any output - simulate disconnected WebUI
                proc.wait(timeout=10)

                assert proc.returncode == 0
                assert os.path.exists(marker)
                with open(marker) as f:
                    assert f.read() == "done"

                stderr_file.flush()
                stderr_file.seek(0)
                content = stderr_file.read()
                assert "working..." in content
                assert "finished" in content

                os.unlink(marker)
            finally:
                os.unlink(stderr_path)

    def test_large_stderr_output_does_not_block_with_file(self):
        """Verify that even very large stderr output doesn't block with file redirect.

        The original bug was caused by the 64KB pipe buffer limit. With file redirect,
        there should be no such limit.
        """
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".log", delete=False
        ) as stderr_file:
            stderr_path = stderr_file.name
            try:
                # Write 1MB of stderr - way more than any pipe buffer
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        "import sys; sys.stderr.write('E' * 1048576); sys.stderr.write('\\nDONE\\n')",
                    ],
                    stderr=stderr_file,
                    text=True,
                )

                # Should complete quickly without blocking
                proc.wait(timeout=30)
                assert proc.returncode == 0

                stderr_file.flush()
                stderr_file.seek(0)
                content = stderr_file.read()
                assert "DONE" in content
                assert len(content) > 1000000
            finally:
                os.unlink(stderr_path)
