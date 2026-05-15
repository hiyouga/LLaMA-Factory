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

"""System metrics collection for training monitoring.

Collects CPU utilization, GPU/XPU utilization and memory usage during training.
"""

import threading
import time
from typing import Any

import psutil
import torch

from .level_zero_monitor import get_xpu_utilization


class SystemMetricsCollector:
    """Collects system metrics during training."""

    def __init__(self, device_type: str = "xpu", collection_interval: float = 1.0) -> None:
        """Initialize the metrics collector.

        Args:
            device_type: Device type ('xpu', 'cuda', or 'cpu')
            collection_interval: Interval in seconds between metric collections
        """
        self.device_type = device_type
        self.collection_interval = collection_interval
        self._collecting = False
        self._thread = None
        self._lock = threading.Lock()

        # Metric accumulators
        self._cpu_utilizations = []
        self._memory_utilizations = []
        self._gpu_memory_utilizations = []
        self._gpu_utilizations = []

        # Check device availability
        self._has_gpu = False
        if device_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            self._has_gpu = True
            self._device_count = torch.xpu.device_count()
        elif device_type == "cuda" and torch.cuda.is_available():
            self._has_gpu = True
            self._device_count = torch.cuda.device_count()

        # Get total system memory
        self._total_memory = psutil.virtual_memory().total

        # Get total GPU memory if available
        self._total_gpu_memory = 0
        if self._has_gpu and device_type == "xpu":
            try:
                # Try to estimate from allocated + reserved
                # XPU doesn't provide total memory directly, use a reasonable estimate
                self._total_gpu_memory = 48 * 1024**3  # 48GB default for Intel Data Center GPU Max
            except Exception:
                self._total_gpu_memory = 0
        elif self._has_gpu and device_type == "cuda":
            try:
                self._total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
            except Exception:
                self._total_gpu_memory = 0

    def _collect_metrics(self) -> None:
        """Background thread function to collect metrics."""
        while self._collecting:
            try:
                # CPU utilization (percentage)
                cpu_percent = psutil.cpu_percent(interval=0.1)

                # Memory utilization (percentage)
                memory_info = psutil.virtual_memory()
                memory_percent = memory_info.percent

                # GPU metrics
                gpu_memory_percent = 0.0
                gpu_util_percent = 0.0

                if self._has_gpu:
                    if self.device_type == "xpu":
                        try:
                            # Get memory usage from PyTorch XPU
                            allocated = torch.xpu.memory_allocated(0)
                            reserved = torch.xpu.memory_reserved(0)
                            # Use reserved as it includes allocated + cached
                            if self._total_gpu_memory > 0:
                                gpu_memory_percent = (reserved / self._total_gpu_memory) * 100
                            # Get GPU utilization from Level Zero API
                            gpu_util_percent = get_xpu_utilization(0)
                        except Exception:
                            pass
                    elif self.device_type == "cuda":
                        try:
                            # CUDA memory
                            allocated = torch.cuda.memory_allocated(0)
                            if self._total_gpu_memory > 0:
                                gpu_memory_percent = (allocated / self._total_gpu_memory) * 100
                            # CUDA doesn't provide utilization directly either
                            gpu_util_percent = 0.0
                        except Exception:
                            pass

                # Store metrics
                with self._lock:
                    self._cpu_utilizations.append(cpu_percent)
                    self._memory_utilizations.append(memory_percent)
                    if self._has_gpu:
                        self._gpu_memory_utilizations.append(gpu_memory_percent)
                        self._gpu_utilizations.append(gpu_util_percent)

            except Exception:
                # Silently continue on errors to avoid disrupting training
                pass

            time.sleep(self.collection_interval)

    def start(self) -> None:
        """Start collecting metrics in background thread."""
        if self._collecting:
            return

        self._collecting = True
        self._thread = threading.Thread(target=self._collect_metrics, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop collecting metrics."""
        self._collecting = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_metrics(self) -> dict[str, Any]:
        """Get collected metrics statistics.

        Returns:
            Dictionary with average metrics:
                - train_cpu_utilization: Average CPU utilization percentage
                - train_memory_utilization: Average memory utilization percentage
                - train_gpu_utilization: Average GPU utilization percentage (0 if not available)
                - train_gpu_memory_utilization: Average GPU memory utilization percentage
        """
        with self._lock:
            cpu_utils = self._cpu_utilizations.copy()
            mem_utils = self._memory_utilizations.copy()
            gpu_mem_utils = self._gpu_memory_utilizations.copy()
            gpu_utils = self._gpu_utilizations.copy()

        metrics = {
            "train_cpu_utilization": round(sum(cpu_utils) / len(cpu_utils), 2) if cpu_utils else 0.0,
            "train_memory_utilization": round(sum(mem_utils) / len(mem_utils), 2) if mem_utils else 0.0,
            "train_gpu_utilization": round(sum(gpu_utils) / len(gpu_utils), 2) if gpu_utils else 0.0,
            "train_gpu_memory_utilization": (
                round(sum(gpu_mem_utils) / len(gpu_mem_utils), 2) if gpu_mem_utils else 0.0
            ),
        }

        return metrics

    def reset(self) -> None:
        """Reset all collected metrics."""
        with self._lock:
            self._cpu_utilizations.clear()
            self._memory_utilizations.clear()
            self._gpu_memory_utilizations.clear()
            self._gpu_utilizations.clear()
