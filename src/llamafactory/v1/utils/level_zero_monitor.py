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

"""Level Zero GPU monitoring for Intel XPU devices.

Uses ctypes to call Level Zero API for GPU utilization metrics.
"""

import ctypes
from typing import Optional


class LevelZeroMonitor:
    """Monitor GPU utilization using Level Zero API."""

    def __init__(self, device_index: int = 0) -> None:
        """Initialize Level Zero monitor.

        Args:
            device_index: GPU device index (default 0)
        """
        self.device_index = device_index
        self._ze = None
        self._initialized = False
        self._driver_handle = None
        self._device_handle = None

        self._init_level_zero()

    def _init_level_zero(self) -> None:
        """Initialize Level Zero library and get device handles."""
        try:
            # Load Level Zero library
            try:
                self._ze = ctypes.CDLL("libze_loader.so.1")
            except OSError:
                self._ze = ctypes.CDLL("libze_loader.so")

            # Define Level Zero types and structures
            ze_result_t = ctypes.c_int
            ZE_RESULT_SUCCESS = 0

            # Initialize Level Zero
            zeInit = self._ze.zeInit
            zeInit.argtypes = [ctypes.c_uint32]
            zeInit.restype = ze_result_t

            result = zeInit(0)  # ZE_INIT_FLAG_GPU_ONLY = 0
            if result != ZE_RESULT_SUCCESS:
                return

            # Get driver count
            zeDriverGet = self._ze.zeDriverGet
            driver_count = ctypes.c_uint32(0)
            result = zeDriverGet(ctypes.byref(driver_count), None)
            if result != ZE_RESULT_SUCCESS or driver_count.value == 0:
                return

            # Get driver handles
            driver_handles = (ctypes.c_void_p * driver_count.value)()
            result = zeDriverGet(ctypes.byref(driver_count), driver_handles)
            if result != ZE_RESULT_SUCCESS:
                return

            self._driver_handle = driver_handles[0]

            # Get device count
            zeDeviceGet = self._ze.zeDeviceGet
            device_count = ctypes.c_uint32(0)
            result = zeDeviceGet(self._driver_handle, ctypes.byref(device_count), None)
            if result != ZE_RESULT_SUCCESS or device_count.value == 0:
                return

            # Get device handles
            device_handles = (ctypes.c_void_p * device_count.value)()
            result = zeDeviceGet(self._driver_handle, ctypes.byref(device_count), device_handles)
            if result != ZE_RESULT_SUCCESS:
                return

            if self.device_index < device_count.value:
                self._device_handle = device_handles[self.device_index]
                self._initialized = True

        except Exception:
            # Silently fail - GPU utilization will remain 0
            self._initialized = False

    def get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage.

        Returns:
            GPU utilization as percentage (0-100), or 0.0 if unavailable
        """
        if not self._initialized:
            return 0.0

        try:
            # Use Sysman (System Management) API for utilization
            # zesDeviceEnumEngineGroups to get engine groups
            # zesEngineGetActivity to get utilization

            # Define structures for Sysman
            ZES_STRUCTURE_TYPE_ENGINE_PROPERTIES = 0x00020001

            class zes_engine_properties_t(ctypes.Structure):
                _fields_ = [
                    ("stype", ctypes.c_int),
                    ("pNext", ctypes.c_void_p),
                    ("type", ctypes.c_int),
                    ("onSubdevice", ctypes.c_int),
                    ("subdeviceId", ctypes.c_uint32),
                ]

            class zes_engine_stats_t(ctypes.Structure):
                _fields_ = [
                    ("activeTime", ctypes.c_uint64),
                    ("timestamp", ctypes.c_uint64),
                ]

            # Get engine group count
            zesDeviceEnumEngineGroups = self._ze.zesDeviceEnumEngineGroups
            count = ctypes.c_uint32(0)
            result = zesDeviceEnumEngineGroups(self._device_handle, ctypes.byref(count), None)

            if result != 0 or count.value == 0:  # ZE_RESULT_SUCCESS = 0
                return 0.0

            # Get engine handles
            handles = (ctypes.c_void_p * count.value)()
            result = zesDeviceEnumEngineGroups(self._device_handle, ctypes.byref(count), handles)

            if result != 0:
                return 0.0

            # Get activity from first compute engine
            zesEngineGetActivity = self._ze.zesEngineGetActivity
            stats = zes_engine_stats_t()

            for handle in handles:
                result = zesEngineGetActivity(handle, ctypes.byref(stats))
                if result == 0 and stats.timestamp > 0:
                    # Calculate utilization: (activeTime / timestamp) * 100
                    utilization = (stats.activeTime / stats.timestamp) * 100.0
                    return min(100.0, max(0.0, utilization))

            return 0.0

        except Exception:
            return 0.0


# Singleton instance
_monitor_instance: Optional[LevelZeroMonitor] = None


def get_xpu_utilization(device_index: int = 0) -> float:
    """Get XPU utilization percentage.

    Args:
        device_index: GPU device index

    Returns:
        GPU utilization percentage (0-100)
    """
    global _monitor_instance

    if _monitor_instance is None:
        _monitor_instance = LevelZeroMonitor(device_index)

    return _monitor_instance.get_gpu_utilization()
