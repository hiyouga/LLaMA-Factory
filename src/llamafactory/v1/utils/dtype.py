# Copyright 2025 Bytedance Ltd. and the LlamaFactory team.
#
# This code is inspired by the Bytedance's verl library.
# https://github.com/volcengine/verl/blob/v0.6.1/verl/utils/torch_dtypes.py
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

from contextlib import contextmanager
from typing import Union

import torch
from transformers.utils import is_torch_bf16_available_on_device, is_torch_fp16_available_on_device

from ..accelerator.interface import DistributedInterface


class DtypeRegistry:
    HALF_LIST = ["fp16", "float16", "half", torch.float16]
    FLOAT_LIST = ["fp32", "float32", "float", torch.float32]
    BFLOAT_LIST = ["bf16", "bfloat16", torch.bfloat16]


class DtypeInterface:
    """Type of precision used."""

    _is_fp16_available = is_torch_fp16_available_on_device(DistributedInterface.current_accelerator)
    _is_bf16_available = is_torch_bf16_available_on_device(DistributedInterface.current_accelerator)
    _is_fp32_available = True

    @staticmethod
    def is_available(precision: Union[str, torch.dtype]) -> bool:
        if precision in DtypeRegistry.HALF_LIST:
            return DtypeInterface._is_fp16_available
        elif precision in DtypeRegistry.FLOAT_LIST:
            return DtypeInterface._is_fp32_available
        elif precision in DtypeRegistry.BFLOAT_LIST:
            return DtypeInterface._is_bf16_available
        else:
            raise RuntimeError(f"Unexpected precision: {precision}")

    @staticmethod
    def is_fp16(precision: Union[str, torch.dtype]) -> bool:
        return precision in DtypeRegistry.HALF_LIST

    @staticmethod
    def is_fp32(precision: Union[str, torch.dtype]) -> bool:
        return precision in DtypeRegistry.FLOAT_LIST

    @staticmethod
    def is_bf16(precision: Union[str, torch.dtype]) -> bool:
        return precision in DtypeRegistry.BFLOAT_LIST

    @staticmethod
    def to_dtype(precision: Union[str, torch.dtype]) -> torch.dtype:
        if precision in DtypeRegistry.HALF_LIST:
            return torch.float16
        elif precision in DtypeRegistry.FLOAT_LIST:
            return torch.float32
        elif precision in DtypeRegistry.BFLOAT_LIST:
            return torch.bfloat16
        else:
            raise RuntimeError(f"Unexpected precision: {precision}")

    @staticmethod
    def to_str(precision: torch.dtype) -> str:
        if precision == torch.float16:
            return "float16"
        elif precision == torch.float32:
            return "float32"
        elif precision == torch.bfloat16:
            return "bfloat16"
        else:
            raise RuntimeError(f"Unexpected precision: {precision}")

    @contextmanager
    def set_dtype(self, precision: Union[str, torch.dtype]):
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.to_dtype(precision))
        try:
            yield
        finally:
            torch.set_default_dtype(original_dtype)
