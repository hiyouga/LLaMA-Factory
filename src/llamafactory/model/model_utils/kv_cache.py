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

from typing import TYPE_CHECKING

from ...extras import logging


logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


def configure_kv_cache(config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool) -> None:
    if not is_trainable:
        setattr(config, "use_cache", model_args.use_kv_cache)
        if hasattr(config, "text_config"):
            setattr(config.text_config, "use_cache", model_args.use_kv_cache)

        if model_args.use_kv_cache:
            logger.info_rank0("KV cache is enabled for faster generation.")
        else:
            logger.info_rank0("KV cache is disabled.")
    else:
        setattr(config, "use_cache", False)
        if hasattr(config, "text_config"):
            setattr(config.text_config, "use_cache", False)

        logger.info_rank0("KV cache is disabled during training.")
