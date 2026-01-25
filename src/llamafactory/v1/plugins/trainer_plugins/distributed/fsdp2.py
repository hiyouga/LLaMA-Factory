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

import gc
import os

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from transformers import PreTrainedModel

from ....accelerator.helper import get_current_accelerator
from ....accelerator.interface import DistributedInterface
from ....utils.logging import get_logger


logger = get_logger(__name__)


def get_transformer_layer_cls(model: PreTrainedModel) -> type[nn.Module] | None:
    no_split_modules = getattr(model, "_no_split_modules", None)
    if no_split_modules:
        if isinstance(no_split_modules, (list, tuple)):
            for name, module in model.named_modules():
                for cls_name in no_split_modules:
                    if module.__class__.__name__ == cls_name:
                        return module.__class__
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return type(model.model.layers[0])
    if hasattr(model, "layers"):
        return type(model.layers[0])

    return None


class FSDP2Engine:
    def __init__(self, dist_config: dict):
        self.dist_interface = DistributedInterface()
        self.rank = self.dist_interface.get_rank()
        self.local_rank = self.dist_interface.get_local_rank()
        self.world_size = self.dist_interface.get_world_size()
        self.mixed_precision = dist_config.get("mixed_precision", "bf16")
        self.reshard_after_forward = dist_config.get("reshard_after_forward", True)
        self.offload_params = dist_config.get("offload_params", False)
        self.pin_memory = dist_config.get("pin_memory", True)
        self.dcp_path = dist_config.get("dcp_path", None)
        self.device_mesh = self.dist_interface.data_device_mesh

        if self.device_mesh is None:
            logger.warning(
                "Device Mesh not found in DistributedInterface. FSDP2 might fail if not running in distributed mode."
            )

        if self.device_mesh is not None:
            try:
                self.fsdp_mesh = self.device_mesh["dp"]
            except Exception:
                self.fsdp_mesh = self.device_mesh

            logger.info(f"Using Device Mesh: {self.fsdp_mesh}")
        else:
            self.fsdp_mesh = None

    def get_mp_policy(self) -> MixedPrecisionPolicy:
        if self.mixed_precision == "bf16":
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
        elif self.mixed_precision == "fp16":
            param_dtype = torch.float16
            reduce_dtype = torch.float32
        else:
            param_dtype = torch.float32
            reduce_dtype = torch.float32

        return MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cast_forward_inputs=True,
        )

    def prepare_model(self, model: PreTrainedModel) -> PreTrainedModel:
        if self.fsdp_mesh is None:
            logger.warning("No FSDP Mesh available, skipping FSDP wrapping.")
            return model

        mp_policy = self.get_mp_policy()
        layer_cls = get_transformer_layer_cls(model)

        if layer_cls is None:
            logger.warning(
                "Could not identify Transformer Layer class, applying FSDP to the whole model structure only."
            )
            transformer_layer_cls_to_wrap = set()
        else:
            logger.info(f"Applying per-layer FSDP to {layer_cls.__name__}")
            transformer_layer_cls_to_wrap = {layer_cls}

        for name, module in model.named_modules():
            should_wrap = False

            if type(module) in transformer_layer_cls_to_wrap:
                should_wrap = True
            elif isinstance(module, nn.Embedding):
                if not getattr(model.config, "tie_word_embeddings", True):
                    should_wrap = True

            if should_wrap:
                fully_shard(
                    module,
                    mesh=self.fsdp_mesh,
                    reshard_after_forward=self.reshard_after_forward,
                    mp_policy=mp_policy,
                    offload_policy=CPUOffloadPolicy(pin_memory=self.pin_memory) if self.offload_params else None,
                )

        use_gradient_checkpointing = True  # Could be configurable
        if use_gradient_checkpointing:
            if self.rank == 0:
                logger.info("Enabling gradient checkpointing (transformers native)...")

            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        fully_shard(
            model,
            mesh=self.fsdp_mesh,
            reshard_after_forward=self.reshard_after_forward,
            mp_policy=mp_policy,
            offload_policy=CPUOffloadPolicy(pin_memory=self.pin_memory) if self.offload_params else None,
        )

        return model

    @torch.no_grad()
    def materialize_and_load(self, model: PreTrainedModel, hf_model_path: str, dcp_path: str = None):
        if self.rank == 0:
            logger.info("Materializing sharded model params...")

        device = get_current_accelerator()
        model.to_empty(device=device)

        if dcp_path and os.path.exists(dcp_path):
            if self.rank == 0:
                logger.info(f"DCP path found at {dcp_path}. Using efficient Sharded Loading (DCP Load).")
            self._load_from_dcp(model, dcp_path)
        else:
            if self.rank == 0:
                if dcp_path:
                    logger.warning(f"DCP path {dcp_path} not found.")
                logger.info("Using HF Meta Loading (Chunk Load).")
            self._load_weights_from_hf_checkpoint(model, hf_model_path)

        return model

    def shard_model(self, model: PreTrainedModel) -> PreTrainedModel:
        if model.device.type == "meta":
            model = self.prepare_model(model)
            model = self.materialize_and_load(model, hf_model_path=model.config.name_or_path, dcp_path=self.dcp_path)
        else:
            model = self.prepare_model(model)
        return model

    def _load_from_dcp(self, model: PreTrainedModel, dcp_path: str):
        import torch.distributed.checkpoint as dcp

        try:
            if self.rank == 0:
                logger.info(f"Loading distributed checkpoint from {dcp_path} ...")

            options = StateDictOptions(full_state_dict=False, cpu_offload=True)
            local_state_dict = get_model_state_dict(model, options=options)
            dcp.load(state_dict=local_state_dict, checkpoint_id=dcp_path)
            set_model_state_dict(model, local_state_dict, options=options)

            if self.rank == 0:
                logger.info("DCP weights loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load from DCP: {e}")
            raise e

    def _load_weights_from_hf_checkpoint(self, model, hf_model_path):
        import glob
        import json

        hf_model_path = self._resolve_hf_checkpoint_dir(hf_model_path)

        if self.rank == 0:
            logger.info(f"Loading weights from {hf_model_path} ...")

        index_file = os.path.join(hf_model_path, "model.safetensors.index.json")
        is_safetensors = True
        checkpoint_files = []

        if os.path.exists(index_file):
            with open(index_file) as f:
                index = json.load(f)
            checkpoint_files = sorted(set(index["weight_map"].values()))
            checkpoint_files = [os.path.join(hf_model_path, f) for f in checkpoint_files]
        elif os.path.exists(os.path.join(hf_model_path, "model.safetensors")):
            checkpoint_files = [os.path.join(hf_model_path, "model.safetensors")]
        else:
            is_safetensors = False
            index_file = os.path.join(hf_model_path, "pytorch_model.bin.index.json")
            if os.path.exists(index_file):
                with open(index_file) as f:
                    index = json.load(f)
                checkpoint_files = sorted(set(index["weight_map"].values()))
                checkpoint_files = [os.path.join(hf_model_path, f) for f in checkpoint_files]
            elif os.path.exists(os.path.join(hf_model_path, "pytorch_model.bin")):
                checkpoint_files = [os.path.join(hf_model_path, "pytorch_model.bin")]
            else:
                checkpoint_files = sorted(glob.glob(os.path.join(hf_model_path, "*.safetensors")))
                if checkpoint_files:
                    is_safetensors = True
                else:
                    checkpoint_files = sorted(glob.glob(os.path.join(hf_model_path, "*.bin")))

        if not checkpoint_files:
            raise ValueError(f"No checkpoint files found in {hf_model_path}")

        param_map = dict(model.named_parameters())
        total_files = len(checkpoint_files)

        for i, ckpt_file in enumerate(checkpoint_files):
            if self.rank == 0:
                logger.info(f"[{i + 1}/{total_files}] Loading {os.path.basename(ckpt_file)} ...")

            if is_safetensors:
                from safetensors import safe_open

                with safe_open(ckpt_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key in param_map:
                            tensor = f.get_tensor(key)
                            self._copy_weights(param_map[key], tensor)
            else:
                state_dict = torch.load(ckpt_file, map_location="cpu")
                for key, tensor in state_dict.items():
                    if key in param_map:
                        self._copy_weights(param_map[key], tensor)
                del state_dict
                gc.collect()

    def _resolve_hf_checkpoint_dir(self, hf_model_path: str) -> str:
        """Resolve a HF model identifier or local path to a local directory containing checkpoint files.

        - If `hf_model_path` is an existing directory, return it.
        - If it's a file path, return its parent directory.
        - Otherwise treat it as a Hugging Face Hub repo id and download/resolve to the local cache dir.
        """
        if not hf_model_path:
            return hf_model_path

        # Local directory or file path.
        if os.path.isdir(hf_model_path):
            return hf_model_path
        if os.path.isfile(hf_model_path):
            return os.path.dirname(hf_model_path)

        # HuggingFace Hub repo id: snapshot to local cache so we can glob/index files.
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ValueError(
                f"hf_model_path='{hf_model_path}' does not exist locally and huggingface_hub is not available "
                f"to download it. Please provide a local model directory or install huggingface_hub. Error: {e}"
            ) from e

        revision = os.getenv("HF_REVISION")
        offline = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"

        # In distributed runs, let rank0 download first to avoid N-way concurrent downloads.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if self.rank == 0:
                local_dir = snapshot_download(
                    repo_id=hf_model_path,
                    revision=revision,
                    local_files_only=offline,
                    allow_patterns=[
                        "*.safetensors",
                        "*.bin",
                        "*.index.json",
                        "model.safetensors",
                        "model.safetensors.index.json",
                        "pytorch_model.bin",
                        "pytorch_model.bin.index.json",
                        "config.json",
                    ],
                )
                logger.info(f"Resolved HF repo id '{hf_model_path}' to local dir: {local_dir}")
            torch.distributed.barrier()
            if self.rank != 0:
                local_dir = snapshot_download(
                    repo_id=hf_model_path,
                    revision=revision,
                    local_files_only=True,
                    allow_patterns=[
                        "*.safetensors",
                        "*.bin",
                        "*.index.json",
                        "model.safetensors",
                        "model.safetensors.index.json",
                        "pytorch_model.bin",
                        "pytorch_model.bin.index.json",
                        "config.json",
                    ],
                )
            return local_dir

        local_dir = snapshot_download(
            repo_id=hf_model_path,
            revision=revision,
            local_files_only=offline,
            allow_patterns=[
                "*.safetensors",
                "*.bin",
                "*.index.json",
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
                "config.json",
            ],
        )
        if self.rank == 0:
            logger.info(f"Resolved HF repo id '{hf_model_path}' to local dir: {local_dir}")
        return local_dir

    def _copy_weights(self, param, loaded_tensor):
        from torch.distributed._tensor import DTensor, Shard

        if loaded_tensor.dtype != param.dtype:
            loaded_tensor = loaded_tensor.to(param.dtype)

        if isinstance(param, DTensor):
            shard_placement = None
            mesh_dim = -1

            for i, placement in enumerate(param.placements):
                if isinstance(placement, Shard):
                    shard_placement = placement
                    mesh_dim = i
                    break

            local_tensor = param.to_local()

            if shard_placement is None:
                local_tensor.copy_(loaded_tensor)
            else:
                dim = shard_placement.dim
                mesh = param.device_mesh
                my_coordinate = mesh.get_coordinate()
                if my_coordinate is None:
                    return

                rank_in_dim = my_coordinate[mesh_dim]
                world_size_in_dim = mesh.size(mesh_dim)

                full_size = param.shape[dim]
                chunk_size = (full_size + world_size_in_dim - 1) // world_size_in_dim

                start = rank_in_dim * chunk_size
                end = min(start + chunk_size, full_size)

                if start >= full_size:
                    return

                sliced_tensor = loaded_tensor.narrow(dim, start, end - start)

                slices = [slice(None)] * local_tensor.ndim
                slices[dim] = slice(0, sliced_tensor.shape[dim])
                local_tensor[tuple(slices)].copy_(sliced_tensor)
        else:
            param.data.copy_(loaded_tensor)
