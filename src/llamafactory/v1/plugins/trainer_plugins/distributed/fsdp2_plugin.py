import gc
import logging
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

from ....accelerator.interface import DistributedInterface, Dim
from ....accelerator.helper import get_current_accelerator

logger = logging.getLogger(__name__)


def get_transformer_layer_cls(model: PreTrainedModel):
    # 策略 1: 优先检查 HF 模型自带的 _no_split_modules
    no_split_modules = getattr(model, "_no_split_modules", None)
    if no_split_modules:
        if isinstance(no_split_modules, list):
            for name, module in model.named_modules():
                for cls_name in no_split_modules:
                    if module.__class__.__name__ == cls_name:
                        return module.__class__

    # 策略 2: 针对常见的 Llama, Qwen, DeepSeek 等结构
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return type(model.model.layers[0])
    if hasattr(model, "layers"):
        return type(model.layers[0])

    return None


class FSDP2Plugin:
    def __init__(self, config: dict):
        self.config = config
        # 使用 DistributedInterface 获取环境信息
        self.dist_interface = DistributedInterface()
        self.rank = self.dist_interface.get_rank()
        self.local_rank = self.dist_interface.get_local_rank()
        self.world_size = self.dist_interface.get_world_size()

        # Parse config
        self.mixed_precision = config.get("mixed_precision", "bf16")
        self.reshard_after_forward = config.get("reshard_after_forward", True)
        self.offload_params = config.get("offload_params", False)
        self.pin_memory = config.get("pin_memory", True)
        # 移除 mesh_shape 和 mesh_dim_names 的手动配置解析，改用 interface
        self.dcp_path = config.get("dcp_path", None)

        # 1. 进程组与设备初始化已由 DistributedInterface 完成

        # 2. 获取 Device Mesh
        # DistributedInterface 提供了 model_device_mesh 和 data_device_mesh
        # FSDP 使用 Data Parallel Mesh (DP)
        self.device_mesh = self.dist_interface.data_device_mesh
        
        # 检查 Mesh 是否存在（可能因为单机或配置原因未初始化）
        if self.device_mesh is None:
            # 如果没有 Device Mesh，尝试回退或者报错。
            # 但 FSDP2 必须依赖 DeviceMesh。
            # 如果 interface 中配置正确，这里应该有值。
            logger.warning("Device Mesh not found in DistributedInterface. FSDP2 might fail if not running in distributed mode.")
        
        # DistributedInterface 默认的 DP Mesh dim name 是 "dp"
        # 我们可以直接使用 self.device_mesh["dp"] 或者根据 dim name 获取
        if self.device_mesh is not None:
             # DistributedInterface 的 mesh 已经是多维的（如果有 Context Parallel），
             # 或者是一维的（只有 DP）。
             # 获取 DP 维度的 SubMesh
             try:
                 # 尝试获取 DP 维度的 mesh
                 self.fsdp_mesh = self.device_mesh["dp"]
             except Exception:
                 # 如果 mesh 本身是一维且名字不是 dp (不太可能，Interface 保证了名字)，或者是 1D mesh 直接用
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
                    offload_policy=CPUOffloadPolicy(pin_memory=self.pin_memory)
                    if self.offload_params
                    else None,
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
    def materialize_and_load(self, model: PreTrainedModel, hf_model_path: str):
        if self.rank == 0:
            logger.info("Materializing sharded model params...")

        device = get_current_accelerator()
        model.to_empty(device=device)

        if self.dcp_path and os.path.exists(self.dcp_path):
            if self.rank == 0:
                logger.info(f"DCP path found at {self.dcp_path}. Using efficient Sharded Loading (DCP Load).")
            self._load_from_dcp(model, self.dcp_path)
        else:
            if self.rank == 0:
                if self.dcp_path:
                    logger.warning(f"DCP path {self.dcp_path} not found.")
                logger.info("Using HF Meta Loading (Chunk Load).")
            self._load_weights_from_hf_checkpoint(model, hf_model_path)

        return model

    def _load_from_dcp(self, model: PreTrainedModel, dcp_path: str):
        import torch.distributed.checkpoint as dcp # 局部引用防止循环依赖风险，虽然这里应该没事
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

    def _copy_weights(self, param, loaded_tensor):
        from torch.distributed._tensor import DTensor, Shard

        if loaded_tensor.dtype != param.dtype:
            loaded_tensor = loaded_tensor.to(param.dtype)

        if isinstance(param, DTensor):
            # 找到 Shard placement 对应的 mesh dimension
            
            shard_placement = None
            mesh_dim = -1
            
            for i, placement in enumerate(param.placements):
                if isinstance(placement, Shard):
                    shard_placement = placement
                    mesh_dim = i
                    break
            
            local_tensor = param.to_local()

            if shard_placement is None:
                # 认为是 Replicate
                local_tensor.copy_(loaded_tensor)
            else:
                dim = shard_placement.dim
                mesh = param.device_mesh
                # mesh_dim 就是上面循环找到的索引

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

    def prepare_optimizer(self, model: torch.nn.Module, lr: float = 1e-4) -> torch.optim.Optimizer:
        # 必须在模型 Sharding 之后构建优化器
        return torch.optim.AdamW(model.parameters(), lr=lr, fused=True)

