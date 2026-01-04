import gc

# 简单的日志工具
import logging
import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)

    torch.npu.deterministic = True
    torch.npu.benchmark = False


class FSDP2Config:
    def __init__(
        self,
        mixed_precision: str = "bf16",  # bf16, fp16, fp32
        reshard_after_forward: bool = True,
        offload_params: bool = False,
        mesh_shape: Optional[tuple[int, ...]] = None,  # (dp, ep) 等，默认 (world_size,)
        mesh_dim_names: Optional[tuple[str, ...]] = None,  # ("dp", "ep") 等，默认 ("fsdp",)
        pin_memory: bool = True,  # 是否将参数内存 pinned 到 CPU，True 可以提高 H2D/D2H 效率，但会占用更多 CPU 内存
    ):
        self.mixed_precision = mixed_precision
        self.reshard_after_forward = reshard_after_forward
        self.offload_params = offload_params
        self.pin_memory = pin_memory
        self.mesh_shape = mesh_shape
        self.mesh_dim_names = mesh_dim_names


class FSDP2Engine:
    def __init__(self, config: FSDP2Config):
        self.config = config
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        # 1. 初始化进程组 (如果尚未初始化)
        if not dist.is_initialized():
            dist.init_process_group(backend="hccl")

        torch.npu.set_device(self.local_rank)

        # 2. 初始化 Device Mesh
        # 如果没有指定 shape，默认是全数据的 FSDP (1D Mesh)
        if self.config.mesh_shape is None:
            self.mesh_shape = (self.world_size,)
            self.mesh_dim_names = ("fsdp",)
        else:
            self.mesh_shape = self.config.mesh_shape
            self.mesh_dim_names = self.config.mesh_dim_names

        logger.info(f"Initializing Device Mesh with shape {self.mesh_shape} and names {self.mesh_dim_names}")
        self.device_mesh = init_device_mesh("npu", self.mesh_shape, mesh_dim_names=self.mesh_dim_names)

        # 获取用于 FSDP 的 sub-mesh (通常是数据并行维度)
        # 如果是 1D mesh, 就是它自己。如果是 2D (DP, EP)，这里我们需要获取 DP 维度的 mesh。
        if "fsdp" in self.mesh_dim_names:
            self.fsdp_mesh = self.device_mesh
        elif "dp" in self.mesh_dim_names:
            self.fsdp_mesh = self.device_mesh["dp"]
        else:
            raise ValueError(f"Mesh dim names {self.mesh_dim_names} must contain 'fsdp' or 'dp'.")

    def get_mp_policy(self) -> MixedPrecisionPolicy:
        if self.config.mixed_precision == "bf16":
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32  # 梯度归约通常用 fp32 保持精度
        elif self.config.mixed_precision == "fp16":
            param_dtype = torch.float16
            reduce_dtype = torch.float32
        else:
            param_dtype = torch.float32
            reduce_dtype = torch.float32

        return MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cast_forward_inputs=True,  # 自动将输入转为 param_dtype，避免手动转换
        )

    def prepare_model(self, model: PreTrainedModel) -> PreTrainedModel:
        mp_policy = self.get_mp_policy()

        # 1. 识别 Transformer Layers (基于 _no_split_modules 或 Heuristic)
        layer_cls = get_transformer_layer_cls(model)

        if layer_cls is None:
            logger.warning(
                "Could not identify Transformer Layer class, applying FSDP to the whole model structure only."
            )
            transformer_layer_cls_to_wrap = set()
        else:
            logger.info(f"Applying per-layer FSDP to {layer_cls.__name__}")
            transformer_layer_cls_to_wrap = {layer_cls}

        # 2. 遍历模型子模块，对每一层应用 fully_shard
        # FSDP2 最佳实践：自底向上，先 shard 层，最后 shard 整个模型

        for name, module in model.named_modules():
            should_wrap = False

            # Wrap Transformer Blocks
            if type(module) in transformer_layer_cls_to_wrap:
                should_wrap = True

            # Wrap Embeddings (如果未绑定权重)
            elif isinstance(module, nn.Embedding):
                if not getattr(model.config, "tie_word_embeddings", True):
                    should_wrap = True

            if should_wrap:
                fully_shard(
                    module,
                    mesh=self.fsdp_mesh,
                    reshard_after_forward=self.config.reshard_after_forward,
                    mp_policy=mp_policy,
                    offload_policy=CPUOffloadPolicy(pin_memory=self.config.pin_memory)
                    if self.config.offload_params
                    else None,
                )

        # 3. 应用 Gradient Checkpointing (使用 transformers 原生能力)
        # 相比我们自己 hack forward，直接利用 transformers 的接口更稳健
        # 假设这里默认开启 (实际应从 config 读取)
        use_gradient_checkpointing = True

        if use_gradient_checkpointing:
            if self.rank == 0:
                logger.info("Enabling gradient checkpointing (transformers native)...")

            # 开启 GC
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

            # 关键：开启输入梯度，确保 checkpoint 链条完整
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                # 兜底：如果模型没有这个方法，手动对第一个参数开启
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # 4. 对整个模型应用 fully_shard (处理 embedding, output head 等剩余参数)
        fully_shard(
            model,
            mesh=self.fsdp_mesh,
            reshard_after_forward=self.config.reshard_after_forward,
            mp_policy=mp_policy,
            offload_policy=CPUOffloadPolicy(pin_memory=self.config.pin_memory) if self.config.offload_params else None,
        )

        return model

    def prepare_optimizer(self, model: torch.nn.Module, lr: float = 1e-4) -> torch.optim.Optimizer:
        """构建优化器.

        注意：必须在模型 Sharding 之后构建优化器，这样优化器状态也会被自动 Sharded。
        """
        # 简单的参数分组逻辑
        return torch.optim.AdamW(model.parameters(), lr=lr, fused=True)

    def load_model(
        self, hf_model_name_or_path: str, dcp_path: str = None
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """统一的模型加载入口，支持 DCP 高效加载和 HF 兼容加载.

        Args:
            hf_model_name_or_path: Hugging Face 模型路径或名称 (必须提供，用于读取 Config 和 Tokenizer)
            dcp_path: (可选) DCP 权重路径。如果提供且存在，优先使用 DCP 加载。

        Returns:
            model: 已经应用 FSDP2 Sharding 的模型
            tokenizer: 加载的 Tokenizer
        """
        if self.rank == 0:
            logger.info(f"Loading config and tokenizer from {hf_model_name_or_path} ...")

        tokenizer = AutoTokenizer.from_pretrained(hf_model_name_or_path, trust_remote_code=True)
        # 确保 tokenizer 有 pad_token，否则训练会报错
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 策略 1: DCP 加载 (高效，推荐)
        if dcp_path and os.path.exists(dcp_path):
            if self.rank == 0:
                logger.info(f"DCP path found at {dcp_path}. Using efficient Sharded Loading (Meta Init -> DCP Load).")
            model = self._load_from_dcp(hf_model_name_or_path, dcp_path)

        # 策略 2: HF 加载 (兼容，无需转换)
        else:
            if self.rank == 0:
                if dcp_path:
                    logger.warning(f"DCP path {dcp_path} not found.")
                logger.info("Using Standard Loading (CPU Load -> Shard). This may consume more CPU memory on Rank 0.")
            model = self._load_from_hf(hf_model_name_or_path)

        return model, tokenizer

    def _load_from_dcp(self, hf_model_path: str, dcp_path: str) -> PreTrainedModel:
        """内部实现：从 DCP 加载.

        注意：DCP 是分布式 Checkpoint 格式，它包含了所有 Rank 的参数，
        所以可以直接在 Meta 设备上初始化模型结构，然后直接加载 DCP 权重。
        """
        try:
            # 1. Meta Init
            model_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)

            if self.rank == 0:
                logger.info("Model structure created on Meta device.")

            # 2. Apply FSDP2 (Sharding)
            model = self.prepare_model(model)

            # 3. Materialize (To Empty)
            if self.rank == 0:
                logger.info("Materializing sharded model params...")
            # 注意：使用 device="npu" (或 cuda) 直接分配显存
            # 这里硬编码 npu 是为了匹配用户环境，实际应动态获取
            device_type = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
            model.to_empty(device=device_type)

            # 4. DCP Load
            if self.rank == 0:
                logger.info(f"Loading distributed checkpoint from {dcp_path} ...")

            options = StateDictOptions(full_state_dict=False, cpu_offload=True)
            local_state_dict = get_model_state_dict(model, options=options)
            dcp.load(state_dict=local_state_dict, checkpoint_id=dcp_path)
            set_model_state_dict(model, local_state_dict, options=options)

            if self.rank == 0:
                logger.info("DCP weights loaded successfully.")

            return model

        except Exception as e:
            logger.error(f"Failed to load from DCP: {e}")
            raise e

    def _load_from_hf(self, hf_model_path: str) -> PreTrainedModel:
        """内部实现：从 HF 加载 (CPU -> Shard).

        注意：这种方式需要在 CPU 上加载整个模型，然后利用 FSDP 自动分片。
        这种方式虽然简单，但会占用更多 CPU 内存，且无法利用 DCP 的分布式特性。
        """
        try:
            # 1. CPU Load
            # 只在 Rank 0 上打印日志，但所有 Rank 都加载 (或者未来优化为 Rank 0 加载 + 广播)
            # 目前为了实现简单，采用所有 Rank CPU 加载，然后利用 FSDP 自动分片
            # 优化：可以配合 device_map="cpu" 和 low_cpu_mem_usage=True
            if self.rank == 0:
                logger.info(f"Loading model from {hf_model_path} to CPU...")

            model = AutoModelForCausalLM.from_pretrained(
                hf_model_path,
                torch_dtype=self.get_mp_policy().param_dtype,  # 使用 MP 策略的精度加载
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            # 清理缓存
            gc.collect()

            if self.rank == 0:
                logger.info("Pretrained model loaded on CPU. Applying FSDP2...")

            # 2. Apply FSDP2 (Sharding)
            # FSDP 会自动将 CPU 上的参数切分并搬运到 GPU
            model = self.prepare_model(model)

            # 3. 确保所有参数都已就位 (同步点)
            dist.barrier()

            return model

        except Exception as e:
            logger.error(f"Failed to load from HF: {e}")
            raise e

    def save_checkpoint(self, model: torch.nn.Module, path: str):
        from torch.distributed.checkpoint import FileSystemWriter, save
        from torch.distributed.checkpoint.state_dict import get_model_state_dict

        # 获取 State Dict (FSDP2 下不需要 context manager，直接 get_model_state_dict 会处理)
        # 注意：这里我们使用 PyTorch 官方推荐的 DCP 格式，它支持并行的保存和加载
        state_dict = get_model_state_dict(model)

        if self.rank == 0:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Saving checkpoint to {path}")

        # 并行保存
        save(state_dict, checkpoint_id=FileSystemWriter(path))
        if self.rank == 0:
            logger.info("Checkpoint saved.")


# --- 辅助函数 ---


def get_transformer_layer_cls(model: PreTrainedModel):
    # 策略 1: 优先检查 HF 模型自带的 _no_split_modules
    # 这通常包含了需要作为整体被 Wrap 的层类名
    no_split_modules = getattr(model, "_no_split_modules", None)
    if no_split_modules:
        if isinstance(no_split_modules, list):
            # 通常是一个列表，取第一个包含 "layer" 或 "block" 的，或者直接取第一个
            # 为了更准确，我们可以尝试在 model modules 里找匹配的类
            for name, module in model.named_modules():
                for cls_name in no_split_modules:
                    if module.__class__.__name__ == cls_name:
                        return module.__class__
            # 如果没找到实例，但有名字，尝试反射（虽然比较难），或者回退到策略 2

    # 策略 2: 针对常见的 Llama, Qwen, DeepSeek 等结构
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        # 取第一层作为样本
        return type(model.model.layers[0])
    # 针对其他结构可以继续扩展
    if hasattr(model, "layers"):
        return type(model.layers[0])

    return None


def run_fsdp2_training_step(model: torch.nn.Module, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer):
    # 1. Move data to GPU
    device = torch.npu.current_device()
    batch = {k: v.to(device) for k, v in batch.items()}

    # 2. Forward
    outputs = model(**batch)
    loss = outputs.loss

    # 3. Backward
    loss.backward()

    # 4. Clip Grad (FSDP2 方式)
    # 所有的参数已经被 Sharded，直接对 model 所有的 parameters 归一化即可，
    # 但 FSDP2 提供了优化的处理方式，这里使用通用方式，PyTorch底层会处理
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 5. Step
    optimizer.step()
    optimizer.zero_grad()

    return loss.item(), grad_norm


class DummyDataset(Dataset):
    def __init__(self, size=100, seq_len=128, vocab_size=32000):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 简单模拟：随机生成数据
        return {
            "input_ids": torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "labels": torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long),
        }


def main():
    # 0. 设置随机种子
    set_seed(42)

    config = FSDP2Config(mixed_precision="bf16")
    engine = FSDP2Engine(config)

    # 加载模型与权重 (使用新的统一接口)
    hf_model_path = "/data/lcx/Qwen3-8B"  # 用于读取 Config
    dcp_path = "/data/lcx/Qwen3-8B-DCP"  # 你的 DCP 权重目录

    # 自动选择策略：有 DCP 用 DCP，没 DCP 用 HF
    # model, tokenizer = engine.load_model(hf_model_path, dcp_path)
    # 或者强制测试 HF 加载：
    model, tokenizer = engine.load_model(hf_model_path, dcp_path=dcp_path)

    # 准备优化器 (必须在 prepare_model 之后)
    optimizer = engine.prepare_optimizer(model)

    #  准备数据
    if engine.rank == 0:
        logger.info("Preparing dummy data...")

    # 动态获取 vocab_size，防止数据越界导致 NaN
    vocab_size = getattr(model.config, "vocab_size", 32000)
    dataset = DummyDataset(size=40, seq_len=64, vocab_size=vocab_size)
    # 使用 DistributedSampler 确保每个 GPU 处理不同的数据切片
    sampler = DistributedSampler(dataset, num_replicas=engine.world_size, rank=engine.rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler)

    # 6. 训练循环
    model.train()
    if engine.rank == 0:
        logger.info("Starting training loop...")

    for epoch in range(20):  # 跑 20 个 epoch
        sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            loss, grad_norm = run_fsdp2_training_step(model, batch, optimizer)

            if engine.rank == 0 and step % 2 == 0:
                # 注意：DTensor 直接 format 可能会报错，先转为 scalar
                loss_val = loss if isinstance(loss, (float, int)) else loss.item()
                grad_norm_val = grad_norm if isinstance(grad_norm, (float, int)) else grad_norm.item()
                print(f"Epoch {epoch} | Step {step} | Loss: {loss_val:.4f} | GradNorm: {grad_norm_val:.4f}")

    # 7. 保存 (仅保存演示，实际保存路径需要有写权限)
    # engine.save_checkpoint(model, "/tmp/fsdp2_ckpt")
    if engine.rank == 0:
        logger.info("Training finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
