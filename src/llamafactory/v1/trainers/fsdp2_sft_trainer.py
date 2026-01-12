import logging
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler

from ..accelerator.interface import DistributedInterface
from ..config.arg_parser import get_args
from ..config.arg_utils import PluginConfig
from ..core.data_engine import DataEngine
from ..core.model_engine import ModelEngine
from ..core.utils.data_collator import DefaultCollator
from ..plugins.trainer_plugins.distributed.fsdp2_plugin import FSDP2Plugin
from .sft_trainer import SFTTrainer


logger = logging.getLogger(__name__)


class FSDP2SFTTrainer(SFTTrainer):
    def __init__(self, model, fsdp_plugin, renderer, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        self.fsdp_plugin = fsdp_plugin
        self.renderer = renderer
        self.train_dataloader = None

    def init_model_and_optimizer(self) -> None:
        self.optimizer = self.fsdp_plugin.prepare_optimizer(self.model, lr=self.args.learning_rate)
        # 简单 Scheduler
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=self.args.num_epochs if hasattr(self.args, "num_epochs") else 1, 
            eta_min=self.args.learning_rate * 0.01
        )
        logger.info("FSDP2 Optimizer and Scheduler initialized.")
    
    def create_dataloader(self) -> None:
        if self.train_dataloader is not None:
            return

        # 初始化 Data Collator
        self.data_collator = DefaultCollator(
            processor=self.processor,
            renderer=self.renderer
        )
        
        # Train Loader with DistributedSampler
        if self.dataset is not None:
            sampler = DistributedSampler(
                self.dataset, 
                num_replicas=self.fsdp_plugin.world_size, 
                rank=self.fsdp_plugin.rank, 
                shuffle=True
            )
            
            self.train_dataloader = DataLoader(
                self.dataset, 
                batch_size=self.args.micro_batch_size, 
                sampler=sampler, 
                collate_fn=self.data_collator,
                num_workers=2,
                pin_memory=True
            )
            logger.info("FSDP2 Train Dataloader initialized.")

    def fit(self) -> None:
        # 1. 准备环境
        self.init_model_and_optimizer()
        self.create_dataloader()
        
        # 简单处理 device，如果有 interface 最好从 interface 获取
        # 但 model 已经在正确设备上了（除了 Meta -> GPU 迁移的部分）
        # FSDP 模型输入需要在 GPU 上
        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
        
        self.model.train()
        logger.info("Starting FSDP2 training loop...")
        
        num_epochs = int(self.args.num_epochs) if hasattr(self.args, "num_epochs") else 1

        for epoch in range(num_epochs):
            if hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(epoch)
            
            for step, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                
                loss.backward()
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                
                if step % 10 == 0 and self.fsdp_plugin.rank == 0:
                     loss_val = loss.item()
                     print(f"Epoch {epoch} | Step {step} | Loss: {loss_val:.4f}")
        
        logger.info("Training completed.")


def run_fsdp2_sft(user_args=None):
    args = get_args(user_args)
    data_args, model_args, training_args, sample_args = args
    
    # 临时硬编码测试参数 (仅供调试，实际应从命令行获取)
    data_args.dataset = "/home/frozen/LLaMA-Factory/data/alpaca_en_demo.json"
    model_args.model = "/mnt/f/pretrain_models/Qwen2.5-0.5B"
    training_args.dist_config = {"name": "fsdp2"}
    training_args.num_epochs = 2
    training_args.micro_batch_size = 1

    # 1. 检测 FSDP2 配置
    use_fsdp2 = training_args.dist_config and training_args.dist_config.get("name", None) == "fsdp2"

    if use_fsdp2:
        logger.info("FSDP2 mode detected. Forcing model initialization on Meta device.")
        # 强制使用 Meta Init
        model_args.init_config = PluginConfig({"name": "init_on_meta"})

    # 2. 初始化基础分布式环境 (Process Group)
    DistributedInterface(training_args.dist_config)

    # 3. 准备数据
    data_engine = DataEngine(data_args)
    # Hack: 注入 alpaca converter 确保数据正确处理
    if "default" in data_engine.dataset_infos:
        data_engine.dataset_infos["default"]["converter"] = "alpaca"

    # 4. 准备模型骨架 (ModelEngine)
    model_engine = ModelEngine(model_args)
    model = model_engine.model
    
    fsdp_plugin = None
    # 5. 应用 FSDP2 插件
    if use_fsdp2:
        logger.info("Applying FSDP2 plugin...")
        fsdp_plugin = FSDP2Plugin(training_args.dist_config)
        
        # Step A: Apply Sharding (on Meta)
        model = fsdp_plugin.prepare_model(model)
        
        # Step B: Materialize & Load Weights
        model = fsdp_plugin.materialize_and_load(model, hf_model_path=model_args.model)

    # 6. 初始化 Trainer
    trainer = FSDP2SFTTrainer(
        args=training_args,
        model=model,
        processor=model_engine.processor,
        dataset=data_engine,
        fsdp_plugin=fsdp_plugin,
        renderer=model_engine.renderer
    )

    # 7. 开始训练
    trainer.fit()


if __name__ == "__main__":
    run_fsdp2_sft()

