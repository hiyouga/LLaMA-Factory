import logging

from ..accelerator.interface import DistributedInterface
from ..config.arg_parser import get_args
from ..config.arg_utils import PluginConfig
from ..core.data_engine import DataEngine
from ..core.model_engine import ModelEngine
from ..plugins.trainer_plugins.distributed.fsdp2_plugin import FSDP2Plugin
from .sft_trainer import SFTTrainer


logger = logging.getLogger(__name__)


def run_fsdp2_sft(user_args=None):
    args = get_args(user_args)
    data_args, model_args, training_args, sample_args = args
    data_args.dataset = "/Users/frozen/PycharmProjects/LLaMA-Factory/data/alpaca_en_demo.json"
    model_args.model = "/Users/frozen/PycharmProjects/Qwen2.5-0.5B"
    training_args.dist_config = {"name": "fsdp2"}

    # 1. 检测 FSDP2 配置
    use_fsdp2 = training_args.dist_config and training_args.dist_config.get("name", None) == "fsdp2"

    if use_fsdp2:
        logger.info("FSDP2 mode detected. Forcing model initialization on Meta device.")
        # 强制使用 Meta Init
        model_args.init_config = PluginConfig({"name": "init_on_meta"})

    # 2. 初始化基础分布式环境 (Process Group)
    # 即使是 FSDP2，也需要基础的 Rank/WorldSize 信息，由 DistributedInterface 统一管理
    DistributedInterface(training_args.dist_config)

    # 3. 准备数据
    data_engine = DataEngine(data_args)

    # 4. 准备模型骨架 (ModelEngine)
    # 由于设置了 init_on_meta，这里返回的模型在 Meta Device 上 (无权重)
    model_engine = ModelEngine(model_args)
    model = model_engine.model

    # 5. 应用 FSDP2 插件 (Sharding + Materialize + Loading)
    if use_fsdp2:
        logger.info("Applying FSDP2 plugin...")
        fsdp_plugin = FSDP2Plugin(training_args.dist_config)

        # Step A: Apply Sharding (on Meta)
        model = fsdp_plugin.prepare_model(model)

        # Step B: Materialize & Load Weights
        # 使用 ModelEngine 解析出的 model_path (通常在 model_args.model)
        # 注意：这里需要传递 dcp_path，它应该在 dist_config 中
        # 如果 model_args 中有其他关于路径的信息，也可以在这里传递
        model = fsdp_plugin.materialize_and_load(model, hf_model_path=model_args.model)

        # Step C: Prepare Optimizer (Optional here, typically done in Trainer)
        # 但 FSDP2 可以在这里提前准备，或者在 Trainer 中通过 plugin hook 调用
        # 为了兼容 BaseTrainer，我们可能需要将 optimizer 传递给 Trainer，或者让 Trainer 识别 FSDP 模型

    # 6. 初始化 Trainer
    trainer = SFTTrainer(
        args=training_args,
        model=model,
        processor=model_engine.processor,
        dataset=data_engine,
    )

    # 如果使用了 FSDP2，我们需要确保 Trainer 使用正确的优化器构建方式
    # FSDP2 要求优化器在 Sharding 之后构建
    if use_fsdp2:
        # 覆盖 Trainer 的默认优化器初始化，或者手动设置
        # 简单的做法：直接在这里创建优化器并赋值给 trainer (如果 BaseTrainer 支持)
        # 或者，更优雅的做法是 SFTTrainer 能够识别 model 是 FSDP 包装过的

        # 这里演示手动设置（假设 BaseTrainer 允许，或者我们在 fit 前调用）
        # trainer.optimizer = fsdp_plugin.prepare_optimizer(model, lr=training_args.learning_rate)
        # 由于 BaseTrainer.fit() 可能会自己创建优化器，我们需要小心。
        # 假设 BaseTrainer 有 init_model_and_optimizer 方法，我们可以 Monkey Patch 或者继承覆盖

        # 为了演示，我们替换 trainer.init_model_and_optimizer
        def fsdp2_init_optimizer(self):
            self.optimizer = fsdp_plugin.prepare_optimizer(self.model, lr=self.args.learning_rate)
            self.lr_scheduler = None  # 或者初始化 scheduler
            logger.info("Initialized FSDP2 optimizer.")

        import types

        trainer.init_model_and_optimizer = types.MethodType(fsdp2_init_optimizer, trainer)

    # 7. 开始训练
    trainer.fit()


if __name__ == "__main__":
    run_fsdp2_sft()
