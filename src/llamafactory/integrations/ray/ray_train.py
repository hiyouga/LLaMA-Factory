
from typing import Any, Callable, Dict

from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

from .ray_train_args import RayTrainArguments

    
def get_ray_trainer(
    training_function: Callable,
    train_loop_config: Dict[str, Any],
    ray_args: RayTrainArguments,
) -> TorchTrainer:
    
    if not ray_args.use_ray:
        raise ValueError("Ray is not enabled. Please set USE_RAY=1 in your environment.")
    
    trainer = TorchTrainer(
        training_function,
        train_loop_config=train_loop_config,
        scaling_config=ScalingConfig(
            num_workers=ray_args.num_workers,
            resources_per_worker=ray_args.resources_per_worker,
            use_gpu=True,
        ),
    )
    return trainer