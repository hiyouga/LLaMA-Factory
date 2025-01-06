from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

from .ray_utils import should_use_ray


@dataclass
class RayTrainArguments:
    r"""
    Arguments pertaining to the Ray training.
    """

    resources_per_worker: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {"GPU": 1},
        metadata={"help": "The resources per worker for Ray training. Default is to use 1 GPU per worker."},
    )
    num_workers: Optional[int] = field(
        default=1, metadata={"help": "The number of workers for Ray training. Default is 1 worker."}
    )
    placement_strategy: Optional[Literal["SPREAD", "PACK", "STRICT_SPREAD", "STRICT_PACK"]] = field(
        default="PACK", metadata={"help": "The placement strategy for Ray training. Default is PACK."}
    )

    @property
    def use_ray(self) -> bool:
        """
        Always returns the value from the environment variable check.
        This prevents manual setting of use_ray.
        """
        return should_use_ray()
