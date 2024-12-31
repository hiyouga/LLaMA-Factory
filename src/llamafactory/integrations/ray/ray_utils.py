
import os


def should_use_ray():
    return os.getenv("USE_RAY", "0").lower() in ["true", "1"]



