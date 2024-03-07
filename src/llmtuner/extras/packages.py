import importlib.metadata
import importlib.util


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _get_package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "0.0.0"


def is_fastapi_availble():
    return _is_package_available("fastapi")


def is_flash_attn2_available():
    return _is_package_available("flash_attn") and _get_package_version("flash_attn").startswith("2")


def is_galore_available():
    return _is_package_available("galore_torch")


def is_jieba_available():
    return _is_package_available("jieba")


def is_matplotlib_available():
    return _is_package_available("matplotlib")


def is_nltk_available():
    return _is_package_available("nltk")


def is_requests_available():
    return _is_package_available("requests")


def is_rouge_available():
    return _is_package_available("rouge_chinese")


def is_starlette_available():
    return _is_package_available("sse_starlette")


def is_unsloth_available():
    return _is_package_available("unsloth")


def is_uvicorn_available():
    return _is_package_available("uvicorn")


def is_vllm_available():
    return _is_package_available("vllm")
