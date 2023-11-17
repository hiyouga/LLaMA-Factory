import importlib.metadata
import importlib.util


def is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def get_package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except:
        return "0.0.0"


_fastapi_available = is_package_available("fastapi")
_flash_attn2_available = is_package_available("flash_attn") and get_package_version("flash_attn").startswith("2")
_jieba_available = is_package_available("jieba")
_matplotlib_available = is_package_available("matplotlib")
_nltk_available = is_package_available("nltk")
_rouge_available = is_package_available("rouge_chinese")
_starlette_available = is_package_available("sse_starlette")
_uvicorn_available = is_package_available("uvicorn")


def is_fastapi_availble():
    return _fastapi_available


def is_flash_attn2_available():
    return _flash_attn2_available


def is_jieba_available():
    return _jieba_available


def is_matplotlib_available():
    return _matplotlib_available


def is_nltk_available():
    return _nltk_available


def is_rouge_available():
    return _rouge_available


def is_starlette_available():
    return _starlette_available


def is_uvicorn_available():
    return _uvicorn_available
