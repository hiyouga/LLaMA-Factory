# 为了解决和我一样的小白问题，在入门阶段环境上面，做了一些检查，希望对大家有帮助。
import sys
import subprocess
from packaging import version

RED_START = "\033[91m"
RED_END = "\033[0m"
YELLOW_START = "\033[93m"
YELLOW_END = "\033[0m"

# 包含模块及其所需的最小和推荐版本
required_versions = {
    'python': ('3.8', '3.10'),
    'torch': ('1.13.1', '2.2.0'),
    'transformers': ('4.37.2', '4.38.2'),
    'datasets': ('2.14.3', '2.17.1'),
    'accelerate': ('0.27.2', '0.27.2'),
    'peft': ('0.9.0', '0.9.0'),
    'trl': ('0.7.11', '0.7.11'),
}


def check_torch_cuda_versions():
    # 执行命令行指令
    torch_version = subprocess.run(["python", "-c", "import torch; print(torch.__version__)", ], capture_output=True,
                                   text=True).stdout
    cuda_version = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout

    if "Torch" in torch_version and "CUDA" in cuda_version:
        if torch_version == cuda_version:
            print("Torch and CUDA versions match")
        else:
            print("Torch and CUDA versions may not match. Check https://pytorch.org/get-started/previous-versions/")
    else:
        print("Torch or CUDA version information not found")


# 调用函数
check_torch_cuda_versions()


def get_version(package_name):
    try:
        if package_name == 'python':
            return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        else:
            module = __import__(package_name)
            return module.__version__
    except ModuleNotFoundError:
        return None


def check_version(package_name, min_version, recommended_version):
    actual_version = get_version(package_name)
    if actual_version is None:
        print(
            f"{RED_START}Error: {package_name} is not installed. Required minimum version is {min_version}. Recommended version is {recommended_version}. Please install it.{RED_END}")
        sys.exit(1)
    elif version.parse(actual_version) < version.parse(min_version):
        print(
            f"{RED_START}Warning: {package_name} version is below the minimum required {min_version}. You have {actual_version}. It's recommended to update to at least {recommended_version}.{RED_END}")
        # Continue execution despite the warning
    elif version.parse(actual_version) > version.parse(recommended_version):
        print(
            f"{YELLOW_START}Notice: {package_name} version {actual_version} is above the recommended version {recommended_version}. You may encounter unexpected behavior.{YELLOW_END}")


def check_torch_cuda_versions():
    torch_version = subprocess.run(["python", "-c", "import torch; print(torch.__version__)", ], capture_output=True,
                                   text=True).stdout
    cuda_version = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout
    # 检查 torch 和 CUDA 版本是否匹配
    if "Torch" in torch_version and "CUDA" in cuda_version:
        if torch_version == cuda_version:
            print("Torch and CUDA versions match")
        else:
            print(
                f"{RED_START}Torch and CUDA versions may not match. Check https://pytorch.org/get-started/previous-versions/ .{RED_END}")
    else:
        print(f"{RED_START}Warning: Torch or CUDA version information not found {RED_END}")


def main():
    for package, (min_version, recommended_version) in required_versions.items():
        check_version(package, min_version, recommended_version)
    # 可选项，检查torch和cuda版本是否匹配 不做硬性检查
    check_torch_cuda_versions()
    # 如果所有检查都通过，输出版本信息和满足条件的提示
    for package in required_versions:
        print(f"{package.capitalize()} version: {get_version('python' if package == 'python' else package)}")

    print(
        "Congratulations, your environment meets the requirements and you can start using LLaMA-Factory now. For more information, please visit https://github.com/LLaMA-Factory/LLaMA-Factory.")


if __name__ == "__main__":
    main()
