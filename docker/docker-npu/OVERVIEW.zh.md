# 面向昇腾 NPU 的 LLaMA Factory 镜像

LLaMA Factory 昇腾 NPU 镜像面向华为昇腾 Atlas NPU，提供可直接用于大语言模型和多模态模型微调、评测与服务部署的运行环境。镜像基于昇腾 CANN 容器镜像构建，预装 LLaMA Factory、Python、PyTorch、torch-npu、Triton Ascend、DeepSpeed 和 LLaMA Factory 评测依赖。

安装方法和问题排查请参考 [LLaMA Factory NPU 安装及配置文档](https://llamafactory.readthedocs.io/zh-cn/latest/multibackend/npu/npu_installation.html)。

## 快速参考

- 镜像仓库：
  - `docker.io/hiyouga/llamafactory`
  - `quay.io/ascend/llamafactory`
- Dockerfile：`docker/docker-npu/Dockerfile`
- Docker Compose 文件：`docker/docker-npu/docker-compose.yml`
- 默认基础镜像：`quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.11`
- 支持的加速器：昇腾 A2、A3
- 支持的容器操作系统：Ubuntu 22.04、openEuler 24.03
- 目标 CPU 架构：`linux/amd64`、`linux/arm64`
- 对外端口：
  - `7860`：LLaMA Board Web UI
  - `8000`：API 服务
- 昇腾环境脚本：`/usr/local/Ascend/ascend-toolkit/set_env.sh`

当前提供以下镜像组合：

| 加速器 | 容器操作系统 | CANN 基础镜像 |
| --- | --- | --- |
| A2 | Ubuntu 22.04 | `quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.11` |
| A3 | Ubuntu 22.04 | `quay.io/ascend/cann:9.0.0-a3-ubuntu22.04-py3.11` |
| A2 | openEuler 24.03 | `quay.io/ascend/cann:9.0.0-910b-openeuler24.03-py3.11` |
| A3 | openEuler 24.03 | `quay.io/ascend/cann:9.0.0-a3-openeuler24.03-py3.11` |

## 镜像介绍

该镜像用于运行 LLaMA Factory 支持的昇腾 NPU 训练、微调、评测、Web UI 和 API 服务，主要包含以下组件：

| 组件 | 版本或来源 |
| --- | --- |
| CANN | 继承自所选 CANN 9.0.0 基础镜像 |
| Python | Python 3.11，继承自基础镜像 |
| PyTorch | `2.7.1` |
| torch-npu | `2.7.1.post4` |
| torchvision | `0.22.1` |
| torchaudio | `2.7.1` |
| Triton Ascend | `3.2.1` |
| DeepSpeed | `>=0.10.0,<=0.18.4` |
| LLaMA Factory | 从构建上下文中的仓库源码安装 |

镜像不包含模型权重和数据集。请通过目录挂载或运行时下载的方式单独提供，并遵守对应的许可证和使用要求。

## 镜像 Tag 说明与 Dockerfile 归档路径

镜像使用以下 tag 格式：

```text
<llamafactory版本>-cann<CANN版本>-torch_npu<torch-npu版本>-<加速器>-<操作系统>-<Python版本>
```

| 字段 | 示例 | 说明 |
| --- | --- | --- |
| `llamafactory版本` | `latest` 或 `0.9.6` | 非 release 构建使用 `latest`，release 构建使用 LLaMA Factory 版本号 |
| `CANN版本` | `9.0.0` | 从 CANN 基础镜像 tag 中提取 |
| `torch-npu版本` | `2.7.1` | 从 `requirements/npu.txt` 中提取，镜像 tag 不包含 `.post4` 等后缀 |
| `加速器` | `A2` 或 `A3` | 当前镜像所适配的昇腾硬件代际 |
| `操作系统` | `ubuntu` 或 `openeuler` | 容器内操作系统类型 |
| `Python版本` | `py3.11` | 从 CANN 基础镜像 tag 中提取 |

示例：

```text
latest-cann9.0.0-torch_npu2.7.1-A2-ubuntu-py3.11
latest-cann9.0.0-torch_npu2.7.1-A3-openeuler-py3.11
0.9.6-cann9.0.0-torch_npu2.7.1-A3-ubuntu-py3.11
```

CPU 架构不写入 tag。发布镜像配置为多架构镜像，Docker 拉取时会根据宿主机自动选择 `linux/amd64` 或 `linux/arm64` 版本。

Dockerfile 和用于镜像分发的概述文件在同一目录归档：

```text
docker/docker-npu/
├── Dockerfile
├── OVERVIEW.md
├── OVERVIEW.zh.md
└── docker-compose.yml
```

## 快速开始

### 前置条件

启动容器前需要：

1. 在宿主机安装与镜像内 CANN 版本兼容的昇腾驱动和固件。
2. 确认宿主机执行 `npu-smi info` 可以正常识别 NPU。
3. 安装 Docker，并确保当前用户有权访问所需的昇腾设备节点和驱动文件。

驱动、固件、CANN、torch-npu 与目标昇腾硬件需要保持兼容。

### 拉取并运行镜像

以下示例使用一张 NPU 启动最新的 A2 Ubuntu 镜像。请根据实际环境修改镜像 tag 和 `/dev/davinci0`。

```bash
export IMAGE=quay.io/ascend/llamafactory:latest-cann9.0.0-torch_npu2.7.1-A2-ubuntu-py3.11

docker pull "$IMAGE"

docker run --rm -it \
  --name llamafactory-npu \
  --ipc=host \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 7860:7860 \
  -p 8000:8000 \
  "$IMAGE" \
  bash
```

部分驱动环境中的 `npu-smi` 位于 `/usr/local/sbin/npu-smi`，此时需要调整挂载源路径。使用多张 NPU 时，继续追加 `--device=/dev/davinci<N>` 参数。

进入容器后验证运行环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())"
llamafactory-cli help
```

需要使用 LLaMA Board 时执行：

```bash
llamafactory-cli webui
```

### 本地构建

在仓库根目录执行构建。以下示例构建 A3 openEuler 镜像：

```bash
docker build \
  -f ./docker/docker-npu/Dockerfile \
  --build-arg BASE_IMAGE=quay.io/ascend/cann:9.0.0-a3-openeuler24.03-py3.11 \
  --build-arg PIP_INDEX=https://pypi.org/simple \
  -t llamafactory:npu-a3-openeuler \
  .
```

可用构建参数：

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `BASE_IMAGE` | A2 Ubuntu CANN 9.0.0 镜像 | 选择加速器和容器操作系统组合 |
| `PIP_INDEX` | `https://pypi.org/simple` | 指定 Python 软件包索引 |
| `PYTORCH_INDEX` | `https://download.pytorch.org/whl/cpu` | 指定配合 torch-npu 使用的 PyTorch wheel 索引 |
| `HTTP_PROXY` | 空 | 构建期间可选的 HTTP/HTTPS 代理 |

也可以通过 Docker Compose 构建并启动各个组合：

```bash
cd docker/docker-npu

# A2 + Ubuntu
docker compose up -d llamafactory-a2-ubuntu

# A3 + Ubuntu
docker compose --profile a3 up -d llamafactory-a3-ubuntu

# A2 + openEuler
docker compose --profile openeuler up -d llamafactory-a2-openeuler

# A3 + openEuler
docker compose --profile a3-openeuler up -d llamafactory-a3-openeuler
```

### 二次开发

交互式开发时，可以将本地源码挂载到容器中，并在容器内以 editable 模式重新安装：

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 同时添加前述昇腾 --device 和驱动目录挂载参数。
docker run --rm -it \
  --ipc=host \
  -v "$PWD:/workspace/LLaMA-Factory" \
  -w /workspace/LLaMA-Factory \
  "$IMAGE" \
  bash

pip install -e . --no-build-isolation
```

需要可复现的派生镜像时，可以新建独立 Dockerfile：

```dockerfile
FROM quay.io/ascend/llamafactory:latest-cann9.0.0-torch_npu2.7.1-A2-ubuntu-py3.11

COPY requirements-extension.txt /tmp/requirements-extension.txt
RUN pip install --no-cache-dir -r /tmp/requirements-extension.txt

COPY . /workspace/application
WORKDIR /workspace/application
```

运行派生镜像时仍需传入昇腾设备和驱动挂载参数，不应将设备访问配置固化到镜像中。

## 硬件支持与兼容性说明

- A2 镜像使用标记为 `910b` 的 CANN 基础镜像，A3 镜像使用标记为 `a3` 的 CANN 基础镜像。
- 镜像构建目标同时包含 x86-64（`linux/amd64`）和 AArch64（`linux/arm64`）宿主机。CPU 架构与加速器属于 A2 还是 A3 无关。
- Ubuntu 22.04 和 openEuler 24.03 指容器内部的操作系统。
- 当前依赖基线将 PyTorch `2.7.1` 与 torch-npu `2.7.1.post4` 配套使用。单独升级其中一个软件包可能破坏兼容性。
- 生产环境建议使用固定 release tag，以确保部署可复现；定时构建可能更新 `latest` tag。
- `latest-npu-a2` 等旧式短 tag 没有体现 CANN、torch-npu、操作系统和 Python 版本，建议迁移到本文所述的完整 tag。
- 正式部署前，请验证具体驱动、固件、CANN 和 SoC 组合的兼容性。

## 许可证与免责声明

LLaMA Factory 基于 [Apache License 2.0](../../LICENSE) 发布。

昇腾 CANN、torch-npu、Triton Ascend、DeepSpeed、基础操作系统软件包、模型权重、数据集和其他第三方组件分别受其自身许可证与条款约束。LLaMA Factory 的许可证不会替代或覆盖这些条款。

本镜像按“原样”提供，不附带任何明示或暗示的保证。用户需要自行验证软硬件兼容性、保障容器及运行配置的安全、遵守适用的许可证和法律，并在训练、评测或部署前审查模型与数据集的使用条款。
