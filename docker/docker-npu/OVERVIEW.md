# LLaMA Factory for Ascend NPU

LLaMA Factory Ascend NPU images provide a ready-to-use environment for fine-tuning, evaluating, and serving large language and multimodal models on Huawei Ascend Atlas NPUs. The images are based on Ascend CANN container images and include LLaMA Factory, Python, PyTorch, torch-npu, Triton Ascend, DeepSpeed, and the metric dependencies used by LLaMA Factory.

For installation and troubleshooting details, see the [English NPU installation guide](https://llamafactory.readthedocs.io/en/latest/multibackend/npu/npu_installation.html).

## Quick Reference

- Image registries:
  - `docker.io/hiyouga/llamafactory`
  - `quay.io/ascend/llamafactory`
- Dockerfile: `docker/docker-npu/Dockerfile`
- Docker Compose file: `docker/docker-npu/docker-compose.yml`
- Default base image: `quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.11`
- Supported accelerators: Ascend A2 and A3
- Supported container operating systems: Ubuntu 22.04 and openEuler 24.03
- Target CPU architectures: `linux/amd64` and `linux/arm64`
- Exposed ports:
  - `7860`: LLaMA Board Web UI
  - `8000`: API service
- Ascend environment script: `/usr/local/Ascend/ascend-toolkit/set_env.sh`

The current image variants are:

| Accelerator | Container OS | CANN base image |
| --- | --- | --- |
| A2 | Ubuntu 22.04 | `quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.11` |
| A3 | Ubuntu 22.04 | `quay.io/ascend/cann:9.0.0-a3-ubuntu22.04-py3.11` |
| A2 | openEuler 24.03 | `quay.io/ascend/cann:9.0.0-910b-openeuler24.03-py3.11` |
| A3 | openEuler 24.03 | `quay.io/ascend/cann:9.0.0-a3-openeuler24.03-py3.11` |

## Image Contents and Intended Use

The image is intended for Ascend NPU training, fine-tuning, evaluation, Web UI, and API workflows supported by LLaMA Factory. It installs the following core components:

| Component | Version or source |
| --- | --- |
| CANN | Inherited from the selected CANN 9.0.0 base image |
| Python | Python 3.11, inherited from the base image |
| PyTorch | `2.7.1` |
| torch-npu | `2.7.1.post4` |
| torchvision | `0.22.1` |
| torchaudio | `2.7.1` |
| Triton Ascend | `3.2.1` |
| DeepSpeed | `>=0.10.0,<=0.18.4` |
| LLaMA Factory | Installed from the repository build context |

The image does not include model weights or datasets. Mount or download them separately and comply with their respective licenses and acceptable-use requirements.

## Image Tags and Dockerfile Archive

Images use the following tag format:

```text
<llamafactory-version>-cann<cann-version>-torch_npu<torch-npu-version>-<accelerator>-<os>-<python-version>
```

| Field | Example | Description |
| --- | --- | --- |
| `llamafactory-version` | `latest` or `0.9.6` | Non-release builds use `latest`; release builds use the LLaMA Factory version |
| `cann-version` | `9.0.0` | Parsed from the CANN base image tag |
| `torch-npu-version` | `2.7.1` | Parsed from `requirements/npu.txt`; a suffix such as `.post4` is not included in the image tag |
| `accelerator` | `A2` or `A3` | Ascend hardware generation selected for the image |
| `os` | `ubuntu` or `openeuler` | Container operating system family |
| `python-version` | `py3.11` | Parsed from the CANN base image tag |

Examples:

```text
latest-cann9.0.0-torch_npu2.7.1-A2-ubuntu-py3.11
latest-cann9.0.0-torch_npu2.7.1-A3-openeuler-py3.11
0.9.6-cann9.0.0-torch_npu2.7.1-A3-ubuntu-py3.11
```

The CPU architecture is not part of the tag. Published images are configured as multi-platform images, and Docker selects the `linux/amd64` or `linux/arm64` manifest for the host automatically.

The Dockerfile and its distribution overview are archived together at:

```text
docker/docker-npu/
├── Dockerfile
├── OVERVIEW.md
├── OVERVIEW.zh.md
└── docker-compose.yml
```

## Quick Start

### Prerequisites

Before starting a container:

1. Install an Ascend driver and firmware compatible with the CANN version in the image.
2. Verify that `npu-smi info` works on the host.
3. Install Docker with permission to access the required Ascend device nodes and driver files.

Driver, firmware, CANN, torch-npu, and the target Ascend hardware must be mutually compatible.

### Pull and Run

The following example starts the latest A2 Ubuntu image with one NPU. Change the image tag and `/dev/davinci0` as needed.

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

The host path for `npu-smi` may be `/usr/local/sbin/npu-smi` on some driver installations. Adjust the mount source when necessary. Add more `--device=/dev/davinci<N>` options to expose additional NPUs.

Verify the runtime inside the container:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())"
llamafactory-cli help
```

Start LLaMA Board when needed:

```bash
llamafactory-cli webui
```

### Build Locally

Run the build from the repository root. The following example builds the A3 openEuler variant:

```bash
docker build \
  -f ./docker/docker-npu/Dockerfile \
  --build-arg BASE_IMAGE=quay.io/ascend/cann:9.0.0-a3-openeuler24.03-py3.11 \
  --build-arg PIP_INDEX=https://pypi.org/simple \
  -t llamafactory:npu-a3-openeuler \
  .
```

Available build arguments:

| Argument | Default | Purpose |
| --- | --- | --- |
| `BASE_IMAGE` | A2 Ubuntu CANN 9.0.0 image | Selects the accelerator and container OS variant |
| `PIP_INDEX` | `https://pypi.org/simple` | Selects the Python package index |
| `PYTORCH_INDEX` | `https://download.pytorch.org/whl/cpu` | Selects the PyTorch wheel index used with torch-npu |
| `HTTP_PROXY` | Empty | Provides an optional HTTP/HTTPS proxy during the build |

Docker Compose can build and start each supported variant:

```bash
cd docker/docker-npu

# A2 with Ubuntu
docker compose up -d llamafactory-a2-ubuntu

# A3 with Ubuntu
docker compose --profile a3 up -d llamafactory-a3-ubuntu

# A2 with openEuler
docker compose --profile openeuler up -d llamafactory-a2-openeuler

# A3 with openEuler
docker compose --profile a3-openeuler up -d llamafactory-a3-openeuler
```

### Extend or Develop from the Image

For interactive development, mount a local checkout and reinstall it in editable mode inside the container:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# Add the same Ascend --device and driver mount options shown above.
docker run --rm -it \
  --ipc=host \
  -v "$PWD:/workspace/LLaMA-Factory" \
  -w /workspace/LLaMA-Factory \
  "$IMAGE" \
  bash

pip install -e . --no-build-isolation
```

For a reproducible derived image, create a separate Dockerfile:

```dockerfile
FROM quay.io/ascend/llamafactory:latest-cann9.0.0-torch_npu2.7.1-A2-ubuntu-py3.11

COPY requirements-extension.txt /tmp/requirements-extension.txt
RUN pip install --no-cache-dir -r /tmp/requirements-extension.txt

COPY . /workspace/application
WORKDIR /workspace/application
```

Pass Ascend devices and driver mounts when running the derived image; device access should not be embedded in the image itself.

## Hardware Support and Compatibility Notes

- A2 images use the `910b` CANN base image; A3 images use the `a3` CANN base image.
- The image build targets both x86-64 (`linux/amd64`) and AArch64 (`linux/arm64`) hosts. This CPU architecture is independent of whether the accelerator is A2 or A3.
- Ubuntu 22.04 and openEuler 24.03 refer to the operating system inside the container.
- The current dependency baseline aligns PyTorch `2.7.1` with torch-npu `2.7.1.post4`. Upgrading either package independently may break compatibility.
- Use a fixed release tag for reproducible production deployments. The `latest` tag can change after scheduled builds.
- Legacy short tags such as `latest-npu-a2` do not encode the CANN, torch-npu, operating system, or Python versions. Prefer the full tag format documented above.
- Validate the exact driver, firmware, CANN, and SoC combination before production deployment.

## License and Disclaimer

LLaMA Factory is distributed under the [Apache License 2.0](../../LICENSE).

Ascend CANN, torch-npu, Triton Ascend, DeepSpeed, base operating-system packages, model weights, datasets, and other third-party components are governed by their respective licenses and terms. The LLaMA Factory license does not replace or override those terms.

The image is provided on an "AS IS" basis, without warranties or conditions of any kind. Users are responsible for validating hardware and software compatibility, securing the container and its runtime configuration, complying with applicable licenses and laws, and reviewing model and dataset terms before training, evaluation, or deployment.
