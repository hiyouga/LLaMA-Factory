# https://hub.docker.com/r/rocm/pytorch/tags
ARG BASE_IMAGE=rocm/pytorch:rocm6.4.1_ubuntu22.04_py3.10_pytorch_release_2.6.0
FROM ${BASE_IMAGE}

# Installation arguments
ARG PIP_INDEX=https://pypi.org/simple
ARG EXTRAS=metrics
ARG INSTALL_FLASHATTN=false
ARG HTTP_PROXY=""
ARG PYTORCH_INDEX=https://download.pytorch.org/whl/rocm6.3

# Define environments
ENV MAX_JOBS=16
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_OPTIONS=""
ENV PIP_ROOT_USER_ACTION=ignore
ENV http_proxy="${HTTP_PROXY}"
ENV https_proxy="${HTTP_PROXY}"

# Use Bash instead of default /bin/sh
SHELL ["/bin/bash", "-c"]

# Set the working directory
WORKDIR /app

# Change pip source
RUN pip config set global.index-url "${PIP_INDEX}" && \
    pip config set global.extra-index-url "${PIP_INDEX}" && \
    pip install --no-cache-dir --upgrade pip packaging wheel setuptools

# Reinstall pytorch rocm
RUN pip uninstall -y torch torchvision torchaudio && \
    pip install --no-cache-dir --pre torch torchvision torchaudio --index-url "${PYTORCH_INDEX}"

# Install the requirements
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application into the image
COPY . /app

# Install LLaMA Factory
RUN pip install --no-cache-dir -e ".[${EXTRAS}]" --no-build-isolation

# Rebuild flash attention
RUN if [ "${INSTALL_FLASHATTN}" == "true" ]; then \
        pip uninstall -y ninja && \
        pip install --no-cache-dir ninja && \
        pip install --no-cache-dir flash-attn --no-build-isolation; \
    fi

# Set up volumes
# VOLUME [ "/root/.cache/huggingface", "/app/shared_data", "/app/output" ]

# Expose port 7860 for LLaMA Board
ENV GRADIO_SERVER_PORT=7860
EXPOSE 7860

# Expose port 8000 for API service
ENV API_PORT=8000
EXPOSE 8000

# unset proxy
ENV http_proxy=
ENV https_proxy=

# Set no_proxy environment variable
ENV no_proxy="localhost, 127.0.0.1, ::1"

# fix pydantic version
RUN pip install pydantic==2.10.6

# Reset pip config
RUN pip config unset global.index-url && \
    pip config unset global.extra-index-url
