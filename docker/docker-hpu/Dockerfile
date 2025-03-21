# Use the Gaudi v1.19.1 official image with PyTorch 2.5.1
FROM vault.habana.ai/gaudi-docker/1.19.1/ubuntu22.04/habanalabs/pytorch-installer-2.5.1:latest

# Define environments
ENV MAX_JOBS=4
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# Define installation arguments
ARG INSTALL_BNB=false
ARG INSTALL_VLLM=false
ARG INSTALL_DEEPSPEED=false
ARG INSTALL_FLASHATTN=false
ARG INSTALL_LIGER_KERNEL=false
ARG INSTALL_HQQ=false
ARG INSTALL_EETQ=false
ARG PIP_INDEX=https://pypi.org/simple
ARG http_proxy=
ARG https_proxy=
ARG ftp_proxy=
ARG no_proxy="localhost"

# Set the working directory
WORKDIR /app

# Set http proxy
RUN if [ -n "$HTTP_PROXY" ]; then \
        echo "Configuring proxy..."; \
        export http_proxy=$HTTP_PROXY; \
        export https_proxy=$HTTP_PROXY; \
        export no_proxy=${no_proxy:-localhost}; \
    elif [ -n "$http_proxy" ]; then \
        echo "Configuring proxy..."; \
        export http_proxy=$http_proxy; \
        export https_proxy=$http_proxy; \
        export ftp_proxy=$ftp_proxy; \
        export no_proxy=${no_proxy:-localhost}; \
    fi

# Install the requirements
COPY requirements-gaudi.txt /app
RUN pip config set global.index-url "$PIP_INDEX" && \
    pip config set global.extra-index-url "$PIP_INDEX" && \
    python -m pip install --upgrade pip && \
    if [ -n "$HTTP_PROXY" ]; then \
        python -m pip install --proxy=$HTTP_PROXY -r requirements-gaudi.txt; \
    elif [ -n "$http_proxy" ]; then \
        python -m pip install --proxy=$http_proxy -r requirements-gaudi.txt; \
    else \
        python -m pip install -r requirements-gaudi.txt; \
    fi

# Copy the rest of the application into the image
COPY . /app

# Install the LLaMA Factory
RUN EXTRA_PACKAGES="metrics"; \
    if [ "$INSTALL_BNB" == "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},bitsandbytes"; \
    fi; \
    if [ "$INSTALL_VLLM" == "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},vllm"; \
    fi; \
    if [ "$INSTALL_DEEPSPEED" == "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},deepspeed"; \
    fi; \
    if [ -n "$HTTP_PROXY" ]; then \
        pip install --proxy=$HTTP_PROXY -e ".[$EXTRA_PACKAGES]"; \
    elif [ -n "$http_proxy" ]; then \
            pip install --proxy=$http_proxy -e ".[$EXTRA_PACKAGES]"; \
    else \
        pip install -e ".[$EXTRA_PACKAGES]"; \
    fi

# Rebuild flash attention
RUN pip uninstall -y transformer-engine flash-attn && \
    if [ "$INSTALL_FLASHATTN" == "true" ]; then \
        pip uninstall -y ninja && \
        if [ -n "$HTTP_PROXY" ]; then \
            pip install --proxy=$HTTP_PROXY ninja && \
            pip install --proxy=$HTTP_PROXY --no-cache-dir flash-attn --no-build-isolation; \
        elif [ -n "$http_proxy" ]; then \
            pip install --proxy=$http_proxy ninja && \
            pip install --proxy=$http_proxy --no-cache-dir flash-attn --no-build-isolation; \
        else \
            pip install ninja && \
            pip install --no-cache-dir flash-attn --no-build-isolation; \
        fi; \
    fi

# Set up volumes
VOLUME [ "/root/.cache/huggingface", "/root/.cache/modelscope", "/app/data", "/app/output" ]

# Expose port 7860 for the LLaMA Board
ENV GRADIO_SERVER_PORT 7860
EXPOSE 7860

# Expose port 8000 for the API service
ENV API_PORT 8000
EXPOSE 8000
