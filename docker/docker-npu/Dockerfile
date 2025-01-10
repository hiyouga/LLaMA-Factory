# Use the Ubuntu 22.04 image with CANN 8.0.rc1
# More versions can be found at https://hub.docker.com/r/ascendai/cann/tags
# FROM ascendai/cann:8.0.rc1-910-ubuntu22.04-py3.8
FROM ascendai/cann:8.0.rc1-910b-ubuntu22.04-py3.8
# FROM ascendai/cann:8.0.rc1-910-openeuler22.03-py3.8
# FROM ascendai/cann:8.0.rc1-910b-openeuler22.03-py3.8

# Define environments
ENV DEBIAN_FRONTEND=noninteractive

# Define installation arguments
ARG INSTALL_DEEPSPEED=false
ARG PIP_INDEX=https://pypi.org/simple
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG HTTP_PROXY=

# Set the working directory
WORKDIR /app

# Set http proxy
RUN if [ -n "$HTTP_PROXY" ]; then \
        echo "Configuring proxy..."; \
        export http_proxy=$HTTP_PROXY; \
        export https_proxy=$HTTP_PROXY; \
    fi

# Install the requirements
COPY requirements.txt /app
RUN pip config set global.index-url "$PIP_INDEX" && \
    pip config set global.extra-index-url "$TORCH_INDEX" && \
    python -m pip install --upgrade pip && \
    if [ -n "$HTTP_PROXY" ]; then \
        python -m pip install --proxy=$HTTP_PROXY -r requirements.txt; \
    else \
        python -m pip install -r requirements.txt; \
    fi

# Copy the rest of the application into the image
COPY . /app

# Install the LLaMA Factory
RUN EXTRA_PACKAGES="torch-npu,metrics"; \
    if [ "$INSTALL_DEEPSPEED" == "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},deepspeed"; \
    fi; \
    if [ -n "$HTTP_PROXY" ]; then \
        pip install --proxy=$HTTP_PROXY -e ".[$EXTRA_PACKAGES]"; \
    else \
        pip install -e ".[$EXTRA_PACKAGES]"; \
    fi

# Unset http proxy
RUN if [ -n "$HTTP_PROXY" ]; then \
        unset http_proxy; \
        unset https_proxy; \
    fi

# Set up volumes
VOLUME [ "/root/.cache/huggingface", "/root/.cache/modelscope", "/app/data", "/app/output" ]

# Expose port 7860 for the LLaMA Board
ENV GRADIO_SERVER_PORT 7860
EXPOSE 7860

# Expose port 8000 for the API service
ENV API_PORT 8000
EXPOSE 8000
