# Use the NVIDIA official image with PyTorch 2.3.0
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-02.html
FROM nvcr.io/nvidia/pytorch:24.02-py3

# Define installation arguments
ARG INSTALL_BNB=false
ARG INSTALL_VLLM=false
ARG INSTALL_DEEPSPEED=false
ARG PIP_INDEX=https://pypi.org/simple

# Set the working directory
WORKDIR /app

# Install the requirements
COPY requirements.txt /app/
RUN pip config set global.index-url $PIP_INDEX
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

# Copy the rest of the application into the image
COPY . /app/

# Install the LLaMA Factory
RUN EXTRA_PACKAGES="metrics"; \
    if [ "$INSTALL_BNB" = "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},bitsandbytes"; \
    fi; \
    if [ "$INSTALL_VLLM" = "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},vllm"; \
    fi; \
    if [ "$INSTALL_DEEPSPEED" = "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},deepspeed"; \
    fi; \
    pip install -e .[$EXTRA_PACKAGES] && \
    pip uninstall -y transformer-engine flash-attn

# Set up volumes
VOLUME [ "/root/.cache/huggingface/", "/app/data", "/app/output" ]

# Expose port 7860 for the LLaMA Board
EXPOSE 7860

# Expose port 8000 for the API service
EXPOSE 8000

CMD [ "llamafactory-cli", "webui" ]
