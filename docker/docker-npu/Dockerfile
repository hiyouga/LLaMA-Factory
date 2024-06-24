# Use the Ubuntu 22.04 image with CANN 8.0.rc1
# More versions can be found at https://hub.docker.com/r/cosdt/cann/tags
FROM cosdt/cann:8.0.rc1-910b-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Define installation arguments
ARG INSTALL_DEEPSPEED=false
ARG PIP_INDEX=https://pypi.org/simple

# Set the working directory
WORKDIR /app

# Install the requirements
COPY requirements.txt /app
RUN pip config set global.index-url $PIP_INDEX
RUN pip config set global.extra-index-url $PIP_INDEX
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

# Copy the rest of the application into the image
COPY . /app

# Install the LLaMA Factory
RUN EXTRA_PACKAGES="torch-npu,metrics"; \
    if [ "$INSTALL_DEEPSPEED" = "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},deepspeed"; \
    fi; \
    pip install -e .[$EXTRA_PACKAGES] && \
    pip uninstall -y transformer-engine flash-attn

# Set up volumes
VOLUME [ "/root/.cache/huggingface/", "/app/data", "/app/output" ]

# Expose port 7860 for the LLaMA Board
ENV GRADIO_SERVER_PORT 7860
EXPOSE 7860

# Expose port 8000 for the API service
ENV API_PORT 8000
EXPOSE 8000
