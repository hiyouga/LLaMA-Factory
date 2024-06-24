# Using ubuntu 22.04 images with cann 8.0.rc1
# More options can be found at https://hub.docker.com/r/cosdt/cann/tags
FROM cosdt/cann:8.0.rc1-910b-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Define installation arguments
ARG INSTALL_DEEPSPEED=false
ARG PIP_INDEX=https://pypi.org/simple

# Set the working directory
WORKDIR /app/LLaMA-Factory

RUN cd /app && \
    git config --global http.version HTTP/1.1 && \
    git clone https://github.com/hiyouga/LLaMA-Factory.git && \
    cd /app/LLaMA-Factory

RUN pip config set global.index-url $PIP_INDEX
RUN python3 -m pip install --upgrade pip

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
EXPOSE 7860

# Expose port 8000 for the API service
EXPOSE 8000

# Launch LLaMA Board
CMD [ "llamafactory-cli", "webui" ]
