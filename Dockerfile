FROM cnstark/pytorch:2.0.1-py3.9.17-cuda11.8.0-ubuntu20.04

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt && \
    pip install tiktoken && \
    pip install transformers_stream_generator

COPY . /app/

VOLUME [ "/root/.cache/huggingface/", "/app/data", "/app/output" ]
EXPOSE 7860

CMD [ "python", "src/train_web.py" ]
