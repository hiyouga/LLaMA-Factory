FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app/
RUN pip install -e .[deepspeed,metrics,bitsandbytes,qwen]

VOLUME [ "/root/.cache/huggingface/", "/app/data", "/app/output" ]
EXPOSE 7860

CMD [ "python", "src/train_web.py" ]
