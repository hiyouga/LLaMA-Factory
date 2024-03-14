FROM python:3.10

WORKDIR /usr/src/app
RUN git clone https://github.com/EleutherAI/lm-evaluation-harness.git

WORKDIR /usr/src/app/lm-evaluation-harness

RUN pip install -e .
RUN pip install protobuf sentencepiece accelerate
RUN pip install -i https://pypi.org/simple/ bitsandbytes

ENV MODEL_TYPE=hf
ENV MODEL_DIR=mistralai/Mistral-7B-Instruct-v0.2
ENV NUM_FEWSHOTS=0
ENV TASKS=arc_easy
ENV BATCH_SIZE=16
ENV OUTPUT_PATH=$MODEL_DIR/eval_output

CMD ["/bin/sh", "-c", "lm_eval --model $MODEL_TYPE --model_args pretrained=$MODEL_DIR --tasks $TASKS --num_fewshot $NUM_FEWSHOTS --device cuda:0 --batch_size auto:$BATCH_SIZE --output_path $OUTPUT_PATH"]
