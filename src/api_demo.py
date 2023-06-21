# coding=utf-8
# Implements API for fine-tuned models.
# Usage: python api_demo.py --model_name_or_path path_to_model --checkpoint_dir path_to_checkpoint

# Request:
# curl http://127.0.0.1:8000 --header 'Content-Type: application/json' --data '{"prompt": "Hello there!", "history": []}'

# Response:
# {
#   "response": "'Hi there!'",
#   "history": "[('Hello there!', 'Hi there!')]",
#   "status": 200,
#   "time": "2000-00-00 00:00:00"
# }


import json
import datetime
import torch
import uvicorn
from threading import Thread
from fastapi import FastAPI, Request
from starlette.responses import StreamingResponse
from transformers import TextIteratorStreamer

from utils import Template, load_pretrained, prepare_infer_args, get_logits_processor


def torch_gc():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for device_id in range(num_gpus):
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


app = FastAPI()


@app.post("/v1/chat/completions")
async def create_item(request: Request):
    global model, tokenizer

    json_post_raw = await request.json()
    prompt = json_post_raw.get("messages")[-1]["content"]
    history = json_post_raw.get("messages")[:-1]
    max_token = json_post_raw.get("max_tokens", None)
    top_p = json_post_raw.get("top_p", None)
    temperature = json_post_raw.get("temperature", None)
    stream = check_stream(json_post_raw.get("stream"))

    if stream:
        generate = predict(prompt, max_token, top_p, temperature, history)
        return StreamingResponse(generate, media_type="text/event-stream")

    input_ids = tokenizer([prompt_template.get_prompt(prompt, history, source_prefix)], return_tensors="pt")[
        "input_ids"]
    input_ids = input_ids.to(model.device)

    gen_kwargs = generating_args.to_dict()
    gen_kwargs["input_ids"] = input_ids
    gen_kwargs["logits_processor"] = get_logits_processor()
    gen_kwargs["max_new_tokens"] = max_token if max_token else gen_kwargs["max_new_tokens"]
    gen_kwargs["top_p"] = top_p if top_p else gen_kwargs["top_p"]
    gen_kwargs["temperature"] = temperature if temperature else gen_kwargs["temperature"]

    generation_output = model.generate(**gen_kwargs)

    outputs = generation_output.tolist()[0][len(input_ids[0]):]
    response = tokenizer.decode(outputs, skip_special_tokens=True)

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": response
                }
            }
        ]
    }

    log = (
            "["
            + time
            + "] "
            + "\", prompt:\""
            + prompt
            + "\", response:\""
            + repr(response)
            + "\""
    )
    print(log)
    torch_gc()

    return answer


def check_stream(stream):
    if isinstance(stream, bool):
        # stream 是布尔类型，直接使用
        stream_value = stream
    else:
        # 不是布尔类型，尝试进行类型转换
        if isinstance(stream, str):
            stream = stream.lower()
            if stream in ["true", "false"]:
                # 使用字符串值转换为布尔值
                stream_value = stream == "true"
            else:
                # 非法的字符串值
                stream_value = False
        else:
            # 非布尔类型也非字符串类型
            stream_value = False
    return stream_value


async def predict(query, max_length, top_p, temperature, history):
    global model, tokenizer
    input_ids = tokenizer([prompt_template.get_prompt(query, history, source_prefix)], return_tensors="pt")["input_ids"]
    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = {
        "input_ids": input_ids,
        "do_sample": generating_args.do_sample,
        "top_p": top_p,
        "temperature": temperature,
        "num_beams": generating_args.num_beams,
        "max_length": max_length,
        "repetition_penalty": generating_args.repetition_penalty,
        "logits_processor": get_logits_processor(),
        "streamer": streamer
    }

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    for new_text in streamer:
        answer = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": new_text
                    }
                }
            ]
        }
        yield "data: " + json.dumps(answer) + '\n\n'


if __name__ == "__main__":
    model_args, data_args, finetuning_args, generating_args = prepare_infer_args()
    model, tokenizer = load_pretrained(model_args, finetuning_args)

    prompt_template = Template(data_args.prompt_template)
    source_prefix = data_args.source_prefix if data_args.source_prefix else ""

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
