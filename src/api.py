# coding=utf-8
# Chat with LLaMA in API mode.
# Usage: python cli_demo.py --model_name_or_path path_to_model --checkpoint_dir path_to_checkpoint

# Call:
# curl --location 'http://127.0.0.1:8000' \
# --header 'Content-Type: application/json' \
# --data '{"prompt": "Hello there!","history": []}'

# Response:
# {
# "response":"'I am a second year student at the University of British Columbia, in Vancouver.\\nMy major
# is Computer Science and my minor (double degree) area was Mathematics/Statistics with an emphasis on Operations
# Research & Management Sciences which means that when it comes to solving problems using computers or any kind data
# analysis; whether its from businesses , governments etc., i can help you out :) .'",
# "history":"[('Hello there!',
# 'I am a second year student at the University of British Columbia, in Vancouver.\\nMy major is Computer Science and
# my minor (double degree) area was Mathematics/Statistics with an emphasis on Operations Research & Management
# Sciences which means that when it comes to solving problems using computers or any kind data analysis; whether its
# from businesses , governments etc., i can help you out :) .')]",
# "status":200,
# "time":"2023-05-30 06:57:38" }

import datetime
import torch
from utils import ModelArguments, auto_configure_device_map, load_pretrained
from transformers import HfArgumentParser
import json
import uvicorn
from fastapi import FastAPI, Request

DEVICE = "cuda"


def torch_gc():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for device_id in range(num_gpus):
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer

    # Parse the request JSON
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')

    # Tokenize the input prompt
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = inputs.to(model.device)

    # Generation arguments
    gen_kwargs = {
        "do_sample": True,
        "top_p": 0.9,
        "top_k": 40,
        "temperature": 0.7,
        "num_beams": 1,
        "max_new_tokens": 256,
        "repetition_penalty": 1.5
    }

    # Generate response
    with torch.no_grad():
        generation_output = model.generate(**inputs, **gen_kwargs)
    outputs = generation_output.tolist()[0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(outputs, skip_special_tokens=True)

    # Update history
    history = history + [(prompt, response)]

    # Prepare response
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": repr(response),
        "history": repr(history),
        "status": 200,
        "time": time
    }

    # Log and clean up
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()

    return answer


if __name__ == "__main__":
    parser = HfArgumentParser(ModelArguments)
    model_args, = parser.parse_args_into_dataclasses()
    model, tokenizer = load_pretrained(model_args)

    if torch.cuda.device_count() > 1:
        from accelerate import dispatch_model

        device_map = auto_configure_device_map(torch.cuda.device_count())
        model = dispatch_model(model, device_map)
    else:
        model = model.cuda()

    model.eval()

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
