# coding=utf-8

import json
from threading import Thread

import torch
import uvicorn
import datetime
from fastapi import FastAPI, Request
from starlette.responses import StreamingResponse
from transformers import TextIteratorStreamer

from utils import (
    Template,
    load_pretrained,
    prepare_infer_args,
    get_logits_processor
)

app = FastAPI()


@app.get("/hello")
def hello():
    return "hello world!"


def parse_text(text):  # copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = "<pre><code class=\"language-{}\">".format(items[-1])
            else:
                lines[i] = "<br /></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br />" + line
    text = "".join(lines)
    return text


def predict(query, max_length, top_p, temperature, history):
    input_ids = tokenizer([prompt_template.get_prompt(query, history)], return_tensors="pt")["input_ids"]
    input_ids = input_ids.to(model.device)
    gen_kwargs = {
        "input_ids": input_ids,
        "do_sample": True,
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
    response = ''
    for new_text in streamer:
        response += new_text
        print(new_text)
        s = parse_text(response)
        yield s[-1]


@app.post("/chat")
async def chat(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    messages = json_post_list.get("messages")[:-1]
    history = []
    if len(messages) > 2:
        for i in range(0, len(messages) - 1, 2):
            history.append([messages[i]['content'], messages[i + 1]['content']])
    prompt = messages[-1]['content']
    model = json_post_list.get("model")  # keep this for future use
    return StreamingResponse(predict(query=prompt, max_length=512, top_p=0.7, temperature=0.95, history=history),
                             media_type="text/event-stream")


if __name__ == "__main__":
    model_args, data_args, finetuning_args, generating_args = prepare_infer_args()
    model, tokenizer = load_pretrained(model_args, finetuning_args)
    prompt_template = Template(data_args.prompt_template)
    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
