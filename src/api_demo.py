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
import torch
import uvicorn
import datetime
from fastapi import FastAPI, Request

from utils import (
    Template,
    load_pretrained,
    prepare_infer_args,
    get_logits_processor
)


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
    global model, tokenizer, prompt_template, generating_args

    # Parse the request JSON
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get("prompt")
    history = json_post_list.get("history")
    max_new_tokens = json_post_list.get("max_new_tokens", None)
    top_p = json_post_list.get("top_p", None)
    temperature = json_post_list.get("temperature", None)

    # Tokenize the input prompt
    input_ids = tokenizer([prompt_template.get_prompt(prompt, history)], return_tensors="pt")["input_ids"]
    input_ids = input_ids.to(model.device)

    # Generation arguments
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["input_ids"] = input_ids
    gen_kwargs["logits_processor"] = get_logits_processor()
    gen_kwargs["max_new_tokens"] = max_new_tokens if max_new_tokens else gen_kwargs["max_new_tokens"]
    gen_kwargs["top_p"] = top_p if top_p else gen_kwargs["top_p"]
    gen_kwargs["temperature"] = temperature if temperature else gen_kwargs["temperature"]

    # Generate response
    with torch.no_grad():
        generation_output = model.generate(**gen_kwargs)
    outputs = generation_output.tolist()[0][len(input_ids[0]):]
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
    log = "[" + time + "] " + "\", prompt:\"" + prompt + "\", response:\"" + repr(response) + "\""
    print(log)
    torch_gc()

    return answer


if __name__ == "__main__":
    model_args, data_args, finetuning_args, generating_args = prepare_infer_args()
    model, tokenizer = load_pretrained(model_args, finetuning_args)
    prompt_template = Template(data_args.prompt_template)

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
