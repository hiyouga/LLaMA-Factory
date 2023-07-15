# coding=utf-8
# Implements user interface in browser for fine-tuned models.
# Usage: python web_demo.py --model_name_or_path path_to_model --checkpoint_dir path_to_checkpoint

import gradio as gr
from threading import Thread
from transformers import TextIteratorStreamer
from transformers.utils.versions import require_version

from llmtuner import Template, get_infer_args, load_model_and_tokenizer, get_logits_processor


require_version("gradio>=3.30.0", "To fix: pip install gradio>=3.30.0")


model_args, data_args, finetuning_args, generating_args = get_infer_args()
model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args)

prompt_template = Template(data_args.prompt_template)
source_prefix = data_args.source_prefix if data_args.source_prefix else ""


def predict(query, chatbot, max_new_tokens, top_p, temperature, history):
    chatbot.append((query, ""))

    input_ids = tokenizer([prompt_template.get_prompt(query, history, source_prefix)], return_tensors="pt")["input_ids"]
    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = generating_args.to_dict()
    gen_kwargs.update({
        "input_ids": input_ids,
        "top_p": top_p,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "logits_processor": get_logits_processor(),
        "streamer": streamer
    })

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    response = ""
    for new_text in streamer:
        response += new_text
        new_history = history + [(query, response)]
        chatbot[-1] = (query, response)
        yield chatbot, new_history


def reset_user_input():
    return gr.update(value="")


def reset_state():
    return [], []


with gr.Blocks() as demo:

    gr.HTML("""
    <h1 align="center">
        <a href="https://github.com/hiyouga/LLaMA-Efficient-Tuning" target="_blank">
            LLaMA Efficient Tuning
        </a>
    </h1>
    """)

    chatbot = gr.Chatbot()

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")

        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_new_tokens = gr.Slider(10, 2048, value=generating_args.max_new_tokens, step=1.0,
                                       label="Maximum new tokens", interactive=True)
            top_p = gr.Slider(0.01, 1, value=generating_args.top_p, step=0.01,
                              label="Top P", interactive=True)
            temperature = gr.Slider(0.01, 1.5, value=generating_args.temperature, step=0.01,
                                    label="Temperature", interactive=True)

    history = gr.State([])

    submitBtn.click(predict, [user_input, chatbot, max_new_tokens, top_p, temperature, history], [chatbot, history], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(server_name="0.0.0.0", share=True, inbrowser=True)
