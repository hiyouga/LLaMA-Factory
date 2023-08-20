import gradio as gr
from transformers.utils.versions import require_version

from llmtuner.webui.components import (
    create_top,
    create_train_tab,
    create_eval_tab,
    create_infer_tab,
    create_export_tab,
    create_chat_box
)
from llmtuner.webui.chat import WebChatModel
from llmtuner.webui.css import CSS
from llmtuner.webui.manager import Manager
from llmtuner.webui.runner import Runner


require_version("gradio>=3.36.0", "To fix: pip install gradio>=3.36.0")


def create_ui() -> gr.Blocks:
    runner = Runner()

    with gr.Blocks(title="Web Tuner", css=CSS) as demo:
        top_elems = create_top()

        with gr.Tab("Train"):
            train_elems = create_train_tab(top_elems, runner)

        with gr.Tab("Evaluate"):
            eval_elems = create_eval_tab(top_elems, runner)

        with gr.Tab("Chat"):
            infer_elems = create_infer_tab(top_elems)

        with gr.Tab("Export"):
            export_elems = create_export_tab(top_elems)

        elem_list = [top_elems, train_elems, eval_elems, infer_elems, export_elems]
        manager = Manager(elem_list)

        demo.load(
            manager.gen_label,
            [top_elems["lang"]],
            [elem for elems in elem_list for elem in elems.values()],
        )

        top_elems["lang"].change(
            manager.gen_label,
            [top_elems["lang"]],
            [elem for elems in elem_list for elem in elems.values()],
            queue=False
        )

    return demo


def create_web_demo() -> gr.Blocks:
    chat_model = WebChatModel(lazy_init=False)

    with gr.Blocks(title="Web Demo", css=CSS) as demo:
        lang = gr.Dropdown(choices=["en", "zh"], value="en")

        _, _, _, chat_elems = create_chat_box(chat_model, visible=True)

        manager = Manager([{"lang": lang}, chat_elems])

        demo.load(manager.gen_label, [lang], [lang] + list(chat_elems.values()))

        lang.select(manager.gen_label, [lang], [lang] + list(chat_elems.values()), queue=False)

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)
