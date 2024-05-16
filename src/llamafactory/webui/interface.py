import os

from ..extras.packages import is_gradio_available
from .common import save_config
from .components import (
    create_chat_box,
    create_eval_tab,
    create_export_tab,
    create_infer_tab,
    create_top,
    create_train_tab,
)
from .css import CSS
from .engine import Engine


if is_gradio_available():
    import gradio as gr


def create_ui(demo_mode: bool = False) -> gr.Blocks:
    engine = Engine(demo_mode=demo_mode, pure_chat=False)

    with gr.Blocks(title="LLaMA Board", css=CSS) as demo:
        if demo_mode:
            gr.HTML("<h1><center>LLaMA Board: A One-stop Web UI for Getting Started with LLaMA Factory</center></h1>")
            gr.HTML(
                '<h3><center>Visit <a href="https://github.com/hiyouga/LLaMA-Factory" target="_blank">'
                "LLaMA Factory</a> for details.</center></h3>"
            )
            gr.DuplicateButton(value="Duplicate Space for private use", elem_classes="duplicate-button")

        engine.manager.add_elems("top", create_top())
        lang: "gr.Dropdown" = engine.manager.get_elem_by_id("top.lang")

        with gr.Tab("Train"):
            engine.manager.add_elems("train", create_train_tab(engine))

        with gr.Tab("Evaluate & Predict"):
            engine.manager.add_elems("eval", create_eval_tab(engine))

        with gr.Tab("Chat"):
            engine.manager.add_elems("infer", create_infer_tab(engine))

        if not demo_mode:
            with gr.Tab("Export"):
                engine.manager.add_elems("export", create_export_tab(engine))

        demo.load(engine.resume, outputs=engine.manager.get_elem_list(), concurrency_limit=None)
        lang.change(engine.change_lang, [lang], engine.manager.get_elem_list(), queue=False)
        lang.input(save_config, inputs=[lang], queue=False)

    return demo


def create_web_demo() -> gr.Blocks:
    engine = Engine(pure_chat=True)

    with gr.Blocks(title="Web Demo", css=CSS) as demo:
        lang = gr.Dropdown(choices=["en", "zh"])
        engine.manager.add_elems("top", dict(lang=lang))

        _, _, chat_elems = create_chat_box(engine, visible=True)
        engine.manager.add_elems("infer", chat_elems)

        demo.load(engine.resume, outputs=engine.manager.get_elem_list(), concurrency_limit=None)
        lang.change(engine.change_lang, [lang], engine.manager.get_elem_list(), queue=False)
        lang.input(save_config, inputs=[lang], queue=False)

    return demo


def run_web_ui() -> None:
    gradio_share = os.environ.get("GRADIO_SHARE", "0").lower() in ["true", "1"]
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    create_ui().queue().launch(share=gradio_share, server_name=server_name, inbrowser=True)


def run_web_demo() -> None:
    gradio_share = os.environ.get("GRADIO_SHARE", "0").lower() in ["true", "1"]
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    create_web_demo().queue().launch(share=gradio_share, server_name=server_name, inbrowser=True)
