import gradio as gr
from typing import Optional
from transformers.utils.versions import require_version

from llmtuner.webui.components import (
    create_top,
    create_train_tab,
    create_eval_tab,
    create_infer_tab,
    create_export_tab,
    create_chat_box
)
from llmtuner.webui.common import save_config
from llmtuner.webui.css import CSS
from llmtuner.webui.engine import Engine


require_version("gradio>=3.38.0,<4.0.0", "To fix: pip install \"gradio>=3.38.0,<4.0.0\"")


def create_ui(demo_mode: Optional[bool] = False) -> gr.Blocks:
    engine = Engine(demo_mode=demo_mode, pure_chat=False)

    with gr.Blocks(title="LLaMA Board", css=CSS) as demo:
        if demo_mode:
            gr.HTML(
                "<h1><center>LLaMA Board: A One-stop Web UI for Getting Started with LLaMA Factory</center></h1>"
            )
            gr.HTML(
                "<h3><center>Visit <a href=\"https://github.com/hiyouga/LLaMA-Factory\" target=\"_blank\">"
                "LLaMA Factory</a> for details.</center></h3>"
            )
            gr.DuplicateButton(value="Duplicate Space for private use", elem_classes="duplicate-button")

        engine.manager.all_elems["top"] = create_top()
        lang: "gr.Dropdown" = engine.manager.get_elem_by_name("top.lang")

        with gr.Tab("Train"):
            engine.manager.all_elems["train"] = create_train_tab(engine)

        with gr.Tab("Evaluate & Predict"):
            engine.manager.all_elems["eval"] = create_eval_tab(engine)

        with gr.Tab("Chat"):
            engine.manager.all_elems["infer"] = create_infer_tab(engine)

        if not demo_mode:
            with gr.Tab("Export"):
                engine.manager.all_elems["export"] = create_export_tab(engine)

        demo.load(engine.resume, outputs=engine.manager.list_elems())
        lang.change(engine.change_lang, [lang], engine.manager.list_elems(), queue=False)
        lang.input(save_config, inputs=[lang], queue=False)

    return demo


def create_web_demo() -> gr.Blocks:
    engine = Engine(pure_chat=True)

    with gr.Blocks(title="Web Demo", css=CSS) as demo:
        lang = gr.Dropdown(choices=["en", "zh"])
        engine.manager.all_elems["top"] = dict(lang=lang)

        chat_box, _, _, chat_elems = create_chat_box(engine, visible=True)
        engine.manager.all_elems["infer"] = dict(chat_box=chat_box, **chat_elems)

        demo.load(engine.resume, outputs=engine.manager.list_elems())
        lang.change(engine.change_lang, [lang], engine.manager.list_elems(), queue=False)
        lang.input(save_config, inputs=[lang], queue=False)

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)
