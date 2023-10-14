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
from llmtuner.webui.common import load_config, save_config
from llmtuner.webui.css import CSS
from llmtuner.webui.engine import Engine


require_version("gradio>=3.36.0", "To fix: pip install gradio>=3.36.0")


def create_ui() -> gr.Blocks:
    engine = Engine(init_chat=False)

    with gr.Blocks(title="Web Tuner", css=CSS) as demo:
        engine.manager.all_elems["top"] = create_top(engine)

        with gr.Tab("Train"):
            engine.manager.all_elems["train"] = create_train_tab(engine)

        with gr.Tab("Evaluate"):
            engine.manager.all_elems["eval"] = create_eval_tab(engine)

        with gr.Tab("Chat"):
            engine.manager.all_elems["infer"] = create_infer_tab(engine)

        with gr.Tab("Export"):
            engine.manager.all_elems["export"] = create_export_tab(engine)

        demo.load(engine.resume, [engine.manager.get_elem("top.config")], engine.manager.list_elems())

    return demo


def create_web_demo() -> gr.Blocks:
    engine = Engine(init_chat=True)

    with gr.Blocks(title="Web Demo", css=CSS) as demo:
        lang = gr.Dropdown(choices=["en", "zh"])
        config = gr.State(value=load_config())
        lang.change(
            engine.change_lang, [lang], engine.manager.list_elems(), queue=False
        ).then(
            save_config, inputs=[config, lang]
        )

        engine.manager.all_elems["top"] = dict(lang=lang)

        _, _, _, engine.manager.all_elems["infer"] = create_chat_box(engine, visible=True)

        demo.load(engine.resume, [config], engine.manager.list_elems())

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)
