import gradio as gr
from transformers.utils.versions import require_version

from llmtuner.webui.components import (
    create_top,
    create_sft_tab,
    create_eval_tab,
    create_infer_tab,
    create_export_tab
)
from llmtuner.webui.css import CSS
from llmtuner.webui.manager import Manager
from llmtuner.webui.runner import Runner


require_version("gradio>=3.36.0", "To fix: pip install gradio>=3.36.0")


def create_ui() -> gr.Blocks:
    runner = Runner()

    with gr.Blocks(title="Web Tuner", css=CSS) as demo:
        top_elems = create_top()

        with gr.Tab("SFT"):
            sft_elems = create_sft_tab(top_elems, runner)

        with gr.Tab("Evaluate"):
            eval_elems = create_eval_tab(top_elems, runner)

        with gr.Tab("Chat"):
            infer_elems = create_infer_tab(top_elems)

        with gr.Tab("Export"):
            export_elems = create_export_tab(top_elems)

        elem_list = [top_elems, sft_elems, eval_elems, infer_elems, export_elems]
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
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)
