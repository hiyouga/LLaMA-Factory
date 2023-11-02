import gradio as gr
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    from gradio.blocks import Block
    from gradio.components import Component
    from llmtuner.webui.engine import Engine


def create_chat_box(
    engine: "Engine",
    visible: Optional[bool] = False
) -> Tuple["Block", "Component", "Component", Dict[str, "Component"]]:
    with gr.Box(visible=visible) as chat_box:
        chatbot = gr.Chatbot()
        history = gr.State([])
        with gr.Row():
            with gr.Column(scale=4):
                system = gr.Textbox(show_label=False)
                query = gr.Textbox(show_label=False, lines=8)
                submit_btn = gr.Button(variant="primary")

            with gr.Column(scale=1):
                clear_btn = gr.Button()
                gen_kwargs = engine.chatter.generating_args
                max_new_tokens = gr.Slider(10, 2048, value=gen_kwargs.max_new_tokens, step=1)
                top_p = gr.Slider(0.01, 1, value=gen_kwargs.top_p, step=0.01)
                temperature = gr.Slider(0.01, 1.5, value=gen_kwargs.temperature, step=0.01)

    submit_btn.click(
        engine.chatter.predict,
        [chatbot, query, history, system, max_new_tokens, top_p, temperature],
        [chatbot, history],
        show_progress=True
    ).then(
        lambda: gr.update(value=""), outputs=[query]
    )

    clear_btn.click(lambda: ([], []), outputs=[chatbot, history], show_progress=True)

    return chat_box, chatbot, history, dict(
        system=system,
        query=query,
        submit_btn=submit_btn,
        clear_btn=clear_btn,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        temperature=temperature
    )
