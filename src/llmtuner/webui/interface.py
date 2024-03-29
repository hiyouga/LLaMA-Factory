import json
from typing import Any, Dict, List, Tuple

import gradio as gr
from gradio.components import Component
from transformers.utils.versions import require_version

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


require_version("gradio>=3.38.0,<4.0.0", 'To fix: pip install "gradio>=3.38.0,<4.0.0"')

class Params_save_manager:
    def __init__(self, engine:"Engine") -> None:
        self.engine = engine

    def save(self,*data: Dict[Component, Any]) -> None:
        config = dict()
        names = self._get_components()[0]
        for index, value in enumerate(data):
            config.update({names[index]:value})
        with open('config.json','w') as file:
            file.write(json.dumps(config))

    def load(self) -> Dict[Component, Dict[str, Any]]:
        components = self._get_components()
        with open('config.json','r') as file:
            config = json.loads(file.read())
        return {
            component: gr.update(value=config[name])
            for name, component in zip(components[0],components[1])
        }

    def _get_components(self) -> Tuple[List[str],List["Component"]]:
        names = list()
        components = list()

        for name, component in self.engine.manager.all_elems["train"].items():
            if (isinstance(component,(gr.components.dropdown.Dropdown,
                                      gr.components.textbox.Textbox,
                                      gr.components.checkbox.Checkbox,
                                      gr.components.slider.Slider))):
                names.append(name)
                components.append(component)

        return (names, components)


def create_ui(demo_mode: bool = False) -> gr.Blocks:
    engine = Engine(demo_mode=demo_mode, pure_chat=False)
    config_manager = Params_save_manager(engine)

    with gr.Blocks(title="LLaMA Board", css=CSS) as demo:
        if demo_mode:
            gr.HTML("<h1><center>LLaMA Board: A One-stop Web UI for Getting Started with LLaMA Factory</center></h1>")
            gr.HTML(
                '<h3><center>Visit <a href="https://github.com/hiyouga/LLaMA-Factory" target="_blank">'
                "LLaMA Factory</a> for details.</center></h3>"
            )
            gr.DuplicateButton(value="Duplicate Space for private use", elem_classes="duplicate-button")

        lang, engine.manager.all_elems["top"] = create_top()

        with gr.Tab("Train"):
            engine.manager.all_elems["train"] = create_train_tab(engine)

            engine.manager.all_elems["train"]["save_param_btn"].click(config_manager.save,config_manager._get_components()[1],queue=False)
            engine.manager.all_elems["train"]["load_param_btn"].click(config_manager.load,outputs=config_manager._get_components()[1],queue=False)

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
    demo.launch(server_name="0.0.0.0", share=False, inbrowser=True)
