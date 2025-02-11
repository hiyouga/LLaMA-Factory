# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import codecs
import os
from typing import TYPE_CHECKING

import fire
from transformers import AutoTokenizer

from llamafactory.data import get_template_and_fix_tokenizer
if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from llamafactory.data.formatter import SLOTS
    from llamafactory.data.template import Template


def _convert_slots_to_ollama(slots: "SLOTS", tokenizer: "PreTrainedTokenizer", placeholder: str = "content") -> str:
    slot_items = []
    for slot in slots:
        if isinstance(slot, str):
            slot_pieces = slot.split("{{content}}")
            if slot_pieces[0]:
                slot_items.append(slot_pieces[0])
            if len(slot_pieces) > 1:
                slot_items.append(placeholder)
                if slot_pieces[1]:
                    slot_items.append(slot_pieces[1])
        elif isinstance(slot, set):  # do not use {{ eos_token }} since it may be replaced
            if "bos_token" in slot and tokenizer.bos_token_id is not None:
                slot_items.append(tokenizer.bos_token)
            elif "eos_token" in slot and tokenizer.eos_token_id is not None:
                slot_items.append(tokenizer.eos_token)
        elif isinstance(slot, dict):
            raise ValueError("Dict is not supported.")

    return "".join(slot_items)


def _split_round_template(user_template_str: "str", template_obj: "Template", tokenizer: "PreTrainedTokenizer") -> tuple:
    if template_obj.format_separator.apply():
        format_separator = _convert_slots_to_ollama(template_obj.format_separator.apply(), tokenizer)
        round_split_token_list = [tokenizer.eos_token + format_separator, tokenizer.eos_token,
                              format_separator, "{{ .Prompt }}"]
    else:
        round_split_token_list = [tokenizer.eos_token, "{{ .Prompt }}"]

    for round_split_token in round_split_token_list:
        round_split_templates = user_template_str.split(round_split_token)
        if len(round_split_templates) >= 2:
            user_round_template = "".join(round_split_templates[:-1])
            assistant_round_template = round_split_templates[-1]
            return user_round_template + round_split_token, assistant_round_template

    return user_template_str, ""


def convert_template_obj_to_ollama(template_obj: "Template", tokenizer: "PreTrainedTokenizer") -> str:
    ollama_template = ""
    if template_obj.format_system:
        ollama_template += "{{ if .System }}"
        ollama_template += _convert_slots_to_ollama(template_obj.format_system.apply(), tokenizer, "{{ .System }}")
        ollama_template += "{{ end }}"

    user_template = _convert_slots_to_ollama(template_obj.format_user.apply(), tokenizer, "{{ .Prompt }}")
    user_round_template, assistant_round_template = _split_round_template(user_template, template_obj, tokenizer)

    ollama_template += "{{ if .Prompt }}"
    ollama_template += user_round_template
    ollama_template += "{{ end }}"
    ollama_template += assistant_round_template

    ollama_template += _convert_slots_to_ollama(template_obj.format_assistant.apply(), tokenizer, "{{ .Response }}")

    return ollama_template


def export_ollama_modelfile(
    model_name_or_path: str,
    gguf_path: str,
    template: str,
    export_dir: str = "./ollama_model_file"
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    template_obj = get_template_and_fix_tokenizer(tokenizer, name=template)
    ollama_template = convert_template_obj_to_ollama(template_obj, tokenizer)

    if not os.path.exists(export_dir):
        os.mkdir(export_dir)
    with codecs.open(os.path.join(export_dir, "Modelfile"), "w", encoding="utf-8") as outf:
        outf.write("FROM {}".format(gguf_path) + "\n")
        outf.write("TEMPLATE \"\"\"{}\"\"\"".format(ollama_template) + "\n")

        if template_obj.stop_words:
            for stop_word in template_obj.stop_words:
                outf.write("PARAMETER stop \"{}\"".format(stop_word) + "\n")
        elif not template_obj.efficient_eos:
            outf.write("PARAMETER stop \"{}\"".format(tokenizer.eos_token) + "\n")


if __name__ == '__main__':
    fire.Fire(export_ollama_modelfile)
