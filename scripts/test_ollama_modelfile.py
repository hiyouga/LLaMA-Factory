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

from transformers import AutoTokenizer

from llamafactory.data import get_template_and_fix_tokenizer
from export_ollama_modelfile import convert_template_obj_to_ollama


def test_qwen2_template():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    template = get_template_and_fix_tokenizer(tokenizer, name="qwen")
    ollama_template = convert_template_obj_to_ollama(template, tokenizer)

    assert ollama_template == ("{{ if .System }}<|im_start|>system\n"
                               "{{ .System }}<|im_end|>\n"
                               "{{ end }}{{ if .Prompt }}<|im_start|>user\n"
                               "{{ .Prompt }}<|im_end|>\n"
                               "{{ end }}<|im_start|>assistant\n"
                               "{{ .Response }}<|im_end|>")


def test_yi_template():
    tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-1.5-9B-Chat")
    template = get_template_and_fix_tokenizer(tokenizer, name="yi")
    ollama_template = convert_template_obj_to_ollama(template, tokenizer)

    assert ollama_template == ("{{ if .System }}<|im_start|>system\n"
                               "{{ .System }}<|im_end|>\n"
                               "{{ end }}{{ if .Prompt }}<|im_start|>user\n"
                               "{{ .Prompt }}<|im_end|>\n"
                               "{{ end }}<|im_start|>assistant\n"
                               "{{ .Response }}<|im_end|>")


def test_llama2_template():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    template = get_template_and_fix_tokenizer(tokenizer, name="llama2")
    ollama_template = convert_template_obj_to_ollama(template, tokenizer)

    assert ollama_template == ("{{ if .System }}<<SYS>>\n"
                               "{{ .System }}\n"
                               "<</SYS>>\n\n"
                               "{{ end }}{{ if .Prompt }}<s>[INST] {{ .Prompt }}{{ end }} [/INST]{{ .Response }}</s>")


def test_llama3_template():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    template = get_template_and_fix_tokenizer(tokenizer, name="llama3")
    ollama_template = convert_template_obj_to_ollama(template, tokenizer)

    assert ollama_template == ("{{ if .System }}<|start_header_id|>system<|end_header_id|>\n\n"
                               "{{ .System }}<|eot_id|>{{ end }}"
                               "{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>\n\n"
                               "{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>\n\n"
                               "{{ .Response }}<|eot_id|>")


def test_phi3_template():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    template = get_template_and_fix_tokenizer(tokenizer, name="phi")
    ollama_template = convert_template_obj_to_ollama(template, tokenizer)
    assert ollama_template == ("{{ if .System }}<|system|>\n"
                               "{{ .System }}<|end|>\n"
                               "{{ end }}{{ if .Prompt }}<|user|>\n"
                               "{{ .Prompt }}<|end|>\n"
                               "{{ end }}<|assistant|>\n"
                               "{{ .Response }}<|end|>")


if __name__ == '__main__':
    test_qwen2_template()
    test_yi_template()
    test_llama2_template()
    test_llama3_template()
    test_phi3_template()
