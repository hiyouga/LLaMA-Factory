# Copyright 2025 the LlamaFactory team.
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


# if __name__ == "__main__":

#     def to_qwen3_messages(template: QwenTemplate, messages: list[dict]):
#         out = []
#         for m in messages:
#             role = m["role"]
#             content = template._extract_content(m.get("content", ""))
#             if role == "assistant":
#                 reasoning = (m.get("reasoning_content") or "").strip()
#                 if reasoning:
#                     content = template.thinking_template.format(content=reasoning) + content
#             out.append({"role": role, "content": content})
#         return out

#     from transformers import AutoTokenizer

#     tok = AutoTokenizer.from_pretrained(
#         "Qwen/Qwen3-30B-A3B-Thinking-2507",
#         trust_remote_code=True,
#     )

#     test_messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": [{"type": "text", "text": "1+1等于几？"}, {"type": "text", "text": "2+2等于几？"}],
#         },
#         {
#             "role": "assistant",
#             "reasoning_content": "这是一个简单的数学问题。1加1的结果是2。",
#             "content": [{"type": "text", "text": "1+1=2"}, {"type": "text", "text": "2+2=4"}],
#         },
#     ]

#     template = QwenTemplate()
#     rendered_custom = "".join([template.render_message(m) for m in test_messages])

#     qwen3_messages = to_qwen3_messages(template, test_messages)
#     rendered_hf = tok.apply_chat_template(qwen3_messages, tokenize=False, add_generation_prompt=False)

#     print("==== custom ====")
#     print(rendered_custom)
#     print("==== hf ====")
#     print(rendered_hf)

#     assert rendered_custom.strip() == rendered_hf.strip(), "Rendered text mismatch"

#     ids_custom = tok.encode(rendered_custom, add_special_tokens=False)
#     ids_hf = tok.apply_chat_template(qwen3_messages, tokenize=True, add_generation_prompt=False)
#     assert ids_custom == ids_hf, f"Token ids mismatch: custom={len(ids_custom)} hf={len(ids_hf)}"
