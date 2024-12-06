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

import json
import os
from typing import Sequence

from openai import OpenAI
from transformers.utils.versions import require_version


require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")


def calculate_gpa(grades: Sequence[str], hours: Sequence[int]) -> float:
    grade_to_score = {"A": 4, "B": 3, "C": 2}
    total_score, total_hour = 0, 0
    for grade, hour in zip(grades, hours):
        total_score += grade_to_score[grade] * hour
        total_hour += hour
    return round(total_score / total_hour, 2)


def main():
    client = OpenAI(
        api_key="{}".format(os.environ.get("API_KEY", "0")),
        base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 8000)),
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate_gpa",
                "description": "Calculate the Grade Point Average (GPA) based on grades and credit hours",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "grades": {"type": "array", "items": {"type": "string"}, "description": "The grades"},
                        "hours": {"type": "array", "items": {"type": "integer"}, "description": "The credit hours"},
                    },
                    "required": ["grades", "hours"],
                },
            },
        }
    ]
    tool_map = {"calculate_gpa": calculate_gpa}

    messages = []
    messages.append({"role": "user", "content": "My grades are A, A, B, and C. The credit hours are 3, 4, 3, and 2."})
    result = client.chat.completions.create(messages=messages, model="test", tools=tools)
    if result.choices[0].message.tool_calls is None:
        raise ValueError("Cannot retrieve function call from the response.")

    messages.append(result.choices[0].message)
    tool_call = result.choices[0].message.tool_calls[0].function
    print(tool_call)
    # Function(arguments='{"grades": ["A", "A", "B", "C"], "hours": [3, 4, 3, 2]}', name='calculate_gpa')
    name, arguments = tool_call.name, json.loads(tool_call.arguments)
    tool_result = tool_map[name](**arguments)
    messages.append({"role": "tool", "content": json.dumps({"gpa": tool_result}, ensure_ascii=False)})
    result = client.chat.completions.create(messages=messages, model="test", tools=tools)
    print(result.choices[0].message.content)
    # Based on the grades and credit hours you provided, your Grade Point Average (GPA) is 3.42.


if __name__ == "__main__":
    main()
