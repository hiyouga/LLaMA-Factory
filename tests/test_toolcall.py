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
    return total_score / total_hour


def main():
    client = OpenAI(
        api_key="0",
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
    tool_call = result.choices[0].message.tool_calls[0].function
    name, arguments = tool_call.name, json.loads(tool_call.arguments)
    messages.append(
        {"role": "function", "content": json.dumps({"name": name, "argument": arguments}, ensure_ascii=False)}
    )
    tool_result = tool_map[name](**arguments)
    messages.append({"role": "tool", "content": json.dumps({"gpa": tool_result}, ensure_ascii=False)})
    result = client.chat.completions.create(messages=messages, model="test", tools=tools)
    print(result.choices[0].message.content)
    # Based on your grades and credit hours, your calculated Grade Point Average (GPA) is 3.4166666666666665.


if __name__ == "__main__":
    main()
