import os

from openai import OpenAI


os.environ["OPENAI_BASE_URL"] = "http://192.168.5.193:8000/v1"
os.environ["OPENAI_API_KEY"] = "0"


if __name__ == "__main__":
    client = OpenAI()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    result = client.chat.completions.create(
        messages=[{"role": "user", "content": "What is the weather like in Boston?"}],
        model="gpt-3.5-turbo",
        tools=tools,
    )
    print(result.choices[0].message)
    result = client.chat.completions.create(
        messages=[
            {"role": "user", "content": "What is the weather like in Boston?"},
            {
                "role": "function",
                "content": """{"name": "get_current_weather", "arguments": {"location": "Boston, MA"}}""",
            },
            {"role": "tool", "content": '{"temperature": 22, "unit": "celsius", "description": "Sunny"}'},
        ],
        model="gpt-3.5-turbo",
        tools=tools,
    )
    print(result.choices[0].message)
