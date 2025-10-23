
import os
from openai import AzureOpenAI
import requests


def test_azure_openai(url, api_key, model_name, messages, tools):
    client = AzureOpenAI(
        azure_endpoint=url,
        api_key=api_key,
        api_version="2025-01-01-preview",
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=1000,
        temperature=0.5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    content = response.choices[0].message.content
    print(f"Content: {content}")
    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        print("No tool calls")
        return
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        print(f"Tool name: {tool_name}")
        print(f"Tool arguments: {tool_args}")


def test_post(url, api_key, model_name, messages, tools):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 1000,
        "temperature": 0.5,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
    response = requests.post(url, headers=headers, json=data)
    result = response.json()
    content = result["choices"][0]["message"]["content"]
    print(f"Content: {content}")
    if "tool_calls" not in result["choices"][0]["message"]:
        print("No tool calls")
        return
    tool_calls = result["choices"][0]["message"]["tool_calls"]
    for tool_call in tool_calls:
        tool_name = tool_call["function"]["name"]
        tool_args = tool_call["function"]["arguments"]
        print(f"Tool name: {tool_name}")
        print(f"Tool arguments: {tool_args}")


def main():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The location to get the weather for"},
                    },
                    "required": ["location"],
                },
            }
        }
    ]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please introduce yourself and tell me the weather in Tokyo."}
    ]
    # 获得一些需要的变量
    url = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    model_name = "gpt-4o-mini-2024-07-18"

    print(f"========== test_azure_openai ==========")
    test_azure_openai(url, api_key, model_name, messages, tools)

    print(f"========== test_post ==========")
    url = url + "/openai/v1/chat/completions?api-version=preview"
    test_post(url, api_key, model_name, messages, tools)


if __name__ == "__main__":
    main()