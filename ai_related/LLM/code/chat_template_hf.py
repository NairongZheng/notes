from jinja2 import Template
import json


def get_template():
    """
    获取chat_template
    """
    chat_template = """
    {%- if tools %}
    <|im_start|>system
    # Tools
    You may call one or more functions to assist with the user query.

    You are provided with function signatures within <tools></tools> XML tags:
    <tools>
    {%- for tool in tools %}
    {{ tool | tojson }}
    {%- endfor %}
    </tools>

    For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
    <tool_call>
    {"name": <function-name>, "arguments": <args-json-object>}
    </tool_call>
    <|im_end|>
    {%- endif %}

    {%- for message in messages %}
        {%- if message.role == "system" %}
    <|im_start|>system
    {{ message.content }}
    <|im_end|>
        {%- elif message.role == "user" %}
    <|im_start|>user
    {{ message.content }}
    <|im_end|>
        {%- elif message.role == "assistant" %}
    <|im_start|>assistant
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
    <tool_call>
    {{ tool_call.function | tojson }}
    </tool_call>
            {%- endfor %}
        {%- else %}
    {{ message.content }}
        {%- endif %}
    <|im_end|>
        {%- elif message.role == "tool" %}
    <|im_start|>user
    <tool_response>
    {{ message.content }}
    </tool_response>
    <|im_end|>
        {%- endif %}
    {%- endfor %}

    <|im_start|>assistant
    """
    return chat_template


def get_messages():
    """
    示例数据
    """
    messages = [
        {"role": "system", "content": "You are a web assistant."},
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "search_web", "arguments": {"query": "weather in Paris"}}}
        ]},
        {"role": "tool", "content": '{"result": "Sunny, 20°C"}'},
        {"role": "assistant", "content": "It's sunny and 20°C in Paris today."}
    ]
    tools = [
        {"name": "search_web", "parameters": {"query": "string"}}
    ]
    return messages, tools


def main():
    chat_template = get_template()
    template = Template(chat_template)
    messages, tools = get_messages()
    rendered = template.render(messages=messages, tools=tools)
    print(rendered)


if __name__ == "__main__":
    main()