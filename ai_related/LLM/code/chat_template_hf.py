from jinja2 import Environment, Template, FileSystemLoader
import textwrap
import os

jinja2_dir = os.path.dirname(os.path.abspath(__file__))

def get_template():
    """
    获取chat_template
    """
    env = Environment(loader=FileSystemLoader(jinja2_dir))
    chat_template = env.get_template("chat_template_qwen.jinja2")
    return chat_template


def get_messages():
    """
    示例数据
    """
    messages = [
        {"role": "system", "content": "You are a web assistant."},
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "search_web", "arguments": {"query": "weather in Paris"}}}],
            "content": "<think>I need to search the web for the weather in Paris.</think>"
        },
        {"role": "tool", "content": '{"result": "Sunny, 20°C"}'},
        {"role": "assistant", "content": "<think>I can get the weather from the tool.</think>It's sunny and 20°C in Paris today."}
    ]
    tools = [
        {"name": "search_web", "parameters": {"query": "string"}}
    ]
    return messages, tools


def main():
    chat_template = get_template()
    messages, tools = get_messages()
    rendered = chat_template.render(
        messages=messages,
        tools=tools,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    print(rendered)


if __name__ == "__main__":
    main()