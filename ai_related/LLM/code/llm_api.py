
import os
import re
import json
import requests
from jinja2 import Template
from openai import AzureOpenAI, OpenAI


def _get_reasoning_content_from_chat_completions(message):
    """
    从 OpenAI chat.completions.create() 的 message 对象中提取 reasoning 内容（兼容 model_extra）
    message: client.chat.completions.create().choices[0].message
    """
    # 优先取显式字段
    if hasattr(message, "reasoning_content") and message.reasoning_content:
        return message.reasoning_content

    # 否则从 model_extra 里取
    if hasattr(message, "model_extra") and isinstance(message.model_extra, dict):
        if "reasoning_content" in message.model_extra:
            return message.model_extra["reasoning_content"]
        elif "thinking_content" in message.model_extra:  # 某些平台这么命名
            return message.model_extra["thinking_content"]

    return None


def _get_reasoning_content_from_response(response):
    """
    从 OpenAI responses.create() 的 response 对象中提取 reasoning / thinking 内容
    response: client.responses.create()
    """
    reasoning_chunks = []

    for item in response.output:
        if item["type"] in ("reasoning", "thinking"):
            # 不同模型字段不统一，尽量兜底
            if "content" in item:
                reasoning_chunks.append(item["content"])
            elif "text" in item:
                reasoning_chunks.append(item["text"])

    return "\n".join(reasoning_chunks) if reasoning_chunks else None


def _get_reasoning_content_from_dict(message: dict):
    """
    从 message 中提取 reasoning 内容（兼容 model_extra）
    """
    if not message:
        return None

    # 1. 尝试直接取字段
    if "reasoning_content" in message:
        return message["reasoning_content"]

    # 2. 尝试从 model_extra 里取
    if "model_extra" in message and isinstance(message["model_extra"], dict):
        extra = message["model_extra"]
        for key in ("reasoning_content", "thinking_content"):
            if key in extra:
                return extra[key]

    return None


def get_chat_template(messages, tools=None, enable_reasoning=False, add_generation_prompt=True):
    """
    根据 messages 和 tools 自动生成 Chat 模板文本。
    支持：
      system + user + assistant 多轮
      工具定义（tool schema）
      reasoning（<think></think> 块）
    """
    # 随便写的，大概这个意思吧（或者去看chat_template_qwen.jinja2）
    chat_template = """
    {%- if tools %}
        {{- '<|im_start|>system\n' }}
        {%- if messages[0].role == "system" %}
            {{ messages[0].content }}
        {%- endif %}
        {%- if enable_reasoning %}
            {{- '\n\nReason step by step and place the thought process within the <think></think> tags, and provide the final conclusion at the end.\n\n' }}
        {%- endif %}
        {{- '\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>' }}
            {%- for tool in tools %}
                {{ tool | tojson }}
            {%- endfor %}
            {{- '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n' }}
    {%- else %}
        {%- if messages[0].role == "system" %}
            {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
        {%- endif %}
        {%- if enable_reasoning %}
            {{- '\n\nReason step by step and place the thought process within the <think></think> tags, and provide the final conclusion at the end.\n\n' }}
        {%- endif %}
    {%- endif %}
    
    {%- for message in messages %}
        {%- if message.role == "user" %}
            {{- '<|im_start|>user\n' }}
            {{- message.content + '\n<|im_end|>\n' }}
        {%- elif message.role == "assistant" %}
            {{- '<|im_start|>assistant\n' }}
            {{- message.content + '\n<|im_end|>\n' }}
        {%- endif %}
    {%- endfor %}
    {{- '<|im_start|>assistant\n' }}
    """
    
    prompt_rendered = Template(chat_template).render(
        messages=messages,
        tools=tools,
        enable_reasoning=enable_reasoning,
        add_generation_prompt=add_generation_prompt
    )
    return prompt_rendered


def parse_lightllm_response(response: dict):
    """
    解析lightllm的响应
    """
    model_response = response['generated_text'][0]
    # 提取think内容
    if "</think>" in model_response:
        parts = model_response.split("</think>")
        reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
    else:
        reasoning_content = None
    # 提取tool内容
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, model_response, re.DOTALL)
    tool_calls = []
    for match in matches:
        try:
            match = json.loads(match.strip())
        except Exception as e:
            pass
        tool_calls.append(match)
    return model_response, reasoning_content, tool_calls


def get_tools():
    chat_tools = [
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
    response_tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Retrieves current weather for the given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country e.g. Bogotá, Colombia"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Units the temperature will be returned in."
                    }
                },
                "required": ["location"],
            }
        }
    ]
    return chat_tools, response_tools


def get_messages():
    # 给旧的 /chat/completions 接口
    chat_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Answer me two questions: first, Please think how to count the sum of 1 to 100. second, Tell me the weather in Tokyo."}
    ]
    # 给新的 /response 接口
    response_input = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Answer me two questions: first, Please think how to count the sum of 1 to 100. second, Tell me the weather in Tokyo."}
            ]
        }
    ]
    return chat_messages, response_input


def test_openai_chat_completions(base_url, api_key, model, messages, tools):
    """
    测试 openai 的 <base_url>/chat/completions
    """
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    # 有些模型是不支持某些参数的，可能会报错的
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=1000,
            temperature=0.6,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            extra_body={
                "thinking": {"type": "enable"}, # 比如siliconflow的deepseek-ai/DeepSeek-R1模型就用这个
                "enable_thinking": True, # 比如siliconflow的zai-org/GLM-4.5模型就用这个
                "reasoning_mode": "high",
            }, # 额外的参数，如开启推理，不知道哪个模型用哪个字段，都写了
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return
    # 推理 thinking/reasoning 的内容（部分模型支持）
    reasoning_content = _get_reasoning_content_from_chat_completions(response.choices[0].message)
    print(f"Reasoning content: {reasoning_content}")
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


def test_openai_response(base_url, api_key, model, messages, tools):
    """
    测试 openai 的 /responses
    """
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    try:
        response = client.responses.create(
            model=model,
            input=messages,   # 注意：responses 用 input，不是 messages
            tools=tools,
            tool_choice="auto",
            max_output_tokens=1000,
            temperature=0.6,
            top_p=0.95,
            extra_body={
                # 不同厂商 / 模型可能支持不同字段
                "thinking": {"type": "enable"},
                "enable_thinking": True,
                "reasoning_mode": "high",
            },
        )
    except Exception:
        import traceback
        traceback.print_exc()
        return

    # 提取推理内容
    reasoning_content = _get_reasoning_content_from_response(response)
    print(f"Reasoning content: {reasoning_content}")

    # 提取最终文本输出
    output_text = []
    tool_calls = []

    for item in response.output:
        # 普通文本
        if item["type"] == "output_text":
            output_text.append(item["text"])

        # tool call
        elif item["type"] == "tool_call":
            tool_calls.append(item)

    content = "\n".join(output_text)
    print(f"Content: {content}")

    # 打印工具调用
    if not tool_calls:
        print("No tool calls")
        return

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("arguments", {})

        print(f"Tool name: {tool_name}")
        print(f"Tool arguments: {tool_args}")


def test_azure_chat_completions(azure_endpoint, api_key, model, messages, tools):
    """
    用AzureOpenAI的api调用
    url: 如 https://{xxxxxx}.openai.azure.com, 这是azure上自己定义的endpoint
    model_name: 也是在部署的时候自己定义的
    api_version: 如 2025-01-01-preview, 也是部署的时候自己定义的
    推理也是类似，这边不知道一些部署模型的参数，就不试了
    """
    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version="2025-01-01-preview",
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            # max_tokens=1000,
            # temperature=0.5,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            extra_body={
                # "thinking": {"type": "enable"},
                # "enable_thinking": True,
                # "reasoning_mode": "high",
            },
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return
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


def test_post(url="", api_key="", model_name="", messages=[], tools=[], test_type="", enable_reasoning=False):
    if test_type == "openai":
        url = url + "/chat/completions"
    elif test_type == "azure":
        api_version = "2025-01-01-preview"
        url = url + f"/openai/deployments/{model_name}/chat/completions?api-version={api_version}"
    else:
        url = url

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name,
        "messages": messages,
        "tool_choice": "auto",
        "max_tokens": 1000,
        "temperature": 0.6,
        "top_p": 0.95,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
    if tools:
        data["tools"] = tools
    if enable_reasoning:
        data.update({
            "thinking": {"type": "enable"},
            "enable_thinking": True,
            "reasoning_mode": "high"
        })
    # 发送请求
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}\n response.text: {response.text}")
        return
    except Exception as e:
        import traceback
        traceback.print_exc()
        return
    # 尝试解析JSON
    try:
        result = response.json()
    except json.JSONDecodeError:
        print("❌ 响应不是JSON格式：", response.text)
        return
    # 解析响应
    choices = result.get("choices", [])
    if not choices:
        print("⚠️ 未返回choices字段")
        return
    message = choices[0].get("message", {})
    content = message.get("content", "")
    print(f"Content: {content}")
    
    # 推理内容
    reasoning = _get_reasoning_content_from_dict(message)
    print(f"Reasoning: {reasoning}")
    
    # 工具调用
    tool_calls = message.get("tool_calls", [])
    if not tool_calls:
        print("No tool calls")
    else:
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = tool_call["function"]["arguments"]
            print(f"Tool name: {tool_name}")
            print(f"Tool arguments: {tool_args}")


def test_post_lightllm_generate(url="", messages=[], tools=[], enable_reasoning=False):
    headers = {
        "Content-Type": "application/json"
    }
    prompt = get_chat_template(messages, tools, enable_reasoning)
    data = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 60000,
            "temperature": 0.6,
            "top_p": 0.95,
        }
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        model_response, reasoning_content, tool_calls = parse_lightllm_response(response.json())
        print(f"Reasoning content: {reasoning_content}")
        print(f"Model response: {model_response}")
        print(f"Tool calls: {tool_calls}")
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}\n response.text: {response.text}")
    except json.JSONDecodeError:
        print("❌ 响应不是JSON格式：", response.text)


def main():
    chat_tools, response_tools = get_tools()
    chat_messages, response_input = get_messages()
    
    # 获得一些需要的变量
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    # 设置模型名字
    model_name = "z-ai/glm-4.5" # "z-ai/glm-4.5", "gpt-4o-mini-2024-07-18"
    
    # 1. 测试 openai 的 /chat/completions
    test_openai_chat_completions(openai_base_url, openai_api_key, model_name, chat_messages, chat_tools)
    
    # # 2. 测试 openai 的 /response
    # test_openai_response(openai_base_url, openai_api_key, model_name, response_input, response_tools)
    
    # # 3. 测试 azure 的 /chat/completions
    # test_azure_chat_completions(azure_openai_endpoint, azure_openai_api_key, model_name, chat_messages, chat_tools)
    
    # # 4. 测试 post openai chat/completions
    # test_post(openai_base_url, openai_api_key, model_name, chat_messages, chat_tools, test_type="openai", enable_reasoning=True)
    
    # # 5. 测试 post azure chat/completions
    # test_post(azure_openai_endpoint, azure_openai_api_key, model_name, chat_messages, chat_tools, test_type="azure", enable_reasoning=False)
    
    # # 6. 测试 post lightllm generate
    # test_post_lightllm_generate(url="http://10.119.20.237:8000/generate", messages=chat_messages, tools=chat_tools, enable_reasoning=True)


if __name__ == "__main__":
    main()