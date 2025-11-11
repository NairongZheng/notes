
import os
import re
import json
import requests
from jinja2 import Template
from openai import AzureOpenAI, OpenAI


def _get_reasoning_content_from_sdk(message):
    """
    从 OpenAI SDK message 对象中提取 reasoning 内容（兼容 model_extra）
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


def test_openai(base_url, api_key, model, messages, tools):
    """
    用OpenAI的api调用，直接调用v1接口
    base_url如: https://api.siliconflow.cn/v1, 当然也可以用openai的
    他会自动往 base_url 的 /chat/completions 路由接口发送请求
    """
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    # 有些模型是不支持某些参数的，可能会报错的
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=1000,
        temperature=0.5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        extra_body={
            "thinking": {"type": "enable"}, # 比如siliconflow的deepseek-ai/DeepSeek-R1模型就用这个
            "enable_thinking": True, # 比如siliconflow的zai-org/GLM-4.5模型就用这个
            "reasoning_mode": "high",
        }, # 额外的参数，如开启推理，不知道哪个模型用哪个字段，都写了
    )
    # 推理 thinking/reasoning 的内容（部分模型支持）
    reasoning_content = _get_reasoning_content_from_sdk(response.choices[0].message)
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


def test_azure_openai(azure_endpoint, api_key, model, messages, tools):
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
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        # max_tokens=1000,
        # temperature=0.5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        extra_body={
            # "thinking": {"type": "enable"},
            # "enable_thinking": True,
            # "reasoning_mode": "high",
        },
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


def test_post_api(url="", api_key="", model_name="", messages=[], tools=[], enable_reasoning=False):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name,
        "messages": messages,
        "tool_choice": "auto",
        "max_tokens": 1000,
        "temperature": 0.5,
        "top_p": 1,
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


def get_chat_template(messages, tools=None, enable_reasoning=False, add_generation_prompt=True):
    """
    根据 messages 和 tools 自动生成 Chat 模板文本。
    支持：
      system + user + assistant 多轮
      工具定义（tool schema）
      reasoning（<think></think> 块）
    """
    # 随便写的，大概这个意思吧
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
    return reasoning_content, model_response, tool_calls


def test_post_lightllm(url="", messages=[], tools=[], enable_reasoning=False):
    headers = {
        "Content-Type": "application/json"
    }
    prompt = get_chat_template(messages, tools, enable_reasoning)
    data = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 60000,
            "temperature": 0.7,
        }
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        reasoning_content, model_response, tool_calls = parse_lightllm_response(response.json())
        print(f"Reasoning content: {reasoning_content}")
        print(f"Model response: {model_response}")
        print(f"Tool calls: {tool_calls}")
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}\n response.text: {response.text}")
    except json.JSONDecodeError:
        print("❌ 响应不是JSON格式：", response.text)

def get_tools():
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
    return tools


def get_messages():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Answer me two questions: first, Please think how to count the sum of 1 to 100. second, Tell me the weather in Tokyo."}
    ]
    return messages


def main():
    tools = get_tools()
    messages = get_messages()
    
    # 获得一些需要的变量
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai_endpoint = os.getenv("OPENAI_ENDPOINT")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # # 1. 测试openai
    # print(f"========== test_openai ==========")
    # model_name = "zai-org/GLM-4.5" # "deepseek-ai/DeepSeek-R1", "zai-org/GLM-4.5"
    # test_openai(openai_endpoint, openai_api_key, model_name, messages, tools)

    # # 2. 测试azure openai
    # print(f"========== test_azure_openai ==========")
    # model_name = "o3-mini" # "o3-mini", "gpt-4o-mini-2024-07-18"
    # test_azure_openai(azure_openai_endpoint, azure_openai_api_key, model_name, messages, tools)

    # ================================ 也可以直接使用requests调用api，不使用sdk ================================
    # # 3. 测试post
    # print(f"========== test_post_api ==========")
    # # 3.1 测试post openai
    # print(f"========== test post openai (跟上面sdk差不多，只不过改成post调用方式，url也需要相应改) ==========")
    # url = openai_endpoint + "/chat/completions"
    # model_name = "zai-org/GLM-4.5" # "deepseek-ai/DeepSeek-R1", "zai-org/GLM-4.5"
    # test_post_api(url, openai_api_key, model_name, messages, tools, enable_reasoning=True)
    # # 3.2 测试post azure openai
    # print(f"========== test post azure_openai (跟上面sdk差不多，只不过改成post调用方式，url也需要相应改) ==========")
    # url = azure_openai_endpoint + "/openai/v1/chat/completions?api-version=preview"
    # model_name = "gpt-4o-mini-2024-07-18" # "o3-mini", "gpt-4o-mini-2024-07-18"
    # test_post_api(url, azure_openai_api_key, model_name, messages, tools, enable_reasoning=False) # 同理也不测这里的推理
    # 3.3 测试post lightllm
    print(f"========== test post lightllm ==========")
    # url = "http://${ip}:${port}/v1/chat/completions"
    # test_post_api(url=url, messages=messages, tools=tools, enable_reasoning=True) # 一般用lightllm就直接调用generate裸接口了，不用这个
    url = "http://${ip}:${port}/generate"
    test_post_lightllm(url, messages=messages, tools=tools, enable_reasoning=True)


if __name__ == "__main__":
    main()