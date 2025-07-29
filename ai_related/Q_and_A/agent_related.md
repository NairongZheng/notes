## Agent相关

<details>
<summary>Agent中的tool use</summary>

<br>

**tool use定义、常见tool及流程**

> **定义**：Tool Use 是指 Agent 在推理过程中调用外部工具（function, API, module）来增强能力的机制。
> 
> **Tool Use 的常见类别**
> 
> | 类型                  | 说明                         | 示例                                   |
> | --------------------- | ---------------------------- | -------------------------------------- |
> | 检索工具              | 从外部知识库或文档中查询信息 | 向量数据库检索、RAG、Wikipedia Search  |
> | 计算工具              | 执行数学或逻辑计算           | Calculator、Math Solver、Python 执行器 |
> | API 调用工具          | 访问 Web 或第三方 API        | 查询天气、股票、商品价格               |
> | 搜索工具              | 实时搜索互联网信息           | Google/Bing Search Tool                |
> | 数据库工具            | 查询数据库内容               | SQL 查询器、GraphQL 工具               |
> | 文件操作工具          | 读写文件、处理本地内容       | 文件上传下载、PDF 阅读器               |
> | 代码执行器            | 编写、运行代码并查看结果     | Jupyter 执行、Python runner            |
> | 多模态工具            | 调用图像、音频、视频模型     | 图像识别、语音识别、视频分析           |
> | 调用子 Agent          | 工具本身是另一个 Agent       | 调用另一个专用 Agent 来分工协作        |
> | 任务规划器（Planner） | 工具用于分解和计划任务       | BabyAGI、AutoGPT 的任务分解模块        |
> 
> **Tool Use 的决策逻辑（推理 + 执行）**
> 
> 一个典型的 tool-use agent 的工作流程如下：
> 
> - 用户输入任务
> - Agent 分析输入 -> 判断是否需要调用工具
> - 选择合适的工具（如 calculator、search）
> - 调用工具，获取结果
> - 将工具结果继续用于下轮推理
> - 最终得出答案并返回
> 
> 这种交替过程也称为 ReAct（Reasoning + Acting） 模式。

**要怎么告诉 Agent 有哪些 Tool**

> 有两种主流方式，具体取决于你用的是哪种 Agent 框架（LangChain、OpenAI Function Calling、Autogen、Transformers Agent 等）：
> 
> **方式一：OpenAI Function Calling 风格（结构化 JSON 描述）**
> 
> 你需要给出工具的 函数签名（name, parameters, description），让语言模型知道：
> - 这个工具叫什么
> - 它能干什么（通过 description）
> - 它怎么用（通过 parameters 的结构）
> 
> ```bash
> # 示例（OpenAI 风格）
> {
>   "name": "get_weather",
>   "description": "获取指定城市的实时天气信息",
>   "parameters": {
>     "type": "object",
>     "properties": {
>       "city": {
>         "type": "string",
>         "description": "要查询天气的城市名，如 Beijing"
>       }
>     },
>     "required": ["city"]
>   }
> }
> ```
> 
> 你把这个作为 tool 的 metadata 提供给 Agent，Agent 会在推理时判断是否要使用该工具，并正确构造调用。
> 
> **方式二：LangChain 风格（Python 函数 + @tool 装饰器）**
> 
> 你定义 Python 函数，并提供函数名与 docstring（描述），LangChain 会将其转为结构化工具，语言模型通过 Prompt 理解这些描述：
> 
> ```python
> from langchain.tools import tool
> 
> @tool
> def get_weather(city: str) -> str:
>     """获取指定城市的天气"""
>     return f"{city} 今日天气晴，26°C"
> ```
> 
> LangChain 会把这个函数转成一段 Prompt，告诉语言模型它能用这个工具。

**Agent 是怎么“知道”怎么用这些工具的？**

> **通过 Prompt（自然语言）和 Schema（函数签名）**
> 
> Agent 的 prompt（系统提示词）里，通常会包含类似：
> 
> > 你有以下可用工具：
> > - get_weather(city: str): 获取城市天气
> > - search_web(query: str): 用搜索引擎查询问题
> 
> 模型通过这段 Prompt + 函数描述来“理解工具用途”。
> 
> 并且：
> - 模型会在内部判断是否要调用工具（例如：识别到“天气”关键词）
> - **模型会输出 tool name + 参数（如 JSON）**
> 
> **交互流程总结（以 Function Calling 为例）**
> 
> 1. 你通过代码注册工具（含描述和参数结构）
> 2. 这些信息被加到系统提示词中（或者直接以工具 JSON Schema 提供）
> 3. 模型理解这些工具的用途和参数格式
> 4. 当判断任务需要工具时，它会输出一个工具调用：
> 
> ```bash
> {
>   "function_call": {
>     "name": "get_weather",
>     "arguments": "{\"city\": \"Beijing\"}"
>   }
> }
> ```
> 
> 5. 系统接收到后真正调用函数（实际 API、代码等）
> 6. 返回结果给模型继续推理或回答用户

**举个完整例子（OpenAI 风格多轮工具调用）**

> **用户问**： “请告诉我北京天气，并用计算器告诉我温度换算成华氏度。”
> 
> 系统提供两个工具：
> - get_weather(city: str) → 返回 "今天 30°C"
> - calculator(expression: str) → 你可以传入 '30 * 9/5 + 32'
> 
> 流程：
> - Agent 调用 get_weather("Beijing")，得到 Observation: "30°C"
> - Agent 继续调用 calculator("30 * 9/5 + 32")，得到 Observation: "86"
> - 最后 Agent 汇总回答：“北京今天 30°C，也就是 86°F”

</details>


