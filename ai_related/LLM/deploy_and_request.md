- [部署llm](#部署llm)
  - [使用 sglang 部署](#使用-sglang-部署)
  - [使用 lightllm 部署](#使用-lightllm-部署)
  - [使用 vllm 部署](#使用-vllm-部署)
- [请求llm](#请求llm)
  - [OpenAI \& AzureOpenAI](#openai--azureopenai)
    - [请求与返回](#请求与返回)
    - [推理字段介绍](#推理字段介绍)
  - [直接采用post请求](#直接采用post请求)


# 部署llm

## 使用 sglang 部署

sglang：[官方文档(en)](https://docs.sglang.io/)

**安装环境**

```shell
conda create -n sglang python=3.12 -y
conda activate sglang
pip install uv
uv pip install "sglang[all]>=0.4.6.post4"
```

**启动**

```shell
python -m sglang.launch_server \
    --model-path ${model_dir} \
    --tp ${tp_size} \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --dist-init-addr ${MASTER_ADDR}:20000 \
    --nnodes ${num_nodes} \
    --node-rank ${node_rank} \
    --tool-call-parser glm  \
    --reasoning-parser glm45
```

## 使用 lightllm 部署

lightllm：[官方文档](https://lightllm-cn.readthedocs.io/en/latest)，lightllm 的文档还是很值得看的。

**安装环境**

```shell
# (推荐) 创建一个新的 conda 环境
conda create -n lightllm python=3.10 -y
conda activate lightllm

# 下载lightllm的最新源码
git clone https://github.com/ModelTC/lightllm.git
cd lightllm

# 安装lightllm的依赖 (cuda 12.4)
apt-get install -y libgmp-dev libmpfr-dev libmpc-dev
pip install uv # 使用 uv 安装较快
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match

# 安装lightllm
python setup.py install
```

**启动**

[启动参数文档](https://lightllm-cn.readthedocs.io/en/latest/tutorial/api_server_args_zh.html)

```shell
python -m lightllm.server.api_server \
    --model_dir ${model_dir} \
    --host 0.0.0.0 \
    --port 8000 \
    --tp ${GPUS_PER_NODE} \
    --max_req_total_len 65000 \
    --mode triton_gqa_flashdecoding \
    --mem_fraction 0.8 \
    --trust_remote_code \
    --enable_multimodal \
    --nccl_port 27938 \
    --data_type bf16 \
    --graph_max_batch_size 64 \
    --use_dynamic_prompt_cache \
    --tool_call_parser qwen25 \
    --visual_infer_batch_size 4 \
    --visual_gpu_ids 0 \
    --visual_nccl_ports 29501 \
    --cache_capacity 300
```

## 使用 vllm 部署

vllm：[官方文档(en)](https://docs.vllm.ai/)，[cli参数(en)](https://docs.vllm.ai/en/latest/cli/)

**安装环境**

```shell
conda create -n vllm python=3.12 -y
conda activate vllm
pip install uv
uv pip install vllm=0.12.0
# 部署的时候可能有一些包例如 numpy 的版本会冲突，改一下就行
```

**部署**

```shell
# 使用 ray 作为分布式后端
# 1. master 节点运行
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265
# 2. 其他 worker 节点运行
ray start --address='${MASTER_ADDR}:6379'
# 3. 启动-有些参数需要修改（注意所有节点的环境需要是一样的）
vllm serve $model_dir \
    --served-model-name ${model_name} \
    --host 0.0.0.0 \
    --port 8090 \
    --distributed-executor-backend ray \
    --tensor-parallel-size ${GPUS_PER_NODE} \
    --pipeline-parallel-size ${num_nodes} \
    --max-model-len 128000 \
    --enable-auto-tool-choice \
    --tool-call-parser glm45 \
    --reasoning-parser glm45
```

**请求 vllm**

```shell
# vllm serve 启动之后有类似的log
(APIServer pid=1209) INFO 11-05 23:12:32 [api_server-py: 1865] Starting VLLM API server 0 on http://0.0.0.0:8090
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher•py:29] Available routes are:
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /openapi.json, Methods: GET, HEAD
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /docs, Methods: GET, HEAD
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /docs/oauth2-redirect, Methods: GET, HEAD
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /redoc, Methods: GET, HEAD
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /health, Methods: GET
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /load, Methods: GET
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /ping, Methods: POST
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /ping, Methods: GET
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /tokenize, Methods: POST
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /detokenize, Methods: POST
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /vl/models, Methods: GET
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /version, Methods: GET
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /vl/responses, Methods: POST
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /vl/responses/{response_id}, Methods: GET
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /vl/responses/{response_id}/cancel, Methods: POST
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /v1/chat/completions, Methods: POST
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher.py: 37] Route: /v1/completions, Methods: POST
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /vl/embeddings, Methods: POST
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /pooling, Methods: POST
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /classify, Methods: POST
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /score, Methods: POST
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /vl/score, Methods: POST
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /vl/audio/transcriptions, Methods: POST
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /vl/audio/translations, Methods: POST
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /rerank, Methods: POST
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /v1/rerank, Methods: POST
(APIServer pid=1209) INFO 11-05 23:12:32 [launcher-py: 37] Route: /v2/rerank, Methods: POST

# 收到请求之后会有类似的log
(APIServer pid=1221) INFO 11-06 19:52:50 [loggers.py:123] Engine 000: Avg prompt throughput: 21331.0 tokens/s, Avg generation throughput: 1221.1 tokens/s, Running: 97 reqs, Waiting: 0 reqs, GPU KV cache usage: 19.0%, Prefix cache hit rate: 88.4%
```

# 请求llm

[参考脚本](./code/llm_api.py)

**几个不同请求接口的介绍（重要！）**

| 接口路由               | 对应模型类型                     | 输入格式                                         | 核心用途                            | 兼容对象                                       |
| ---------------------- | -------------------------------- | ------------------------------------------------ | ----------------------------------- | ---------------------------------------------- |
| `/v1/responses`        | 新一代 OpenAI 标准接口           |                                                  | 对话 / 推理 / 工具 / 多模态统一入口 |                                                |  |
| `/v1/chat/completions` | Chat 类型模型（ChatML 模板）     | `messages=[{"role":"user","content":"..."}]`     | 多轮对话、工具调用、Reasoning       | OpenAI Chat API（GPT 系列）                    |
| `/v1/completions`      | Base 模型（非 Chat 指令模型）    | `prompt="..."`                                   | 单轮续写、补全、翻译、改写          | 旧版 OpenAI Completion API（慢慢弃用）         |
| `/generate`            | 通用接口（厂商自定义、很灵活！） | 结构随实现不同（多为 chat_template 拼好的 text） | 任意文本生成                        | HuggingFace / vLLM / LightLLM 风格通用生成接口 |

**/v1/responses - 新接口**

> 这是新一代 OpenAI 标准接口，可以对话/推理/工具/多模态。
> - 支持状态会话管理，可以让 API 自动维持上下文，而不需要每次从头传 messages
> - 支持工具调用
> - 支持结构化输出，如要求返回某个 JSON Schema，而不是自由形式文本
> - 支持多模态输入
> 
> （具体查看[官方文档：https://platform.openai.com/docs/api-reference/responses](https://platform.openai.com/docs/api-reference/responses)）

**/v1/chat/completion — 聊天接口（Chat 模型专用）**

> 这是 OpenAI GPT 系列标准接口。
> 
> 用于「多轮对话式模型」—— 模型是按消息角色（system、user、assistant）训练的。
> 
> ```shell
> # input
> {
>   "model": "gpt-4o",
>   "messages": [
>     {"role": "system", "content": "你是一个AI助理"},
>     {"role": "user", "content": "写一首关于春天的诗"}
>   ],
>   "temperature": 0.7
> }
> ```
> 
> ```shell
> # output
> {
>   "choices": [
>     {
>       "message": {
>         "role": "assistant",
>         "content": "春风拂面百花开……"
>       },
>       "finish_reason": "stop"
>     }
>   ]
> }
> ```
> 
> 特点：
> - system/user/assistant 多角色
> - 工具调用（function calling）
> - reasoning / thinking 扩展
> - 多轮上下文（messages 堆叠）
> - streaming 流式输出
> - 仅适用于已通过指令微调（Instruction tuning）或 ChatML 模板训练的模型（例如 GPT、Qwen-Chat、LLaMA-Instruct）。

**/v1/completions — 旧式补全接口（Base 模型 / Prompt 模型）**

> 这是 最早期 OpenAI API（text-davinci-003） 的接口，主要用于“续写”类任务，比如自动补全句子、翻译、生成文本。
> 
> ```shell
> # input
> {
>   "model": "text-davinci-003",
>   "prompt": "写一首关于春天的诗：",
>   "max_tokens": 100
> }
> ```
> 
> ```shell
> # output
> {
>   "choices": [
>     {
>       "text": "春风拂面百花开，绿柳垂烟映碧苔……",
>       "finish_reason": "stop"
>     }
>   ]
> }
> ```
> 
> 优点：
> - 结构简单、通用性强
> - 适合纯文本生成任务（摘要、补全、翻译、代码续写）
> 
> 缺点：
> - 不支持多轮 messages
> - 不支持 tool_calls / function_call
> - 无 role / system 概念
> - 逐渐被 /v1/chat/completions 取代

**/generate — 通用生成接口（框架自定义）**

> /generate 是 LightLLM 等框架为方便兼容不同模型而定义的通用路径。
> 
> ```shell
> # input
> # 不同框架不完全一致，例如lightllm：
> 
> {
>   "inputs": "经过chat_template拼好的prompt",    # 使用openai的sdk本质上也是会拼成字符串再传给llm
>   "parameters": {
>     "max_new_tokens": 100,
>     "temperature": 0.7,
>   }
> }
> ```
> 
> ```shell
> # output
> # 不同框架不完全一致，例如lightllm：
> {
>   "generated_text": ["XXXXXX"]    # 需要自己解析
> }
> ```
> 
> 优点：
> - 框架最简通用接口，开发调试方便
> - 可直接用于各种 HuggingFace 模型
> - 无需遵循 OpenAI 消息结构
> - 性能高、接口轻量
> 
> 缺点：
> - 不兼容 OpenAI SDK（因为结构不同）
> - 不支持多轮 messages / tools （用chat_template拼完就可以支持）
> - reasoning、思维链、多角色功能通常需另行实现（同上，用template）


## OpenAI & AzureOpenAI

**二者基础对比介绍**

| 对比项       | OpenAI                                          | AzureOpenAI (继承自 OpenAI)                        |
| ------------ | ----------------------------------------------- | -------------------------------------------------- |
| 提供方       | 官方 OpenAI 平台（https://platform.openai.com） | 微软 Azure OpenAI 服务（https://portal.azure.com） |
| API Base URL | https://api.openai.com/v1 （或自定义兼容服务）  | https://your-resource-name.openai.azure.com        |
| 身份验证     | 通过 api_key（OpenAI key）                      | 通过 Azure 提供的 api_key 和 api_version           |
| 模型命名方式 | gpt-4o-mini, gpt-4o, gpt-3.5-turbo 等           | 自定义模型部署名，如 "my-gpt4o-deployment"         |
| 调用方式     | 直接指定模型名                                  | 指定部署名（对应 Azure 门户里的“Deployment Name”） |
| 主要用途     | 调用 OpenAI 的官方云端模型                      | 在企业环境中调用同样的模型（托管在 Azure）         |
| 返回结构     | 标准 OpenAI 格式                                | 与 OpenAI 一致（微软保持兼容）                     |
| 额外安全性   | 一般用于个人/小规模项目                         | 支持企业安全、VNet、合规、SLA、高级监控            |

**二者区别对比介绍**

| 项目                          | OpenAI                                   | AzureOpenAI                                                                                        |
| ----------------------------- | ---------------------------------------- | -------------------------------------------------------------------------------------------------- |
| 认证方式                      | 直接使用 api_key                         | 使用 api_key + azure_endpoint + api_version                                                        |
| 模型字段                      | 直接用官方模型名（如 "gpt-4o")           | 用部署名（如 "gpt4o-deploy"）                                                                      |
| 是否要设置 api_version        | 否                                       | ✅ 必须                                                                                             |
| Base URL                      | 默认 https://api.openai.com/v1           | 必须设置为 Azure endpoint                                                                          |
| Request URL                   | 会在 bsee_url 后面拼 `/chat/completions` | 会在 base_url 后面拼 `/openai/deployments/{model_name}/chat/completions?api-version={api-version}` |
| 是否支持自定义网络 / 安全策略 | ❌                                        | ✅ 支持企业 VNet、Private Link、合规性控制                                                          |
| 兼容性                        | 用于所有 OpenAI 兼容服务                 | 专用于 Azure 环境                                                                                  |

### 请求与返回

**请求**

> 关于 `extra_body` 介绍：
> - `OpenAI.chat.completions.create`不支持但是部分模型又需要的参数可以通过`extra_body`传
> - `extra_body`里面的各个字段实际上是**解开传进去的**，而不是直接传 `extra_body` 这个字段
> 
> 可去看源码`<conda_env_path>/lib/python3.12/site-packages/openai/_base_client.py`中的`request`函数

**返回**

> 模型的返回，两个包都会包装好再返回，跟 requests 方式返回的 dict 有区别，但本质上一样的，因为最后都是使用的 requsets.post 发送请求的。
> 
> 可以从源码`<conda_env_path>/lib/python3.12/site-packages/openai/resources/chat/completions/completions.py`一路看进去
> 
> 返回的类型是`ChatCompletionMessage <- BaseModel <- pydantic.BaseModel`。`pydantic.BaseModel`中：
> - `model_fields`: 正式字段，所以会在调试的时候看到
> - `model_extra`: 动态字段/未声明字段，所以调试的时候看不到
> - 某些模型会有额外的返回就会放到`model_extra`中
>
> `OpenAI`的`BaseModel`定义的是`model_config = {"extra": "allow", "defer_build": True}`
> 
> 所以在拿到返回时，有些放在`model_extra`中的参数在对象展开时看不到，但是实际上是存在的，可以直接对象引用到（如下面例子中的推理内容）


```python
# 其实只有 client 创建方式不一样而已

# OpenAI client:
client = OpenAI(base_url="", api_key="")
# OpenAI 的请求:
response = client.chat.completions.create(model="", messages=[], tools=[], extra_body={})
# OpenAI 的返回: response.choices[0].message, response 里有多个字段，都可以看看

# AzureOpenAI client:
client = AzureOpenAI(azure_endpoint="", api_key="", api_version="")
# AzureOpenAI 的请求:
response = client.chat.completions.create(model="", messages=[], tools=[], extra_body={})
# AzureOpenAI 的返回: response.choices[0].message, response 里有多个字段，都可以看看
```

### 推理字段介绍

**入参**

> 不同平台不同提供商不同，常见有以下几种（不一定对，仅供参考）：
> 
> | 平台/模型      | 开启推理参数                                |
> | -------------- | ------------------------------------------- |
> | OpenAI o1 / o3 | 自动启用，无需传参                          |
> | DeepSeek-R1    | `extra_body={"thinking":{"type":"enable"}}` |
> | Qwen2.5-Think  | `extra_body={"enable_thinking":True}`       |
> | Yi-Reasoning   | `extra_body={"reasoning_mode":"high"}`      |

**返回**

> 在新版的 OpenAI 官方 SDK（>= v1.0）中：
> - OpenAI SDK 采用了严格的 Pydantic Schema 定义，
> - 它会把「未定义的额外字段」自动收集到`model_extra`里。
> 
> 也就是说：
> - 如果服务器响应中包含了 SDK 未定义的字段；
> - 或者返回结构扩展了（例如厂商自定义的 reasoning 字段）；
> - 那么 SDK 不会报错，而是把这些未知字段都放进 model_extra。
> 
> 换句话说：
> - 如果 SDK 不认识服务器返回的键，就会自动塞进 model_extra。
> - 所以你看到 reasoning 在 model_extra 里，其实说明：模型真的返回了 reasoning；但 SDK 暂时还没内置该字段。
> 
> **最佳操作建议**
> 
> | 操作                   | 建议                                                                                |
> | ---------------------- | ----------------------------------------------------------------------------------- |
> | 想取推理文本           | 优先看 `message.reasoning_content`，其次 `message.model_extra["reasoning_content"]` |
> | 想取推理元信息（若有） | 检查 `reasoning_details`（同理在 `model_extra`）                                    |
> | **查看原始返回**       | 打印 `response.model_dump()` 看全量 JSON                                            |


## 直接采用post请求

其实 openai 的 sdk 最后也是使用 post。

所以也可以不使用 openai 的 sdk，直接采用`requests.post`的方式来调用。

看[请求llm](#请求llm)最开头的介绍和脚本就懂了。

```python
# 请求: 正常情况下用这种方式都可以
response = requests.post(url, headers=headers, data=json.dumps(data))
# 不过有些部署方式有规定 data 的形式，比如 lightllm 部署的 /generate 接口就要求: data = {"inputs": "", "parameters": {}}

# 返回:
    # 1. 一般的 /v1/chat/completions 接口返回: model_response = response.json(), 主要 content 字段: model_response["choices"][0]["message"]["content"]
    # 2. lightllm 的 /generate 接口返回: model_response = response['generated_text'][0]
```
