- [basic](#basic)
- [部署llm](#部署llm)
- [请求llm](#请求llm)
  - [OpenAI \& AzureOpenAI](#openai--azureopenai)
    - [推理字段介绍](#推理字段介绍)
  - [直接采用post请求](#直接采用post请求)
- [some\_pkg](#some_pkg)
  - [AutoTokenizer](#autotokenizer)
  - [AutoConfig \& AutoConfig](#autoconfig--autoconfig)
  - [AzureOpenAI](#azureopenai)


# basic

```shell
.
├── config.json                        # 模型结构配置文件，定义模型的结构参数，比如隐藏层维度、层数、注意力头数、RoPE 范围、词表大小等。
├── generation_config.json             # 文本生成参数默认值。定义推理时的默认参数：如 max_new_tokens、temperature、top_p、do_sample 等。被 model.generate() 调用时使用。
├── chat_template.jinja                # message组织的模板，hugging face会通过这个模板将message组织成文本传给llm。
├── merges.txt                         # BPE 合并规则。定义哪些字符对可以合并成新 token，例如 “th” + “e” → “the”。BPE 分词器必须配合 vocab.json 使用。
├── model-00001-of-00008.safetensors   # 模型权重文件（分片）
├── model-00002-of-00008.safetensors
├── model-00003-of-00008.safetensors
├── model-00004-of-00008.safetensors
├── model-00005-of-00008.safetensors
├── model-00006-of-00008.safetensors
├── model-00007-of-00008.safetensors
├── model-00008-of-00008.safetensors
├── model.safetensors.index.json       # 指明每个权重张量在哪个分片文件里。例如 "model.layers.0.self_attn.q_proj.weight" 在第 2 个 safetensor 文件。模型加载时用它拼接参数。
├── tokenizer_config.json              # 分词器的配置文件。定义分词器类名、特殊 token（BOS、EOS、PAD、UNK）等元信息。
├── tokenizer.json                     # 分词器完整定义（Hugging Face 格式）。包含分词算法、词表、merge 规则等，是最主要的 tokenizer 文件。
└── vocab.json                         # 词表文件（BPE 或 SentencePiece）。旧版兼容文件，部分 tokenizer 会用它和 merges.txt 一起加载。
```

- 模型核心文件
  - config.json：结构蓝图
  - model.safetensors.\*：权重数据
  - model.safetensors.index.json：装配说明书
- 分词器（Tokenizer）相关文件（tokenizer.json 已经足够完整，但 vocab.json + merges.txt 会保留是为了兼容旧版加载方式。）
  - tokenizer.json：分词器完整定义（Hugging Face 格式）
  - tokenizer_config.json：分词器的配置文件
  - vocab.json：词表文件（BPE 或 SentencePiece）
  - merges.txt：BPE 合并规则
- 生成配置文件
  - generation_config.json：文本生成参数默认值

<details>

<summary>config.json</summary>

```shell
{
  "architectures": ["QwenForCausalLM"],
  "model_type": "qwen",
  "hidden_size": 5120,
  "intermediate_size": 13824,
  "num_attention_heads": 40,
  "num_hidden_layers": 40,
  "vocab_size": 151936,
  "max_position_embeddings": 32768,
  "rms_norm_eps": 1e-5,
  "rope_scaling": {"type": "linear", "factor": 2.0},
  "use_cache": true,
  "torch_dtype": "bfloat16"
}
```

| 字段                    | 含义                                                                    |
| ----------------------- | ----------------------------------------------------------------------- |
| architectures           | 模型对应的类名（Transformers 自动加载时使用）。例如 "QwenForCausalLM"。 |
| model_type              | 模型家族类型（如 "qwen", "llama", "gpt2"），决定加载逻辑。              |
| hidden_size             | 每层 Transformer 的隐藏维度。                                           |
| intermediate_size       | FFN 层（前馈层）的中间维度。                                            |
| num_attention_heads     | 注意力头数量。                                                          |
| num_hidden_layers       | Transformer 层数。                                                      |
| vocab_size              | 词表大小，对应 tokenizer。                                              |
| max_position_embeddings | 最大序列长度。                                                          |
| rms_norm_eps            | RMSNorm 中的 epsilon。                                                  |
| rope_scaling            | 对于 RoPE（旋转位置编码）的扩展方式。                                   |
| use_cache               | 是否在生成时使用 KV cache。                                             |
| torch_dtype             | 权重数据类型（如 bfloat16、float16）。                                  |

</details>


<details>

<summary>model.safetensors.index.json</summary>

```shell
{
  "metadata": {
    "total_size": 31987654321
  },
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00008.safetensors",
    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00008.safetensors",
    "model.layers.0.self_attn.k_proj.weight": "model-00002-of-00008.safetensors",
    ...
  }
}
```

| 字段                | 含义                                                       |
| ------------------- | ---------------------------------------------------------- |
| metadata.total_size | 所有分片权重文件加起来的总大小（字节）。                   |
| weight_map          | 每个参数张量的路径映射（键为参数名，值为所在分片文件名）。 |

</details>

<details>

<summary>generation_config.json</summary>

定义模型生成时的默认超参数。

```shell
{
  "max_new_tokens": 512,
  "temperature": 0.8,
  "top_p": 0.9,
  "do_sample": true,
  "repetition_penalty": 1.05,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0
}
```

| 字段                                       | 含义                        |
| ------------------------------------------ | --------------------------- |
| max_new_tokens                             | 最大生成 token 数。         |
| temperature                                | 采样温度，控制生成多样性。  |
| top_p                                      | nucleus sampling 截断阈值。 |
| do_sample                                  | 是否启用采样模式。          |
| repetition_penalty                         | 重复惩罚因子。              |
| bos_token_id / eos_token_id / pad_token_id | 特殊 token 的 ID。          |

</details>

<details>

<summary>tokenizer.json</summary>

这是最重要的 tokenizer 文件，结构较复杂，包含分词算法、词表、合并规则等。

```shell
{
  "model": {
    "type": "BPE",
    "vocab": {"the": 0, "a": 1, "an": 2, ...},
    "merges": ["t h", "th e", ...]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<bos>", "special": true},
    {"id": 2, "content": "<eos>", "special": true}
  ],
  "normalizer": {"type": "Sequence", "normalizers": [{"type": "NFKC"}]},
  "pre_tokenizer": {"type": "ByteLevel"},
  "post_processor": {"type": "ByteLevel"}
}
```

| 字段           | 含义                                         |
| -------------- | -------------------------------------------- |
| model.type     | 分词算法类型（BPE、Unigram、WordPiece 等）。 |
| vocab          | token → id 的映射表。                        |
| merges         | 字符组合规则（仅 BPE 模型有）。              |
| added_tokens   | 特殊 token 定义。                            |
| normalizer     | 文本归一化规则（如小写、NFKC 形式）。        |
| pre_tokenizer  | 切分前的预处理方式。                         |
| post_processor | 添加特殊符号的逻辑。                         |

</details>

<details>

<summary>tokenizer_config.json</summary>

这是 tokenizer 的“元信息”文件，Hugging Face 自动生成。

```shell
{
  "tokenizer_class": "QwenTokenizer",
  "do_lower_case": false,
  "model_max_length": 32768,
  "padding_side": "right",
  "truncation_side": "right",
  "special_tokens_map_file": null,
  "bos_token": "<bos>",
  "eos_token": "<eos>",
  "pad_token": "<pad>"
}
```

| 字段                              | 含义               |
| --------------------------------- | ------------------ |
| tokenizer_class                   | 对应的分词器类。   |
| model_max_length                  | 最大可编码长度。   |
| do_lower_case                     | 是否将输入转小写。 |
| padding_side                      | pad 填充方向。     |
| bos_token / eos_token / pad_token | 特殊 token 定义。  |

</details>

<details>

<summary>vocab.json</summary>

BPE 分词器的核心之一。内容是一个 token→id 字典：

```shell
{
  "the": 0,
  "a": 1,
  "an": 2,
  "##ing": 3,
  ...
}
```

通常配合 merges.txt 一起使用。

</details>

<details>

<summary>merges.txt</summary>

```shell
#version: 0.2
t h
th e
a n
##i ng
```

分词器会根据这些规则逐步合并。

</details>

# 部署llm

可以用lightllm或者vllm部署。

1. vllm：[官方文档](https://docs.vllm.ai/)，[一些参数说明](https://docs.vllm.ai/en/latest/configuration/engine_args.html)
2. lightllm：[官方文档](https://docs.litellm.ai/docs)，[一些参数说明](https://docs.litellm.ai/docs/proxy/cli)

# 请求llm

[参考脚本](./code/llm_api.py)

**几个不同请求接口的介绍（重要！）**

| 接口路径               | 对应模型类型                     | 输入格式                                     | 核心用途                      | 兼容对象                                       |
| ---------------------- | -------------------------------- | -------------------------------------------- | ----------------------------- | ---------------------------------------------- |
| `/v1/chat/completions` | Chat 类型模型（ChatML 模板）     | `messages=[{"role":"user","content":"..."}]` | 多轮对话、工具调用、Reasoning | OpenAI Chat API（GPT 系列）                    |
| `/v1/completions`      | Base 模型（非 Chat 指令模型）    | `prompt="..."`                               | 单轮续写、补全、翻译、改写    | 旧版 OpenAI Completion API（慢慢弃用）         |
| `/generate`            | 通用接口（厂商自定义、很灵活！） | 结构随实现不同（多为 prompt）                | 任意文本生成                  | HuggingFace / vLLM / LightLLM 风格通用生成接口 |

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

> /generate 是 LightLLM / vLLM / Text Generation WebUI / HuggingFace Transformers 等 为方便兼容不同模型而定义的通用路径。
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
>   "generated_text": "XXXXXX"    # 需要自己解析
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

| 对比项       | OpenAI                                          | AzureOpenAI                                        |
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

| 项目                          | OpenAI                         | AzureOpenAI                                 |
| ----------------------------- | ------------------------------ | ------------------------------------------- |
| 认证方式                      | 直接使用 api_key               | 使用 api_key + azure_endpoint + api_version |
| 模型字段                      | 直接用官方模型名（如 "gpt-4o") | 用部署名（如 "gpt4o-deploy"）               |
| 是否要设置 api_version        | 否                             | ✅ 必须                                      |
| Base URL                      | 默认 https://api.openai.com/v1 | 必须设置为 Azure endpoint                   |
| 是否支持自定义网络 / 安全策略 | ❌                              | ✅ 支持企业 VNet、Private Link、合规性控制   |
| 兼容性                        | 用于所有 OpenAI 兼容服务       | 专用于 Azure 环境                           |

### 推理字段介绍

**入参**

> 不同平台不同提供商不同，常见有以下几种（不一定对，仅供参考）：
> 
> | 平台/模型      | 开启推理参数                                                    |
> | -------------- | --------------------------------------------------------------- |
> | OpenAI o1 / o3 | 自动启用，无需传参                                              |
> | DeepSeek-R1    | `extra_body={"thinking":{"type":"enable"}}`                     |
> | Qwen2.5-Think  | `enable_thinking=True` 或 `extra_body={"enable_thinking":True}` |
> | Yi-Reasoning   | `extra_body={"reasoning_mode":"high"}`                          |

**返回**

> 一般会在 `response.choices[0].message.reasoning_content`
> 
> 但是有时候会在 `response.choices[0].message.model_extra`
> 
> **为什么有的模型返回在顶层，有的在 model_extra？**
> 
> 在新版的 OpenAI 官方 SDK（>= v1.0） 以及兼容实现（如 Cloudsway、SiliconFlow、Qwen、DeepSeek）中：
> - OpenAI SDK 采用了严格的 Pydantic Schema 定义，
> - 它会把「模型未定义的额外字段」自动收集到 model_extra 里。
> 
> 也就是说：
> - 如果服务器响应中包含了 SDK 未定义的字段；
> - 或者返回结构扩展了（例如厂商自定义的 reasoning 字段）；
> - 那么 SDK 不会报错，而是把这些未知字段都放进 model_extra。
> 
> | 模型来源                          | SDK 识别状态        | reasoning字段位置                          |
> | --------------------------------- | ------------------- | ------------------------------------------ |
> | OpenAI 官方（如 o1, o3, o4-mini） | ✅ SDK 已注册        | `message.reasoning_content`                |
> | 新模型（如 DeepSeek, Qwen-Think） | ❌ SDK 未注册        | `message.model_extra['reasoning_content']` |
> | Cloudsway / SiliconFlow 兼容API   | ❌ 兼容层不传 schema | `message.model_extra`                      |
> | 自建模型 / proxy                  | ❌ 完全自定义返回    | 可能在任意层（甚至 `choices.model_extra`） |
> 
> 换句话说：
> - 如果 SDK 不认识服务器返回的键，就会自动塞进 model_extra。
> - 所以你看到 reasoning 在 model_extra 里，其实说明：模型真的返回了 reasoning；但 SDK 暂时还没内置该字段。
> 
> **最佳操作建议**
> 
> | 操作                | 建议                                                                                |
> | ------------------- | ----------------------------------------------------------------------------------- |
> | 想取推理文本        | 优先看 `message.reasoning_content`，其次 `message.model_extra["reasoning_content"]` |
> | 想取推理元信息      | 检查 `reasoning_details`（同理在 `model_extra`）                                    |
> | 想知道 SDK 支不支持 | 打印 `dir(response.choices[0].message)`                                             |
> | 想兼容所有服务      | 自己写一个统一 `extract_reasoning()` 函数                                           |
> | 想调试原始返回      | 打印 `response.model_dump()` 看全量 JSON                                            |


## 直接采用post请求

其实就是不使用openai的sdk，直接采用requests.post的方式来调用。看[请求llm](#请求llm)最开头的介绍和脚本就懂了。

```python
response = requests.post(url, headers=headers, data=json.dumps(data))
```

# some_pkg

## AutoTokenizer

[示例脚本](./code/demo_AutoTokenizer.py)

## AutoConfig & AutoConfig

[示例脚本](./code/demo_AutoConfig_and_AutoConfig.py)

## AzureOpenAI

AzureOpenAI及一些llm的api调用，[示例脚本](./code/llm_api.py)