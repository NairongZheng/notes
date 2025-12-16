- [LLM](#llm)
- [basic](#basic)
  - [模型文件介绍](#模型文件介绍)
  - [一些参数介绍](#一些参数介绍)
    - [seq\_len](#seq_len)
    - [dp\_size、tp\_size、pp\_size](#dp_sizetp_sizepp_size)
    - [temperature and top\_p](#temperature-and-top_p)
- [部署llm](#部署llm)
  - [使用 sglang 部署](#使用-sglang-部署)
  - [使用 lightllm 部署](#使用-lightllm-部署)
  - [使用 vllm 部署](#使用-vllm-部署)
- [请求llm](#请求llm)
  - [OpenAI \& AzureOpenAI](#openai--azureopenai)
    - [请求与返回](#请求与返回)
    - [推理字段介绍](#推理字段介绍)
  - [直接采用post请求](#直接采用post请求)
- [查看 tensorboaed](#查看-tensorboaed)
- [some\_pkg](#some_pkg)
  - [AutoTokenizer](#autotokenizer)
  - [AutoModel \& AutoConfig](#automodel--autoconfig)

# LLM

1. code: llm生成的一些示例代码
2. 大模型微调
3. Attention: LLM的注意力机制，包括MHA、MQA、GQA、MLA
4. mcp: Model Context Protocol
5. model_train: 模型训练，主要是并行训练/分布式训练的介绍
6. temperature_and_top_p: Temperature跟top_p的介绍
7. tool_calls: 工具调用


# basic

## 模型文件介绍

下面用[Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B/tree/main)举例

```shell
.
├── config.json                        # 模型结构配置文件，定义模型的结构参数，比如隐藏层维度、层数、注意力头数、RoPE 范围、词表大小等。
├── generation_config.json             # 文本生成参数默认值。定义推理时的默认参数：如 max_new_tokens、temperature、top_p、do_sample 等。被 model.generate() 调用时使用。
├── chat_template.jinja                # message组织的模板，hugging face会通过这个模板将message组织成文本传给llm。（Qwen3-14B直接放在tokenizer_config中）
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
├── tokenizer_config.json              # 分词器的配置文件。定义分词器类名、特殊 token（BOS、EOS、PAD、UNK）等元信息。(一些模型也会把chat template放在这里)
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
  "model_type": "qwen3",
  "hidden_size": 5120,
  "intermediate_size": 17408,
  "num_attention_heads": 40,
  "num_hidden_layers": 40,
  "vocab_size": 151936,
  "max_position_embeddings": 40960,
  "rms_norm_eps": 1e-6,
  "rope_scaling": null,
  "use_cache": true,
  "torch_dtype": "bfloat16",
  ...
}
```

| 字段                    | 含义                                                                    |
| ----------------------- | ----------------------------------------------------------------------- |
| architectures           | 模型对应的类名（Transformers 自动加载时使用）。例如 "QwenForCausalLM"。 |
| model_type              | 模型家族类型（如 "qwen3", "llama", "gpt2"），决定加载逻辑。             |
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
    "total_size": 29536614400
  },
  "weight_map": {
    "lm_head.weight": "model-00008-of-00008.safetensors",
    "model.embed_tokens.weight": "model-00001-of-00008.safetensors",
    "model.layers.0.input_layernorm.weight": "model-00001-of-00008.safetensors",
    "model.layers.0.mlp.down_proj.weight": "model-00001-of-00008.safetensors",
    "model.layers.0.mlp.gate_proj.weight": "model-00001-of-00008.safetensors",
    "model.layers.0.mlp.up_proj.weight": "model-00001-of-00008.safetensors",
    "model.layers.0.post_attention_layernorm.weight": "model-00001-of-00008.safetensors",
    "model.layers.0.self_attn.k_norm.weight": "model-00001-of-00008.safetensors",
    "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00008.safetensors",
    "model.layers.0.self_attn.o_proj.weight": "model-00001-of-00008.safetensors",
    "model.layers.0.self_attn.q_norm.weight": "model-00001-of-00008.safetensors",
    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00008.safetensors",
    "model.layers.0.self_attn.v_proj.weight": "model-00001-of-00008.safetensors",
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
    "transformers_version": "4.51.0",
    "bos_token_id": 151643,
    "do_sample": true,
    "eos_token_id": [
        151645,
        151643
    ],
    "pad_token_id": 151643,
    "repetition_penalty": 1.05,
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.95,
    "max_length": 1024,
    "max_new_tokens": 512,
}
```

| 字段               | 含义                                                                                                                                                                                                     |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| bos_token_id       | Beginning Of Sentence Token，句子起始 token ID（模型输入的开头符号）。<br> Qwen3 使用自己的特殊 token：151643 → "<\|endoftext\|>"（为什么不是151644 "<\|im_start\|>"）                                   |
| do_sample          | 是否启用采样模式。<br> 1. true → 使用 temperature / top-p 随机采样） <br> 2. false → 使用 greedy decoding（贪心，确定性）                                                                                |
| eos_token_id       | End Of Sentence Tokens（结束标记），Qwen3 设置了 多个结束符，包括：<br> 1. 151645 →  "<\|im_end\|>" <br> 2. 151643 →  "<\|endoftext\|>"（与 bos_token_id 同 ID 在 Qwen3 是合理的，因为会复用同类 token） |
| pad_token_id       | 处理 batch 时用于填充的 token ID。                                                                                                                                                                       |
| repetition_penalty | 重复惩罚因子。用于减少模型重复生成内容的概率。值越大，惩罚越强。<br> 1. 1.0 → 不惩罚 <br> 2. 1.05 → 轻微惩罚（Qwen3 默认值）<br> 3. 1.2 → 文本质量会下降（太严格会变怪）                                 |
| temperature        | 采样温度，控制生成多样性。<br> 1. 低温（0～0.7）：更稳、更像“确定回答” <br> 2. 高温（0.8～1.5）：更创造性、更多样                                                                                        |
| top_k              | 只从最高概率的 k 个 token 中采样（通常为0，表示不启用） <br> 1. 较低（如 20）：更稳定、更收敛 <br> 2. 较高（如 50+）：更发散                                                                             |
| top_p              | 核采样（nucleus sampling）。在概率累积达到 0.95 的 token 子集中采样。<br> 1. 低值（0.7） → 模型更确定 <br> 2. 高值（0.95~1.0） → 更多样化                                                                |
| max_new_tokens     | 限制最大生成 token 数。（优先级比max_length高，有时候混用会代表max_length）                                                                                                                              |
| max_length         | 限制输入+输出总长度（不推荐单独使用）                                                                                                                                                                    |

</details>

<details>

<summary>tokenizer.json</summary>

这是最重要的 tokenizer 文件，结构较复杂，包含分词算法、词表、合并规则等。

```shell
{
    "version": "1.0",
    "trunction": null,
    "padding": null,
    "added_tokens": [
        {"id": 151643, "content": "<endoftext>", "special": true},
        {"id": 151655, "content": "<im_start>", "special": true},
        {"id": 151645, "content": "<im_end>", "special": true},
        ...
    ],
    "normalizer": {"type": "NFC"},
    "pre_tokenizer": {
        "type": "Sequence",
        "pretokenizers": [
            {
                "type": "Split",
                "pattern": {"Regex": ""},
                "behavior": "Isolated",
                "invert": false
            },
            {
                ...
            }
        ],
    },
    "post_processor": {
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": false,
        "use_regex": false
    },
    "decoder": {
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": false,
        "use_regex": false
    },
    "model": {
        "type": "BPE",
        "dropout": null,
        "vocab": {"the": 0, "a": 1, "an": 2, ...},
        "merges": ["t h", "th e", ...]
    }
}
```

| 字段           | 含义                                                                                                                     |
| -------------- | ------------------------------------------------------------------------------------------------------------------------ |
| trunction      | null 表示分词器层面 未启用任何截断规则。模型的最大长度截断由模型控制（generate() / model.config）而不是 tokenizer 控制。 |
| padding        | null 表示分词器不自动填充                                                                                                |
| added_tokens   | 这里列出 不属于 BPE 词表、但需要特殊处理的 token。                                                                       |
| normalizer     | 文本归一化规则。保证不同输入的 Unicode 变体映射到相同 token                                                              |
| pre_tokenizer  | pre-tokenizer 决定了如何把原始文本切分成“初步 token”再喂给 BPE 模型。                                                    |
| post_processor | 后处理步骤，在 BPE 分词完成后执行。保证解码时：还原原始空格，还原原始字节序列                                            |
| decoder        | 用于“解码”token → 文本。                                                                                                 |
| model          | 这是核心，BPE模型本体。                                                                                                  |
| model.type     | 分词算法类型（BPE、Unigram、WordPiece 等）。                                                                             |
| model.vocab    | token → id 的映射表。                                                                                                    |
| model.merges   | 表示要合并哪些字符对：先把文本按字节切成 token，再根据 merges 规则合并为更大的语言单元                                   |


</details>

<details>

<summary>tokenizer_config.json</summary>

这是 tokenizer 的“元信息”文件，Hugging Face 自动生成。

```shell
{
    "add_bos_token": false,
    "add_prefix_space": false,
    "added_tokens_decoder": {
        "151643": {...},
        "151644": {...},
        ...
    },
    "additional_special_tokens": [
        "<|im_start|>",
        "<|im_end|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>"
    ],
    "bos_token": null,
    "chat_template": "...",
    "clean_up_tokenization_spaces": false,
    "eos_token": "<|im_end|>",
    "errors": "replace",
    "model_max_length": 131072,
    "pad_token": "<|endoftext|>",
    "split_special_tokens": false,
    "tokenizer_class": "Qwen2Tokenizer",
    "unk_token": null
}
```

| 字段                         | 含义                                                                                                                                                                          |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| add_bos_token                | 是否自动在 序列开头加入 BOS                                                                                                                                                   |
| add_prefix_space             | 是否在文本开头自动添加一个空格                                                                                                                                                |
| added_tokens_decoder         | HuggingFace 将 tokenizer.json 中的 "added_tokens" 反向映射到一个 dict <br> decoder 需要知道 ID → token 的映射 <br> 包含特殊符号如 "<\|im_start\|>"、视觉 token 等             |
| additional_special_tokens    | 列出了所有额外添加的特殊 token（不属于 BPE vocabulary）                                                                                                                       |
| chat_template                | 将对话结构转换为模型 input 格式，`tokenizer.apply_chat_template()` 时使用                                                                                                     |
| clean_up_tokenization_spaces | 是否在 decode 时清理多余空格                                                                                                                                                  |
| eos_token                    | 模型遇到该 token 将停止生成。                                                                                                                                                 |
| errors                       | 这是 Python 的字符串错误处理策略：tokenizer 在 decode 时遇到非法字节 → 替换为 "�"                                                                                             |
| model_max_length             | 模型能够处理的最大上下文长度（token 数）。用于 tokenizer 层面做截断检查，这是硬上限，即你不能给模型输入长度超过这个数，否则会报错。（注意与max_length跟max_new_tokens的区别） |
| split_special_tokens         | 是否允许对特殊 token 进行拆分，必须为 false，否则 "<\|im_end\|>" 会被拆坏成 "<" "\|" "im_end" ...                                                                             |

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

## 一些参数介绍

这边会介绍一些训练中可能会出现的参数

### seq_len

> （序列长度 / context length）
> 单条训练样本的最大 token 长度，即模型可以看到的**上下文窗口大小**。
> - 决定 Transformer 的注意力矩阵大小：`seq_len × seq_len`
> - 决定 GPU 显存消耗（注意力成本 ~ `O(seq_len²)`）
> - 决定模型的上下文能力（长上下文任务必然要更大的 seq_len）
> 
> 常见用法：
> 
> ```shell
> seq_len=4096   # 训练 4K 上下文
> ```
> 
> - seq_len 越大，显存开销越高。
> - 推理时也受限于此值，但训练时可以采用 RoPE scaling、YaRN、NTK scaling 允许推理更长上下文。

### dp_size、tp_size、pp_size

满足：$\text{总GPU数}=\text{dp\_size}\times\text{tp\_size}\times\text{pp\_size}$

> 具体查看 [model_train](./model_train.md) 中的相关介绍。

### temperature and top_p

> 详见 [temperature_and_top_p](./temperature_and_top_p.md)

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

> 注意请求中的 extra_body 里面的各个字段实际上是解开传进去的，而不是直接传 "extra_body" 这个字段

**返回**

> 模型的返回，两个包都会包装好再返回，跟 requests 方式返回的 dict 有区别，但本质上一样的，因为最后都是使用的 requsets.post 发送请求的。


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

其实 openai 的 sdk 最后也是使用 post。

所以也可以不使用 openai 的 sdk，直接采用requests.post的方式来调用。

看[请求llm](#请求llm)最开头的介绍和脚本就懂了。

```python
# 请求: 正常情况下用这种方式都可以
response = requests.post(url, headers=headers, data=json.dumps(data))
# 不过有些部署方式有规定 data 的形式，比如 lightllm 部署的 /generate 接口就要求: data = {"inputs": "", "parameters": {}}

# 返回:
    # 1. 一般的 /v1/chat/completions 接口返回: model_response = response.json(), 主要 content 字段: model_response["choices"][0]["message"]["content"]
    # 2. lightllm 的 /generate 接口返回: model_response = response['generated_text'][0]
```

# 查看 tensorboaed

```shell
tensorboard --logdir <tensorboard_dir> --port <port>
```

# some_pkg

## AutoTokenizer

[示例脚本](./code/demo_AutoTokenizer.py)

## AutoModel & AutoConfig

[示例脚本](./code/demo_AutoModel_and_AutoConfig.py)
