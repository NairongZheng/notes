- [LLM](#llm)
- [basic](#basic)
  - [模型文件介绍](#模型文件介绍)
  - [一些参数介绍](#一些参数介绍)
  - [资源计算(参数量、存储、显存)](#资源计算参数量存储显存)
- [部署与请求](#部署与请求)
- [查看 tensorboaed](#查看-tensorboaed)
- [some\_pkg](#some_pkg)
  - [AutoTokenizer](#autotokenizer)
  - [AutoModel \& AutoConfig](#automodel--autoconfig)

# LLM

1. code: llm生成的一些示例代码
2. mcp: Model Context Protocol
3. transformer: 详细介绍 transformer 的各个部分
4. Attention: LLM的注意力机制，包括MHA、MQA、GQA、MLA、DSA
5. deploy_and_request: 部署和请求LLM
6. fine_tuning: 大模型微调
7. model_train: 模型训练，主要是并行训练/分布式训练的介绍
8. probability_theory_in_LLM: 大语言模型中的概率论
9. temperature_and_top_p: Temperature跟top_p的介绍
10. tool_calls: 工具调用


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

**seq_len**

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

**dp_size、tp_size、pp_size**

> 满足：$\text{总GPU数}=\text{dp\_size}\times\text{tp\_size}\times\text{pp\_size}$
> 具体查看 [model_train](./model_train.md) 中的相关介绍。

**temperature and top_p**

> 详见 [temperature_and_top_p](./temperature_and_top_p.md)

## 资源计算(参数量、存储、显存)

假设现在模型参数如下：

```shell
{
    "hidden_size": 5120,            # Transformer 主维度（d_model）
    "intermediate_size": 17408,     # FFN 中间层
    "num_attention_heads": 40,      # Transformer Block 数
    "num_hidden_layers": 40,        # 注意力头数
    "vocab_size": 151936            # 词表大小
}
```

**参数量计算**

> Attention 层参数：$4\times \text{hidden\_size}^2$
> - Q、K、V 权重：$3\times \text{hidden\_size}^2$
> - 线性输出层：$\text{hidden\_size}^2$
> 
> FFN 层（两次线性变换）参数：$2\times \text{hidden\_size} \times \text{intermediate\_size}$
> 
> 词表 Embedding 与输出层（量级较小）：
> - $\text{Embedding}=\text{vocab\_size}\times\text{hidden\_size}$
> - $\text{Output}$：可以与 Embedding 共享
> 
> 该例子总量计算：
> 
> ```shell
> Attention per layer = 4 * 5120^2 = 4 * 26,214,400 = 104,857,600
> FFN per layer       = 2 * 5120 * 17,408 ≈ 178,257,920
> === per layer total ≈ 283M params
> 
> Total layers = 40
> Total params from blocks = 40 * 283M ≈ 11.32B
> 
> + Embedding ≈ 151,936 * 5,120 ≈ 778M
> ```
> 
> 加上 bias / LayerNorm，该模型大概是 14B 大小。

**储存计算**

> $\text{占用存储}=\text{参数量}\times\text{每参数占用字节数}$
> 
> 不同数据格式占用字节数：
> 
> | 类型        | Bytes/参数 |
> | ----------- | ---------- |
> | FP32        | 4          |
> | FP16 / BF16 | 2          |
> | INT8        | 1          |
> | INT4        | 0.5        |
> 
> 该模型是 FP16 / BF16，所以大概是（换算成常见单位）：
> 
> ```shell
> 14B × 2 bytes = 28 GB
> 28 × (1000/1024)^3 ≈ 26 GiB
> ```

**训练显存计算**

> 训练显存比推理要大很多，因为不仅要存参数，还要存梯度、优化器状态、激活值。
> 
> | 类别             | 占显存                                                        |
> | ---------------- | ------------------------------------------------------------- |
> | Parameters       | 模型权重                                                      |
> | Gradients        | 与权重同量级                                                  |
> | Optimizer states | 例如 Adam 有 2 个额外状态（一阶动量，二阶动量）               |
> | Activations      | 前向中间值（反向传播必须使用的中间张量与 batch、seq线性相关） |
> | 额外开销         | CUDA / 框架占用                                               |
> 
> 该模型训练时，大概需要显存：
> 
> ```shell
> 模型权重 = 14B × 2 bytes = 28 GB
> 梯度 = 28 GB
> Optimizer = 14B × 4 bytes = 58 GB
> 
> 以上基础部分，再加上激活值、batch、seq、CUDA 开销等
> 实际训练显存可能 > 140 GB
> ```

**推理显存计算**

> 推理时不需要梯度和优化器状态，推理显存需求主要包括：
> - 参数权重
> - KV Cache（对长上下文）
> - Batch / Seq / 框架开销
> 
> 该模型推理时，大概需要显存：
> 
> ```shell
> 14B × 2 bytes × 1.1 ≈ 30.8 GB
> ```

# 部署与请求

详见 [deploy_and_request](./deploy_and_request.md)

# 查看 tensorboaed

```shell
tensorboard --logdir <tensorboard_dir> --port <port>
```

# some_pkg

## AutoTokenizer

[示例脚本](./code/demo_AutoTokenizer.py)

## AutoModel & AutoConfig

[示例脚本](./code/demo_AutoModel_and_AutoConfig.py)
