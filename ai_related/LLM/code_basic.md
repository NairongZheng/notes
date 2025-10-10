- [basic](#basic)
- [some\_pkg](#some_pkg)
  - [AutoTokenizer](#autotokenizer)
  - [AutoConfig \& AutoConfig](#autoconfig--autoconfig)


# basic

```shell
.
├── config.json                        # 模型结构配置文件，定义模型的结构参数，比如隐藏层维度、层数、注意力头数、RoPE 范围、词表大小等。
├── generation_config.json             # 文本生成参数默认值。定义推理时的默认参数：如 max_new_tokens、temperature、top_p、do_sample 等。被 model.generate() 调用时使用。
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

# some_pkg

## AutoTokenizer

[示例脚本](./code/demo_AutoTokenizer.py)

## AutoConfig & AutoConfig

[示例脚本](./code/demo_AutoConfig_and_AutoConfig.py)
