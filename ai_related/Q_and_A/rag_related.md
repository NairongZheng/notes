## RAG相关

<details>
<summary>RAG（检索增强生成，Retrieval-Augmented Generation）技术</summary>

<br>

**RAG 是什么**

> RAG是一种将外部知识检索（Retrieval）与文本生成模型（如 GPT、BERT 等）结合的架构。
> **核心思想**：与其让大模型“死记硬背”所有知识，不如让它在生成时“查资料”！
> 它的目标是**增强语言模型的知识覆盖和实时性**，尤其在问答、聊天、摘要等任务中，**避免模型“胡编”或“知识过时”**。

**RAG 的完整链路包含哪些步骤**

> RAG 的链路分为两个阶段：
> 
> 1. 检索（Retrieval）阶段：找到相关知识
> 2. 生成（Generation）阶段：基于知识回答问题
> 
> 可以将其细化为以下步骤：
> 
> 1. 用户输入 Query：用户给出一个问题、指令或待补全文本。
> 2. Query 编码成向量（Query Embedding）：使用一个编码器（如 BERT、MiniLM、sentence-transformer）将 query 编码为一个向量，表示其语义含义。
> 3. 向量检索（Dense Retrieval）或关键词检索（Sparse Retrieval）：
>    1. 在一个大型文档数据库中（如企业知识库、PDF、网页等），**使用 Q 向量去检索与其最相似的 K 条文本（top-K 文档）**
>    2. 常用方式：向量检索：基于 FAISS、Milvus 等 ANN 引擎；稀疏检索：如 BM25
>    3. 如果是 hybrid retrieval（混合检索），会结合向量 + 关键词两个角度。
> 4. 构建上下文：将检索到的多个段落拼接成一个 Prompt context，准备作为语言模型的输入。
> 5. 输入到大模型进行生成：将拼接后的上下文作为 prompt 输给大语言模型（如 GPT-3.5/4、LLaMA、Claude），生成最终回答。

**为什么 RAG 能提升生成质量**

> 1. **提升事实性和准确率**：大模型训练截止时间有限（如 GPT-4 截止到 2023/2024），知识会过时。而 RAG 可以实时检索 最新的、权威的数据源，显著降低“胡编乱造”风险。
> 2. **提升知识覆盖范围（知识外推）**：即使是 1T 参数的大模型也不可能记住所有知识，比如一些公司内部的手册等。**RAG 让语言模型具备外部知识接入口，不再局限于训练数据**。
> 3. **增强模型解释能力和引用能力**：很多 RAG 系统（如 Bing Chat、Perplexity.ai）会直接在回答中标注引用来源，让用户能验证事实来源，增加可追溯性和透明性。
> 4. **对长文档、长上下文任务更友好**：通过 chunk 分段检索和语义匹配，**RAG 可以让模型“聚焦”在相关段落**，不受最大 token 长度的限制（相比直接输入长 prompt 更高效、准确）。
> 5. **灵活性强，易更新，不需要微调大模型**：RAG 只需更新文档库，或更新向量索引，无需重新训练语言模型。这比重新训练 GPT 等模型成本低很多。

</details>



<details>
<summary>RAG 过程中的 Query 改写技术</summary>

<br>

**为什么要改写 Query？**

> 在 RAG（Retrieval-Augmented Generation）过程中，对用户 query 进行改写（Query Rewriting）可以大幅提升检索质量，特别是在多轮对话或用户输入模糊、简略的情况下。
> 
> 用户原始 query 可能存在以下问题：
> 
> | 问题类型 | 示例 | 问题描述 |
> |---------|------|----------|
> | 语义不完整 | "他是谁？" | 缺少上下文，难以检索 |
> | 用词模糊 | "资料" | 不清楚是技术资料、背景资料还是什么 |
> | 多轮省略 | "这个公司在哪？" | 无法明确是哪一个"公司" |
> | 检索不精确 | "请介绍一下GPT的应用" | 太宽泛，不利于定位相关 chunk |

**Query 改写的方式（4类）**

> RAG 系统中的 Query 改写主要分为四种类型，每种针对不同的场景和问题：
> 
> **1. 上下文增强型改写（Contextual Rewriting）**
> 
> 用于多轮对话，明确代词、指代、省略等。结合历史对话记录，生成清晰的检索 query。
> 
> **示例：**
> - 原始输入： "他是谁？"
> - 改写后： "乔布斯是谁？"（假设上一轮提到乔布斯）
> 
> **适合工具/模型：**
> - T5、ChatGPT、FLAN 微调模型
> - 特定训练的 query-rewriting 模型，如 QReCC 数据集
> 
> **2. 多子问题拆分（Decomposition）**
> 
> 将复杂 query 拆分成多个检索子问题，适合开放问答或复杂 reasoning 的场景。
> 
> **示例：**
> - 原始输入： "介绍一下 GPT-4 的原理和历史发展"
> - 改写为两个 query：
>   - "GPT-4 的原理是什么？"
>   - "GPT 模型的发展历史是怎样的？"
> 
> **适合工具/模型：**
> - AutoDecompose
> - ReACT + RAG
> - LangChain、LlamaIndex 的高级 routing 组件
> 
> **3. 关键词抽取 / 扩展型（Keyword Extraction / Expansion）**
> 
> 从 query 中提取实体、关键短语、术语，或加入同义词、上下义词增强 recall。
> 
> **示例：**
> - 原始输入： "怎么用 Milvus 做相似度检索？"
> - 改写后： "Milvus 向量数据库 相似度搜索 示例 实现方法"
> 
> **实现方式：**
> - 使用 keyBERT、spaCy、TextRank 抽取关键词
> - 使用 word embedding / embedding similarity 扩展术语
> 
> **4. 基于 Prompt 的语义重写（LLM Rewrite）**
> 
> 用大语言模型直接用 prompt 生成改写 query。
> 
> **示例 Prompt：**
> ```
> 请将以下查询改写成适合用于知识库检索的清晰问题：
> 原始问题：他是谁？
> 背景对话：上一轮用户问的是乔布斯
> 输出：
> 乔布斯是谁？
> ```
> 
> **提示工程技巧：**
> - 加入 Instruction（改写为清晰问题）
> - 结合上下文 history
> - 控制输出风格（如简洁 / 多样 / 多个候选）

**实际系统中的改写流程（RAG 前处理）**

```
[用户 query]
     ↓
[Query Rewriter 模块] ← (可选:历史对话)
     ↓
[向量化检索引擎]（如 FAISS、Milvus）
     ↓
[文档 chunk 召回]
     ↓
[LLM + Prompt + RAG 生成答案]
```

**常用的 Query Rewrite 工具/模块**

> | 名称/方法 | 说明 |
> |----------|------|
> | QReCC | 微调数据集，多轮问答改写 |
> | LlamaIndex QueryTransformer | 插件式 query 改写模块 |
> | LangChain ConversationalRetrievalChain | 自动接管历史对话 + 改写 |
> | ChatGPT / GPT-4 | 提示生成清晰检索语句 |
> | R2R、T5-QReCC | 微调后的 Seq2Seq 模型用于改写 |

**总结**

> | 目标 | 方法 |
> |------|------|
> | 解决多轮省略 / 指代 | 上下文重写（Context Rewriting） |
> | 提高 recall | 关键词扩展 / embedding 近义词 |
> | 提高 precision | prompt 精炼查询 / 子问题拆解 |
> | 易用性 | LLM prompt 即时改写或微调模型调用 |

</details>