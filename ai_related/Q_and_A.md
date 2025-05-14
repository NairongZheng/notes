## 深度学习基础

<details>
<summary>查全率、查准率、PR曲线、ROC曲线</summary>

<br>

<table>
    <tr align='center'>
        <th rowspan ='2'>真实情况</th>
        <th colspan ='2'>预测结果</th>
    </tr>
    <tr align='center'>
        <th colspan ='1'>正例</th>
        <th colspan ='1'>反例</th>
    </tr>
    <tr  align='center'>
        <td>正例</td>
		<td>TP(真正例)</td>
        <td>FN(假反例)</td>
    </tr>
    <tr  align='center'>
        <td>反例</td>
		<td>FP(假正例)</td>
        <td>TN(真反例)</td>
    </tr>
</table>

- **准确率（Accuracy）**：对于给定的测试数据集，分类正确的样本数与总样本数之比
  
$$
\frac{TP+TN}{总样本数}
$$

- **精确率/查准率（Precision）**：预测为正的样本中，又多少是真正的正样本

$$
\frac{TP}{TP+FP}
$$

- **召回率/查全率（Recall）**：样本中有多少正例被预测正确了

$$
\frac{TP}{TP+FN}
$$


**查准率和查全率是一对矛盾的度量**。

**PR曲线**：以查准率为纵轴、查全率为横轴作图 ，就得到了查准率-查全率曲线。

![](../images/20220702/20220702_1_机器学习.jpg)

**ROC曲线**：以​​假正率（FPR）​​为横轴，​​真正率（TPR）​​为纵轴，反映模型在不同阈值下的分类性能。

| 维度​          | ​	​​ROC曲线​                     | ​	​​PR曲线​​                     |
| -------------- | -------------------------------- | -------------------------------- |
| ​横轴​​	​​     | 假正率（FPR）	​​                 | 召回率（Recall）                 |
| ​纵轴​​​       | ​	真正率（TPR）​​                | 精确率（Precision）              |
| ​敏感度​​	​​   | 对类别平衡数据更敏感	​​          | 对类别不平衡数据更敏感           |
| ​典型场景​​	​​ | 医疗诊断、金融风控（平衡数据）​​ | 欺诈检测、推荐系统（正样本极少） |
| ​AUC意义​​	​​  | ROC-AUC越高，整体分类性能越好​​  | PR-AUC越高，正样本识别能力越强   |

</details>


## 大模型相关

<details>
<summary>transformer 中为什么使用 layer normalization 而不是用 batch normalization</summary>

<br>

> 1. 对批次大小的敏感性​​
>    1. ​批归一化（BN）​​：依赖于当前批次的统计量（均值和方差），​**在​小批次或批次大小变化时​​表现不稳定**。例如，在自然语言处理（NLP）任务中，由于**句子长度不同**，常需动态填充（padding）或截断，导致批次内有效样本数不一致，影响BN的统计量计算。
>    2. ​层归一化（LN）​​：对​**​单个样本的所有特征维度​​计算统计量**，与批次无关。无论批次大小如何，LN始终能稳定归一化，更适合Transformer中变长序列和动态批次的场景。
>    3. 大模型训练时，多机多卡情况下，BN还有通信消耗。
> 2. Transformer处理的是​**​​序列数据​**​​（如文本中的单词），其自注意力机制使得​**每个位置的输出依赖于所有其他位置​**。此时：
>    1. ​BN的缺陷​​：若对整个批次的不同位置计算统计量，​**不同样本间的依赖关系可能引入噪声，破坏局部模式​**。
>    2. ​LN的优势​​：对同一序列内的所有位置独立归一化，​**保留序列内部的一致性​**，避免跨样本的信息干扰。
> 3. 训练与推理的一致性​​
>    1. ​​​BN在推理阶段​需要维护全局的移动平均统计量，而​​**​训练阶段的批次统计量可能与推理阶段分布不同​**​​（尤其在小批次或在线学习时），导致不一致。
>    2. ​LN无此问题​​：归一化仅依赖当前样本的特征，训练与推理行为完全一致，简化了部署流程。
> 4. 位置编码的兼容性​​
>    1. Transformer依赖位置编码（Positional Encoding）注入序列顺序信息。若使用BN，不同位置的统计量可能被混合，削弱位置信息的作用；而​**​LN仅在单个序列内操作，保留了位置编码的独立性​**​。
> 
> | 特性                | 层归一化（LN）                       | 批归一化（BN）                  |
> | ------------------- | ------------------------------------ | ------------------------------- |
> | ​统计量计算范围​​   | 单个样本的所有特征                   | 当前批次的所有样本的同一特征    |
> | ​依赖批次大小​​     | 否                                   | 是                              |
> | ​处理变长序列​​     | 更稳定                               | 需填充/掩码，可能引入噪声       |
> | ​训练与推理一致性​​ | 完全一致                             | 需维护移动平均，可能不一致      |
> | ​适用场景​​         | 序列模型（Transformer、RNN）、小批次 | 图像模型（CNN）、大批次稳定场景 |

</details>

<details>
<summary>multi-head self-attention（及transformer代码）</summary>

<br>

**self-attention**

Q：查询矩阵（理解：搜索栏中输入的查询内容）

$$
Q=XW^Q
$$

K：键矩阵（理解：数据库中与Q相关的一组关键字）

$$
K=XW^K
$$

V：值矩阵（理解：系统通过计算，展示最匹配K的所对应的内容V）

$$
V=XW^V
$$

总的公式：

$$
Attention(Q,K,V)=softmax(\frac {QK^T}{\sqrt{d_k}})V
$$

Attention 就是将想要查询的 Q 与数据库中的 K 进行比较，一对一地测量它们之间的相似度，并最终从最高相似度顺序向下依次并排列索引到的 V。所以，也可以理解 Attention 为一个数据库查表的过程。


拿出一组多头自注意力来解释流程：

![](../images/20211125/20211125_TRM_MSHA2.png)

1. 先计算 Q 与 K 的转置的点积。
2. 点积的结果就是生成注意力矩阵（**上图**）。
3. 然后用SoftMax进行归一化，这样每个字跟其他所有字的注意力权重的和为1。注意力矩阵的第一行就是第一个字c1与这六个字分别的相关程度（**这个理解很关键**）。
4. 接着用注意力矩阵给V加权，就可以找到最相关的值。

**multi-head**

> 多头注意力机制就是对同一个输入，使用**不同的** Q、K、V 权重**进行多组注意力计算**，得到多个结果后拼接起来，再通过**线性变换融合为最终输出**。

举个例子：

```bash
# 假设有一句话：I LOVE AI。
# 在输入 Transformer 之前，首先每个词（token）会被嵌入（embedding）成一个向量，比如（可见上图）：
# "I" → [0.2, -1.1, ..., 0.5]
# "love" → [1.3, 0.8, ..., -0.4]
# "AI" → [0.7, -0.9, ..., 1.2]
# 这个向量的长度就是 d_model，比如 512，那就是每个词用一个 512维的向量表示。就类似于CV中卷积完的“通道”维度。
# 假如 head 的个数为 8，那么就是每个头处理 64 个“通道”。
```


| 问题                     | 答案                                                                                                                                                                  |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 多头注意力计算量是否更大 | 是，确实增加了计算量，因为有h组QKV计算而不是一组<br>每一组都要做一次完整的 attention 运算<br>最后还要做一次拼接与线性映射                                             |
| 为什么计算量“增加但没炸” | 虽然用了多个 attention head，但**每个 head 的维度更小**，从而控制住了总计算量。<br>没有重复计算整份，每个 head 只负责**分工处理**一个低维空间，而不是全维度重复处理。 |
| 多头比单头效果更好       | 是，能捕捉多种语义关系，提升表达能力                                                                                                                                  |
| 多头效率低，难以训练     | 否，框架优化良好，都会对 multi-head attention 做高效并行化处理                                                                                                        |

<br>

**softmax**：

$$
softmax=\frac{e^{z_i}}{\sum_{j=1}^{n}{e^{z_j}}}
$$

**code**

```python
import torch
import torch.nn as nn
import math

# 1. Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        return attn @ V, attn

# 2. Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention()

    def forward(self, x, mask=None):
        B, L, D = x.size()
        Q = self.W_q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2) # transpose之后：(batch_size, num_heads, length, d_k)
        K = self.W_k(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2) # 同上
        V = self.W_v(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2) # 同上

        out, attn = self.attn(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.fc(out)

# 3. Position-wise Feedforward
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))

# 4. Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# 5. Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.attn(x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

# 6. Full Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

</details>


<details>
<summary>位置编码相关</summary>

<br>

**为什么transformer需要位置编码**

> - transformer本身不具备对序列中位置信息的天然捕捉能力，而位置信息对于理解和处理序列数据非常重要。
> - transformer在经过多头注意力之后，虽然保留了 token 顺序在输入排列中，但其核心注意力机制**完全不理解“第几个”**，它只看内容相关性（Query 和 Key 的匹配），所以**必须注入显式的位置信息**。

**为什么用正余弦函数做绝对位置编码**

> - 周期性 + 多频率，能表达多尺度的位置关系。使用不同频率的正余弦函数，能让模型同时看到：
>   - 粗粒度（低频）的位置变化（如 token 之间的长距离关系）
>   - 细粒度（高频）的位置变化（如短距离关系）
> - 可泛化到未见过的位置（外推能力）：
>   - 正余弦是连续且无限扩展的函数，不像词嵌入那样只能训练固定位置。
>   - 这使得模型能**泛化到比训练时更长的输入序列**（位置编码值是函数计算，不需要词表）
> - 内积保留相对位置信息：
>   - 两个位置编码的点积值随着相对距离变化是可预测的，方便模型感知相对位置。
> - 为什么不用别的函数：
>   - 正余弦函数具有良好的数学结构（傅里叶分析），能被神经网络有效学习、稳定训练。
>   - 用别的函数可能会导致：不可微、不平滑；不具备周期性和可推广性；不支持推理时生成新位置的编码

**为什么使用相对位置编码（绝对位置编码对长文本建模能力不足）**

> - 位置不再具有区分度：
>   - 在较长的文本中，远距离位置的正余弦编码值**趋于平滑或重叠**。
>   - 比如输入是 10000 长的序列，对于位置 500 和位置 502，它们编码差异很小。很多位置编码会“模糊在一起”，模型难以识别远距离结构。
> - 绝对位置编码只是告诉模型 token 处在第几位，并不告诉模型“我距离你多远”。在长文本中，这种**缺乏相对偏差的信息**，使得模型难以准确处理长距离依赖。
> - 不能跨上下文对齐（不支持滑动窗口）：
>   - 在长文本切分成段处理时，**绝对位置不具有平移不变性**，无法对齐 token 的上下文。

**ROPE 为什么能表示位置信息？旋转 QK 向量和“位置”有什么关系？**

> **ROPE 的核心思想**是：用二维向量的“旋转角度”来编码 token 所处的位置，并且这种角度变化能够影响 attention 的结果。
> 
> 怎么理解：
> 
> - 假设每个向量都是二维平面上的点，如 (x,y)
> - 给位置 1 的 token 转一个角度α，位置 2 的 token 转角度β。
> - 计算 QK 的点积时，这个旋转角度会影响它们的相关性（因为旋转后的向量方向不同）。
> 
> 数学一点：
> 
> 将 Q 和 K 的每一对维度当成一个二维向量，乘以一个旋转矩阵：
> 
> $$
> R(\theta)=
> \begin{bmatrix}
> cos\theta & -sin\theta \\
> sin\theta & cos\theta
> \end{bmatrix}
> $$
> 
> 这样，每个位置的向量就像顺时针旋转一定角度，而这个角度是基于其位置 `p` 和频率 `f` 设定的。
> 结果，注意力中的 Query 和 Key 相乘时，就带上了相对位置信息。
> 模型可以从这种“旋转差异”中学习相对距离，而不是像绝对编码那样只看“你在第几位”。

</details>


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