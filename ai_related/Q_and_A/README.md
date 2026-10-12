
问题跟答案按主题整理，以下目录链接到各专题文档。

## 计算机基础

[计算机基础](./conputer_basic.md)，包括以下问题：

```bash
1. 进程、线程、协程
```

## 深度学习基础

[深度学习基础](./deeplearning_basic.md)，包括以下几个问题：

```bash
1. 查全率、查准率、PR曲线、ROC曲线
2. 损失函数
3. 激活函数
4. 反向传播与梯度下降
5. 梯度下降与优化器
6. 正则化
```

## 强化学习相关

[强化学习相关](./rl_related.md)，包括以下几个问题：

```bash
1. 强化学习的两种优化方法
2. on-policy与off-policy
3. PPO (Proximal Policy Optimization, 近端策略优化)
4. loss不断下降，但reward一直震荡，是什么原因
```

## 大模型相关

[大模型相关](./llm_related.md)，包括以下几个问题：

```bash
1. DPO (Direct Preference Optimization, 直接偏好优化)
2. TRPO (Trust Region Policy Optimization, 信赖域策略优化)
3. GRPO (Group Relative Policy Optimization)
4. 大模型幻觉
5. transformer 中为什么使用 layer normalization 而不是用 batch normalization
6. multi-head self-attention（及transformer代码）
7. 位置编码相关
8. transformer在softmax中为什么要除以根号d_k
9. LoRA (Low-Rank Adaptation of Large Language Models)
10. 模型在训练和推理时的显存占用如何估算？
11. Seq2Seq模型的核心组件是什么？Encoder-Decoder结构如何解决长程依赖问题？输入序列过长，如何解决计算量问题？
12. MoE（专家混合模型，Mixture of Experts）
13. KV Cache
```

## RAG相关

[RAG相关](./rag_related.md)，包括以下几个问题：

```bash
1. RAG（检索增强生成，Retrieval-Augmented Generation）技术
2. RAG 过程中的 Query 改写技术
```

## Agent相关

[Agent相关](./agent_related.md)，包含以下几个问题：

### Agent 与 harness 的基本概念

```text
1. 一个最小可用的 Agent loop 应该包含哪些步骤？什么时候应该停止？
2. 固定 Workflow 和自主 Agent 各适合什么场景？应该如何选择？
3. ReAct、先规划再执行、边执行边调整有什么区别？
4. 为什么同一个模型换一个 harness，任务成功率可能差很多？
5. 长时间运行的 Agent 如何记录进度、恢复中断，并确认任务真的完成？
```

### 主流 harness 的特点、优缺点与对比

```text
1. 主流 harness 各有什么设计特点？优缺点如何对比？
2. Codex 和 Claude Code 的 harness 有什么异同？
3. DeepSeek Harness 的插件化有什么特点？相比只接几个工具，好在哪里、代价是什么？
4. OpenHands 为什么强调状态与执行环境分离？这种设计有什么优缺点？
5. OpenCode 的多模型设计有什么优势？什么时候只换 API 地址就够，什么时候不够？
6. WorkBuddy 这类办公 Agent，与 Coding Agent 的 harness 有什么不同？
7. 面试中如何回答“你会怎么选择和评估一个 harness”？
```

### 上下文、记忆与 Skills

```text
1. Context Engineering 和 Prompt Engineering 有什么区别？
2. 为什么上下文窗口足够大，Agent 仍然会遗漏信息或偏离任务？
3. 上下文裁剪、摘要压缩、保存到文件、按需检索分别适合什么场景？
4. 上下文压缩时必须保留哪些信息？如何避免丢失用户约束和任务进度？
5. Session 触发压缩后，下一次发给 LLM 的内容有哪些？哪些信息需要重新加载？
6. 对话历史、工作记忆、长期记忆有什么区别？什么信息值得长期保存？
7. Skill 的按需加载是怎么实现的？谁决定加载，加载后如何进入上下文？
8. Agentic RAG 和普通 RAG 有什么区别？Agent 如何决定是否检索、继续检索还是停止？
```

### Agent 基础、工具与评估

```bash
1. Agent关键技术与流程
2. Agent中的tool use
3. Agent/LLM 输出质量评估方法
```
