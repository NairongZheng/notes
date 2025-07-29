
## 大模型相关

[大模型微调](../LLM/大模型微调.md)

[RLHF](../RL/RLHF.md)


<details>
<summary>DPO (Direct Preference Optimization, 直接偏好优化)</summary>

<br>

**背景回顾**

> 传统 RLHF 三步骤：
> - SFT（Supervised Fine-Tuning）：用高质量数据监督微调基础模型。
> - RM（Reward Model）训练：基于人类选择 A 优于 B 的对比数据训练奖励模型。
> - PPO 强化学习：用奖励模型优化语言模型，强化“人类喜欢的输出”。
> 
> 存在问题：
> - RM + PPO 很复杂，训练难调
> - PPO 不稳定，样本效率低
> - 架构复杂，训练耗时

**DPO 的工作流程**

> **核心思想**：绕过奖励模型，直接优化原模型使其符合人类偏好
> 
> 它是RLHF的一种替代方案，但不需要强化学习，不用什么奖励模型，**只用监督学习方式**就能搞定。
> 
> **第一步：准备数据（成对偏好数据）**
> 
> - Prompt: “介绍一下猫和狗的区别。”
> - 回答 A（chosen）：猫和狗都是常见的宠物，猫通常独立，狗更黏人。
> - 回答 B（rejected）：狗是动物，猫不是狗。
> - 很明显，人更喜欢回答 A
> 
> 那么就记录一个三元组：`[prompt, chosen_response=A, rejected_response=B]`
> 
> **第二步：准备两个模型**
> 
> - 当前模型 $\pi$（要训练的模型）
> - 参考模型 $\pi_{ref}$（固定的，不训练，用来当参照物）
> 
> **第三步：给两个回答打分（模型算概率）**
> 
> 得到了 4 个数字：
> 
> ```bash
> log π(chosen | prompt)
> log π(rejected | prompt)
> log π_ref(chosen | prompt)
> log π_ref(rejected | prompt)
> ```
> 
> **第四步：计算“谁更好”的损失函数**
> 
> $$
> L=-\log \sigma (\beta · [\log \pi (y^+) - \log \pi (y^-) - (\log \pi_{ref} (y^+) - \log \pi_{ref} (y^-))])
> $$

**DPO 损失函数详解**

> **第一步：比较两个答案的概率差**
> 
> 用 log-prob（对数概率）来表示模型对两个回答的倾向：
> 
> $$
> \Delta_{\pi}=\log \pi (y^+) - \log \pi (y^-)
> $$
> 
> 如果这个值越大，说明模型越偏向好回答。
> 
> **第二步：减去参考模型的偏好差**
> 
> 如果我们有一个参考模型（比如 SFT 模型 $\pi_{ref}$），我们可以只优化“比参考模型更好的那部分”：
> 
> $$
> \Delta_{diff}=\Delta_{\pi} - \Delta_{\pi_{ref}}
> $$
> 
> **第三步：放进 sigmoid + log 中变成分类损失**
> 
> $$
> L=-\log \sigma (\beta · (\Delta_{\pi} - \Delta_{\pi_{ref}}))
> $$
> 
> 其中：
> - $\sigma(z) = \frac{1}{1 + e^{-z}}$是 sigmoid 函数；
> - $\beta$ 是一个温度参数，控制 sharpness（一般取 1.0）。
> 
> 所以最后就是：
> 
> $$
> L=-\log \sigma (\beta · [\log \pi (y^+) - \log \pi (y^-) - (\log \pi_{ref} (y^+) - \log \pi_{ref} (y^-))])
> $$
> 
> 也可以用softmax的形式表示：
> 
> $$
> L=-\log (\frac{e^{\beta (\log \pi (y^+) - \log \pi_{ref} (y^+))}}{e^{\beta (\log \pi (y^+) - \log \pi_{ref} (y^+))} + e^{\beta (\log \pi (y^-) - \log \pi_{ref} (y^-))}})
> $$

**DPO 相对于 RLHF 的优点总结**

> 1. 无需奖励模型（Reward Model）
>    1. RLHF：需要先训练一个奖励模型（Reward Model, RM）来估计人类偏好，然后再通过 PPO 等算法微调策略模型。
>    2. DPO：直接使用人类偏好数据进行优化，不需要显式训练奖励模型，省去一个步骤，减少误差传播。
> 2. 更稳定、更简单的训练过程
>    1. RLHF（如 PPO）：是基于强化学习的复杂优化过程，需要价值函数估计、advantage 计算、clip 等技巧，训练过程不稳定，调参困难。
>    2. DPO：是一个纯监督学习形式（logistic loss），没有 RL 的不稳定性，更容易训练和复现。
> 3. 无策略偏移（Policy Misalignment）问题
>    1. PPO 优化的是 reward，而不是人类真实偏好，有可能导致策略漂移。
>    2. DPO 明确建模偏好概率分布，优化的目标是让模型产生更偏好的人类答案，目标更贴近实际偏好数据。
> 4. 更强的可解释性
>    1. DPO 的 loss 是一个有明确意义的对数偏好概率（log-sigmoid），结果更易解释；
>    2. RLHF 的 reward 是间接学习到的，缺乏可解释性。

</details>


<details>
<summary>TRPO (Trust Region Policy Optimization, 信赖域策略优化)</summary>
<br>

**背景：为什么要 TRPO？**

> 在传统的策略梯度方法（如 REINFORCE、Vanilla Policy Gradient）中，策略更新步长过大时，容易导致策略发生剧烈变化，训练过程不稳定，甚至性能退化。
>
> - 问题1：策略更新“跳太远”
>   - 策略是概率分布，参数稍微变化就可能导致行为完全不同。
>   - 大步长更新可能让策略“崩掉”。
> - 问题2：目标函数无约束
>   - 传统方法直接最大化期望回报，没有限制策略变化幅度。
> - 问题3：训练不稳定
>   - 策略更新过大，导致采样分布和目标分布差异过大，梯度估计不准确，训练容易发散。

**TRPO 的核心思想**

> TRPO 的核心思想：每次更新策略时，限制新旧策略之间的“距离”不超过一个信赖域（trust region），保证策略变化平稳，训练更稳定。
>
> - 用 KL 散度（Kullback-Leibler Divergence）度量新旧策略的差异。
> - 通过约束 KL 散度，确保每次策略更新不会“跳太远”。
> - 本质上是“在保证安全的范围内，尽可能提升策略性能”。

**TRPO 的算法流程**

> 1. 采样：用当前策略$\pi_{old}$与环境交互，收集一批轨迹（state, action, reward）。
> 2. 估计优势函数：用 GAE（Generalized Advantage Estimation）等方法估算每个动作的优势$A_t$。
> 3. 构建目标函数：最大化新旧策略概率比加权的优势期望。
> 4. 信赖域约束：约束新旧策略的平均 KL 散度不超过$\delta$（如 0.01）。
> 5. 求解优化问题：用二阶优化（如共轭梯度法）近似求解带约束的最大化问题，得到新的策略参数。
> 6. 更新策略：用新参数替换旧策略，进入下一轮。

**TRPO 的目标函数**

> TRPO 的优化目标是：
>
> $$
> \max_{\theta} \ \mathbb{E}{s,a \sim \pi{\text{old}}} \left[ \frac{\pi_{\theta}(a|s)}{\pi_{\text{old}}(a|s)} A^{\pi_{\text{old}}}(s,a) \right]
> $$
>
> 同时约束新旧策略的平均 KL 散度不超过阈值$\delta$：
>
> $$
> \text{subject to} \ \mathbb{E}{s \sim \pi{\text{old}}} \left[ D_{KL}(\pi_{\text{old}}(\cdot|s) \| \pi_{\theta}(\cdot|s)) \right] \leq \delta
> $$
>
> - 其中 $A^{\pi_{\text{old}}}(s,a)$ 是优势函数。
> - $D_{KL}$ 是 KL 散度，衡量新旧策略的“距离”。
> - δ 是一个超参数，控制每次更新的最大幅度。

**TRPO 的优缺点总结**

> | 优点 | 说明 |
> |------------------------------|--------------------------------------------------------------|
> | 更新稳定 | 通过 KL 散度约束，防止策略剧烈变化，训练过程更平滑 |
> | 理论收敛性好 | 有严格的理论保证，更新不会导致性能下降 |
> | 支持大步长 | 可以安全地使用较大的步长，提升优化效率 |
> | 适合高维、复杂策略 | 在大规模神经网络策略中表现良好 |
>
> | 缺点 | 说明 |
> |------------------------------|--------------------------------------------------------------|
> | 实现复杂，计算量大 | 需要二阶优化（如共轭梯度法、Hessian 向量积），实现较复杂 |
> | 计算资源消耗高 | 每次更新涉及大量矩阵运算，训练速度慢于 PPO |
> | 不易扩展到大规模分布式场景 | 二阶优化和全局 KL 约束难以高效并行化 |
> | 实践中常被 PPO 替代 | PPO 用剪切近似代替 KL 约束，效果相近但实现更简单，效率更高 |

**TRPO 与 PPO 的对比**

> - TRPO：严格的 KL 散度约束，二阶优化，理论保证强，计算复杂。
> - PPO：PPO中的 KL 散度只是加在损失函数后的类似于“正则项”的操作，并不是严格约束，而是“软KL惩罚”！

其中KL散度相关的部分可以查看[概率论部分](../basic/probability_theory.md#kl散度kullback-leibler-divergence)

</details>


<details>

<summary>GRPO (Group Relative Policy Optimization)</summary>

<br>

**从PPO到GRPO**

> PPO通过引入**裁剪（Clipping）**和**KL散度约束**来限制策略更新的幅度，从而在保证训练稳定性的同时，尽可能大地利用样本。
> 
> PPO的核心思想是：在每次迭代中，新策略与旧策略之间的差异不能太大，以避免策略的剧烈波动导致训练不稳定。
> 
> PPO的成功在于其在稳定性、样本效率和实现复杂度之间取得了良好的平衡，使其成为目前应用最广泛的强化学习算法之一。
> 
> 然而，PPO以及其他许多策略梯度算法，通常依赖于一个**价值网络**（Value Network）来估计状态的价值（或优势函数）。**价值网络的作用是为策略网络的更新提供一个基准，帮助策略网络判断当前动作的好坏**。虽然价值网络在一定程度上降低了策略梯度的方差，但它也带来了新的问题：
> 
> 1.  **计算开销和内存占用：** 价值网络本身是一个神经网络，其训练需要额外的计算资源和内存。在大规模模型（如大型语言模型）的训练中，这会成为一个显著的瓶颈。
> 2.  **价值估计的准确性：** 价值网络的训练也可能不稳定，其估计的准确性直接影响策略更新的质量。如果价值估计不准确，可能会导致策略更新的方向错误，甚至使训练过程发散。
> 3.  **超参数调优：** 价值网络的训练引入了额外的超参数，增加了算法的复杂性和调优难度。
> 
> 这些挑战促使研究者们探索新的策略优化方法，旨在在保持训练稳定性的同时，减少对价值网络的依赖，提高计算效率和样本效率。正是在这样的背景下，**组相对策略优化**（Group Relative Policy Optimization, GRPO）应运而生。

**GRPO核心思想与流程**

> GRPO的核心思想是：**通过在“组”内比较不同动作的相对奖励，直接估计优势函数，从而完全摒弃对价值网络的依赖。**
> 
> 这种方法更加直接和高效，尤其适用于那些生成式任务，例如大型语言模型（LLMs）的微调，因为LLMs通常会生成多个候选序列。
> 
> GRPO的算法流程可以概括为以下几个步骤：
> 
> 1. **数据收集（Rollout）：** 使用当前策略（旧策略）在环境中进行采样，收集一系列轨迹（状态-动作-奖励序列）。
> 2. **组内采样与生成：** 对于每个状态，不只生成一个动作，而是生成一组（K个）候选动作序列。
> 3. **奖励计算与归一化：** 对这K个候选动作序列分别计算其累积奖励。然后，对这些奖励进行归一化处理，得到相对优势。
> 4. **策略更新：** 利用计算出的相对优势，结合KL散度约束，更新策略网络。

**GRPO核心技术：分组相对优势**

> GRPO的数据结构与传统强化学习有所不同。它不再是简单的`(state, action, reward)`三元组，而是针对每个状态，拥有一组`(动作序列_i, 奖励_i)`的对。（下面用LLM来举例）
> 
> **1. 分组机制（Grouping）**
> 
> 对同一提示（prompt）生成K个响应构成一个组：
> 
> $$
> G=((y_1,r_1),(y_2,r_2),...,(y_K,r_K))
> $$
> 
> 其中，$r_i=R(y_i|x)$为响应$y_i$的奖励值
> 
> **2. 相对优势计算**
> 
> 组内标准化优势函数：
> 
> $$
> \tilde{A_i}=\frac{r_i-\mu_G}{\sigma_G}
> $$
> 
> 其中：
> - $\mu_G=\frac{1}{K}\sum_{j=i}^K r_j$为组内平均奖励
> - $\sigma_G=\sqrt{\frac{1}{K}\sum_{j=1}^K (r_j-\mu_G)^2}$为组内标准差
> 
> 相对排名优势：
> 
> $$
> A_i^{rank}=\frac{rank(r_i)-(K+1)/2}{K/2}
> $$
> 
> 其中，$rank(r_i)$为$r_i$在组内的排名（1到K）
> 
> **3. 混合优势函数**
> 
> 最终优势函数为标准化优势与排名优势的加权和：
> 
> $$
> A_i^{GRPO}=\lambda \tilde{A_i}+(1-\lambda)A_i^{rank}
> $$
> 
> （实验表明$\lambda=0.7$效果最佳）

**GRPO目标函数设计**

> GRPO的损失函数与PPO类似，也包含一个**策略比率（Policy Ratio）**和**KL散度约束**。
> 
> 策略比率 $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$。
> 
> GRPO的损失函数可以表示为：
> 
> $L(\theta) = \mathbb{E}_{s, a \sim \pi_{\theta_{old}}} \left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) - \beta \text{KL}(\pi_{\theta_{old}}(\cdot|s_t), \pi_{\theta}(\cdot|s_t)) \right]$
> 
> 其中：
> - $A_t$ 是通过组内采样计算得到的优势函数。
> - $\epsilon$ 是裁剪参数，用于限制策略更新的幅度。
> - $\beta$ 是KL散度项的权重，用于控制新旧策略之间的距离。
> 
> 这个损失函数的目标是最大化相对优势，同时通过裁剪和KL散度约束来确保策略更新的稳定性。

**GRPO创新点与解决的问题**

> 1.  **摒弃价值网络：**
>     *   **创新点：** 这是GRPO最显著的创新。它不再需要训练一个独立的价值网络来估计状态价值或优势函数。
>     *   **解决的问题：**
>         *   **计算效率：** 大幅减少了训练所需的计算资源和内存占用，使得GRPO在大规模模型训练中更具可行性。
>         *   **训练稳定性：** 消除了价值网络训练可能带来的不稳定性和误差传播，简化了算法的整体训练流程。
>         *   **超参数简化：** 减少了需要调优的超参数数量，降低了算法的复杂性。
> 
> 2.  **组内相对优势估计：**
>     *   **创新点：** 通过在同一状态下采样多个动作序列，并计算它们之间的相对奖励来直接估计优势。
>     *   **解决的问题：**
>         *   **优势估计的直接性：** 避免了通过复杂的函数逼近来估计优势，使得优势估计更加直接和鲁棒。
>         *   **样本效率：** 在一定程度上提高了样本效率，因为每次策略更新都利用了同一状态下的多个样本信息。
> 
> 3.  **适用于生成式任务：**
>     *   **创新点：** GRPO的“组”概念与生成式任务（如文本生成）中模型可以生成多个候选输出的特性天然契合。
>     *   **解决的问题：**
>         *   **LLMs微调：** 为大型语言模型（LLMs）的强化学习微调提供了一种高效且稳定的方法。在LLMs中，模型可以生成多个响应，GRPO可以利用这些响应的相对质量来指导模型学习。
> 
> 4.  **KL散度约束的保留：**
>     *   **创新点：** 虽然摒弃了价值网络，但GRPO依然保留了PPO中有效的KL散度约束。
>     *   **解决的问题：**
>         *   **策略稳定性：** 确保了策略更新的幅度得到有效控制，防止策略发生剧烈变化，从而保持训练的稳定性。
>         *   **收敛性：** 有助于算法的收敛，避免策略在训练过程中发散。

**GRPO的优势总结**

> *   **高效性：** 无需价值网络，显著降低计算和内存开销。
> *   **稳定性：** 通过组内相对优势和KL散度约束，确保策略更新的稳定。
> *   **样本效率：** 利用组内采样，更有效地利用数据。
> *   **适用性广：** 特别适用于生成式任务，如大型语言模型的微调。

</details>


<details>
<summary>大模型幻觉</summary>

<br>

**大模型幻觉的定义**

> 大模型生成的内容在语法上合理、语言上流畅，但在事实层面是错误的、不存在的、虚构的。
> 比如：
> 
> - 编造一个不存在的学术引用或论文标题；
> - 虚构一个 API 函数或参数；
> - 错误地归因某个概念；
> - 给出不存在的历史事件。
> 
> 这些看似“有根有据”的内容，其实完全是语言生成模型自我联想出来的产物。

**大模型产生幻觉的根本原因**

> 1. 预测下一个词，而非理解世界
>    1. 大语言模型的核心训练目标是：**最大化下一个词的预测概率，而不是最大化“事实正确性”**。
>    2. 这意味着：模型学到的是“给定上下文，什么词更可能出现”，不是“什么词真实存在”。
>    3. 所以它会倾向于生成“语言上合理”的内容，而不是“客观上正确”的内容。
> 2. 没有访问真实世界的机制
>    1. 语言模型在推理时**并不访问知识库或数据库**，它所有的**信息来自预训练语料内部的统计关系**。
>    2. 如果在训练数据中看到“爱因斯坦是物理学家”很多次，它会记住这个模式。
>    3. 但如果被问“爱因斯坦出生在哪个城市”，训练数据没有明确出现，它就会“推测”一个听起来合理的答案，比如“柏林”或“法兰克福”，即使是错误的。
> 3. 缺乏事实验证机制
>     1. 人类说话时会主动校验知识的真伪（或查询资料），而语言模型并不会“怀疑自己”，不知道“自己不知道”，也**没有机制去查证信息**。这就导致它有时“自信地胡说八道”。
> 4. 语言模型的泛化能力 ≠ 事实归纳能力
>    1. 语言模型具有很强的“模式泛化”能力。
>    2. 但它泛化的结果可能在形式上合理、在内容上却是编造的。
>    3. 这种幻觉本质上来自于它**在语言空间中“走捷径”模拟真实语境，但忽略了事实基础**。
> 5. 训练数据本身可能含有错误或矛盾
>    1. 模型的知识来源（如互联网语料）中可能包含伪科学、虚假信息、语义歧义、讽刺或误传。
>    2. 模型并不会区分“真假”，而是学习“出现频率高、上下文自然”的内容。
>    3. 这也会进一步导致幻觉。
> 
> | 根本原因     | 说明                                         |
> | ------------ | -------------------------------------------- |
> | 目标错位     | 模型训练目标是“语言流畅性”，不是“事实正确性” |
> | 缺乏世界模型 | 模型没有知识图谱或物理世界的真实建模         |
> | 无检索能力   | 推理时不能动态查询真实信息，靠“记忆”瞎猜     |
> | 不具备“意识” | 不知道自己何时知道或不知道（缺乏元认知能力） |
> | 数据噪声     | 训练数据本身就可能包含错误和模糊信息         |

**如何缓解幻觉**

> 虽然幻觉无法彻底消除，但以下技术可以显著缓解：
> 
> - RAG（检索增强生成）：先查资料再回答
> - 指令微调（SFT）：用高质量、指令数据微调模型
> - 知识注入（KNN-LM、LoRA + 专业知识）
> - 校验模块：在后处理时进行事实校验与剔除
> - 限制生成范围：如有限选项、多轮对话确认
> - 训练阶段优化：使用 RLHF（人类反馈强化学习）减少幻觉


</details>

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

![](../../images/20211125/20211125_TRM_MSHA2.png)

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
<summary>transformer在softmax中为什么要除以根号d_k</summary>

<br>

**注意力回顾**

> 在Transformer的自注意力（Self-Attention）机制中，计算注意力分数时有如下公式：
> 
> $$
> \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
> $$
> 
> 其中，$Q$（Query）、$K$（Key）、$V$（Value）是通过输入和参数矩阵线性变换得到的，$d_k$是Key向量的维度。

**如果不除，会发生什么？**

> 设 $Q$ 和 $K$ 是随机初始化的，元素是独立的零均值高斯变量，维度为 $d_k$，则它们的点积 $Q \cdot K$ 的期望和方差如下：
> 期望：$E[Q \cdot K] = 0$
> 方差：$\text{Var}[Q \cdot K] = d_k$
> 也就是说：维度越高，点积值就越大（方差正比于 $d_k$）。
> 结果是：
> - 点积过大 -> softmax 输出变成 one-hot
> - 梯度传播变得不稳定（几乎只对一个位置有梯度）
> - 模型训练困难，学习缓慢甚至不收敛

**为什么除以 $\sqrt{d_k}$ 有用？**

> 这是为了把 $Q \cdot K^T$ 的值“标准化”到一个比较稳定的范围。
> 如果点积期望方差为 $d_k$，那么除以 $\sqrt{d_k}$ 后：
> - 方差变成 $\text{Var}[Q \cdot K] / d_k = 1$（随机变量 $X$ 除以常数 $c$，其 方差变化为：$\text{Var}(X / c) = \text{Var}(X) / c^2$）
> - 保证 softmax 输入值的尺度稳定在一个合理区间（比如 -4 到 +4）
> - softmax 输出更平滑，不至于极端化，利于训练

**举个极端例子（数值说明）**

> 假设没有除以 $\sqrt{d_k}$，某一对 $QK$ 点积值为 50，而其他值为 1：
> - $softmax([50, 1, 1, 1]) ≈ [~1, 0, 0, 0]$ -> 几乎 one-hot
> - 梯度集中在一个位置，模型变得不稳定
> 
> 而如果我们除以 $\sqrt{d_k}$（比如 $\sqrt{64} = 8$），点积值变成 6.25，softmax 输出就会更加平滑。

</details>


<details>
<summary>LoRA (Low-Rank Adaptation of Large Language Models)</summary>

<br>

在[大模型微调](../LLM/大模型微调.md)里面有详细介绍。

</details>

<details>
<summary>模型在训练和推理时的显存占用如何估算？</summary>

<br>

**显存消耗的组成部分**

> | 类型                                  | 说明                                                 | 是否训练阶段特有     |
> | ------------------------------------- | ---------------------------------------------------- | -------------------- |
> | 模型参数                              | 存储模型权重本身（如 Linear 层的权重矩阵）           | 否                   |
> | 参数梯度                              | 每个参数的梯度，训练时需要                           | ✅ 是                 |
> | 优化器状态                            | Adam 优化器需要记录每个参数的动量等状态              | ✅ 是                 |
> | 激活值（中间特征）                    | Transformer 每一层前向传播结果，训练时为反向传播保留 | ✅ 是（推理保留极少） |
> | 临时缓存（临时变量、LayerNorm缓存等） | 推理和训练都需要                                     | 否                   |
> | 显存碎片/其他系统开销                 | CUDA 系统、库分配的一些空间                          | 否                   |

**显存与模型参数量的关系**

> 以 `7B` 模型，参数精度 `fp16` 为例：
> 
> $$
> 显存 = 参数数量 \times 精度(bytes) = 7 \times 10^9 \times 2 bytes = 14 GB（模型权重）
> $$
> 
> **如果是训练，还需要加上**：
> 
> - 参数梯度（同样大小）：14 GB
> - Adam 优化器状态（通常是 2 × 参数大小）：28 GB（1 倍 动量m（momentum）内存，1 倍 动量v（RMS）内存）
> 
> **训练总共：~56 GB（不含激活）**

**显存与批次大小（Batch Size）的关系**

> 批次大小对显存的影响：
> 
> - 激活值（Activation）缓存增加：
>   - 每个输入样本在经过模型的每一层时都会产生中间结果（激活值），训练时这些> 激活值需要缓存，用于反向传播。
>   - 如果 batch size 是 1，就缓存 1 个样本的激活值；
>   - 如果 batch size 是 64，就缓存 64 个样本的激活值；
>   - 所以激活值缓存随 batch size **线性增长**。
> - 注意力矩阵缓存增加：
>   - 注意力权重矩阵的维度是 `[B,n_heads,seq_len,seq_len]`
>   - 所以显存中要存储的注意力矩阵量也会随着 batch size 增加而**线性增长**。
> 
> 因此显存占用大致与 batch size 成线性关系：
> 
> $$
> 显存 \propto BatchSize
> $$

**显存与序列长度（seq_len）的关系**

> 序列长度影响：
> 
> - 激活缓存量（如上所示）
> - 注意力矩阵大小：`[batch_size, n_heads, seq_len, seq_len]`
> 
> 因此注意力的显存增长为**平方级别**：
> 
> $$
> 显存 \propto {SeqLen}^2
> $$

</details>


<details>
<summary>Seq2Seq模型的核心组件是什么？Encoder-Decoder结构如何解决长程依赖问题？输入序列过长，如何解决计算量问题？</summary>

<br>

**Seq2Seq 模型的核心组件**

> Seq2Seq（Sequence-to-Sequence）是一类**将一个序列映射到另一个序列**的模型架构，广泛应用于：
> 
> - 机器翻译（如英文到法文）
> - 文本摘要
> - 问答系统
> - 语音识别等
> 
> 核心组件：
> 
> 1. Encoder：负责接收输入序列，并将其编码为一个向量（或一组向量）表示输入的语义信息
> 2. Decoder：接收 Encoder 的输出，并逐步生成目标序列（一个 token 一个 token 地预测）
> 3. Attention（可选）：提高模型对长序列的处理能力，允许 Decoder 在每一步关注输入序列的不同部分

**Encoder-Decoder 是如何解决长程依赖问题的？**

> 最初的 Seq2Seq 模型（无 Attention）的问题：
> 
> - Encoder 把整个输入序列**压缩成一个固定大小的向量**（称为上下文向量 context），再传给 Decoder。
> - 如果输入序列很长，固定向量无法承载全部信息 → **信息丢失，长程依赖难以建模**。
> 
> 引入 Attention 机制后的改进：
> 
> - Attention 机制让 Decoder 在生成每个词时动态地关注 Encoder 的输出中的不同位置。
> - 不再依赖一个固定向量，而是每个时间步都能参考整个输入序列。
> 
> 现代 Transformer 架构下的 Seq2Seq：
> 
> - 完全抛弃 RNN，用**自注意力（Self-Attention）+ 多头注意力**实现 Encoder 和 Decoder。
> - 每个位置可以直接访问序列中所有其他位置，**天然支持长程依赖**。
>
> **总结一句话**：Encoder-Decoder 架构通过引入 Attention 机制，让 Decoder 在生成序列时可以灵活关注输入序列的不同部分，从而有效解决长程依赖问题。

**如果输入序列很长，注意力矩阵的计算量和显存占用会迅速膨胀，如何解决计算量问题？**

> 注意力矩阵的维度是：`[batch_size,n_heads,seq_len,seq_len]`
> 计算复杂度：$O(SeqLen^2)$
> 
> 解决方法（**减少计算量/减少精度两种方式**）：
> 
> 1. **稀疏注意力**：复杂度从 $O(n^2)$ 降低到 $O(n \cdot \sqrt{n})$ 或 $O(n \cdot \log n)$
>    1. Longformer：局部窗口 + 全局 token 关注机制
>    2. BigBird：局部 + 稀疏跳跃 + 全局 token，理论上具备 Transformer 表达能力
>    3. Sparse Transformer：使用规则设计的稀疏注意力模式
>    4. Reformer：使用 LSH（局部敏感哈希）减少注意力计算
> 2. **线性注意力**：复杂度降为 $O(n)$，但可能会损失精度
>    1. Performer：利用核函数重写注意力为线性形式
>    2. Linformer：假设注意力矩阵是低秩的，对 K/V 做降维
>    3. Linear Transformer：修改注意力定义为线性形式
> 3. **分块输入（Chunking）或滑动窗口**：
>    1. 把长序列拆成多个短块，分别计算注意力，再用跨块机制（如 sliding window）进行上下文传播。
> 4. **使用低精度**：
>    1. 虽然不减少计算复杂度，但可以降低显存占用，让长序列训练更现实。

</details>


<details>
<summary>MoE（专家混合模型，Mixture of Experts）</summary>

**基本原理与优势**

> MoE（专家混合模型，Mixture of Experts）是一种深度学习模型结构，旨在通过引入多个“专家”子网络和一个“门控”机制，实现模型参数的高效利用和推理加速。MoE在大规模模型（如NLP中的大语言模型）中尤为流行，因为它能在不显著增加推理成本的情况下提升模型容量。
> 
> **基本原理**
> 
> - **专家（Experts）**：每个专家通常是一个独立的神经网络（如MLP），专注于处理输入空间的某一部分。
> - **门控网络（Gate）**：门控网络根据输入内容，动态选择最合适的一个或几个专家进行激活和计算，**其余专家不参与本次前向传播**。
> - **稀疏激活**：每次只激活少量专家（如Top-1或Top-2），大大减少了实际计算量。
> 
> **主要优势**
> 
> 1. **参数高效扩展**：可以在不增加推理计算量的前提下，极大地增加模型参数量（如Switch Transformer、GLaM等）。
> 2. **推理加速**：由于每次只激活部分专家，推理速度远快于同等参数量的全连接大模型。
> 3. **模型多样性**：不同专家可以学习输入空间的不同特征，提高模型泛化能力。
> 
> **典型结构**
> 
> 以Transformer中的MoE层为例，通常在Feed Forward部分插入MoE结构：
> 
> 1. 输入经过门控网络，计算每个专家的权重（通常用softmax）。
> 2. 选择Top-k个专家（如Top-1或Top-2），只将输入分配给这几个专家。
> 3. 专家输出加权求和，作为MoE层的输出。


**MoE常见问题及解决方案**

> **1. 专家负载不均衡（Load Imbalance）**
> 
> **问题描述**：门控网络可能会偏向某些专家，导致部分专家过载，部分专家几乎不被激活，影响训练效率和模型效果。
> 
> **解决方法**：
> - **负载均衡损失（Load Balancing Loss）**：在总损失中加入负载均衡项，鼓励门控网络均匀分配输入到各个专家。
> - **噪声门控（Noisy Gating）**：在门控分数中加入噪声，增加探索性，防止陷入局部最优。
> - **专家容量限制（Capacity Constraint）**：限制每个专家每次最多接收多少输入，超出部分丢弃或分配到其他专家。
> 
> **2. 通信和并行效率问题**
> 
> **问题描述**：大规模MoE模型通常需要跨多卡/多机分布式训练，专家之间的数据交换和同步会带来通信瓶颈。
> 
> **解决方法**：
> - **专家并行（Expert Parallelism）**：将不同专家分布到不同设备上，利用分布式框架（如DeepSpeed、FairScale）优化通信。
> - **局部专家分组（Local Expert Grouping）**：将专家分组，每组只在本地设备内通信，减少全局通信量。
> - **稀疏调度优化**：只传递被激活的专家数据，减少无用通信。
> 
> **3. 门控网络训练不稳定**
> 
> **问题描述**：门控网络如果训练不充分，可能导致专家选择不合理，影响模型性能。
> 
> **解决方法**：
> - **门控网络预热（Warm-up）**：训练初期对门控网络参数进行预热，或采用更平滑的激活策略。
> - **正则化**：对门控输出加正则项，防止过度偏向某些专家。
> - **温度调整（Temperature Scaling）**：调整softmax温度，控制门控分布的平滑度。
> 
> **4. 专家漂移（Expert Drift）**
> 
> **问题描述**：部分专家长期不被激活，导致参数更新缓慢甚至“死亡”。
> 
> **解决方法**：
> - **周期性重置专家**：定期重置长期未被激活的专家参数。
> - **专家激活监控**：监控每个专家的激活频率，动态调整门控策略。

**代码实现**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

class Top1Gate(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x: [batch, d_model]
        logits = self.gate(x)  # [batch, num_experts]
        gate_scores = F.softmax(logits, dim=-1)
        top1_idx = torch.argmax(gate_scores, dim=-1)  # [batch]
        one_hot = F.one_hot(top1_idx, num_classes=gate_scores.size(-1)).float()
        mask = one_hot  # [batch, num_experts]
        return mask, gate_scores

class MoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.gate = Top1Gate(d_model, num_experts)

    def forward(self, x):
        # x: [batch, d_model]
        mask, gate_scores = self.gate(x)  # mask: [batch, num_experts]

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [batch, num_experts, d_model]

        # 只取 top1 的输出（可以修改成 top-k）
        mask = mask.unsqueeze(-1)  # [batch, num_experts, 1]
        out = torch.sum(mask * expert_outputs, dim=1)  # [batch, d_model]

        return out
```

</details>

