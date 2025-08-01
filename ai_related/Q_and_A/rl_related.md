
## 强化学习相关

强化学习相关的也可以去看看[强化学习](../RL/强化学习.md)、[白话强化学习笔记](../RL/白话强化学习笔记.md)或者[slaythespire_AI](../RL/slaythespire_AI.md)

<details>
<summary>强化学习的两种优化方法</summary>

<br>

**基于值函数（Value Function）的优化方法**

> **直观理解**
> 
> 想象一只老鼠，在迷宫中寻找奶酪。
> - 会给每个位置和行为一个“评分”（比如：在这个位置往上走可能会得到高分）。
> - 这些“评分”就是所谓的 Q值，也就是「在状态s下采取动作a能获得的预期总奖励」。
> - 不断尝试走不同的路线，观察哪条路最终会带你到奶酪（获得奖励最多），然后更新对各个选择的评分。
> 
> **优化方式**
> 
> - 学习一个函数$Q(s,a)$：在状态$s$下选择动作$a$的预期收益
> - 每次尝试后根据新的经验调整$Q$值，比如`Qlearning`：
> 
> $$
> Q(s,a){\leftarrow} Q(s,a)+\alpha[r+{\gamma} \mathop{\max}\limits_{a'} Q(s',a')-Q(s,a)]
> $$
> 
> - 目标是学到一个好用的$Q$函数，之后只要“贪心”地选$Q$值最大的动作即可。
> 
> **特点总结**
> 
> | 特性         | 描述                                          |
> | ------------ | --------------------------------------------- |
> | 优化对象     | 值函数（Value/Q-Function）                    |
> | 策略间接学习 | 通过值函数推导出最优策略                      |
> | 算法代表     | Q-Learning、SARSA、DQN（深度Q网络）           |
> | 适合场景     | 状态/动作空间不太大，或可用神经网络近似值函数 |

**基于策略（Policy）的优化方法**

> **直观理解**
> 
> 还是老鼠找奶酪的故事。
> - 这次老鼠不再记录每个动作的评分，而是直接“学习一种习惯”。
> - 比如：它学会了“见到岔路口，70% 向右走，30% 向上走”，这个就是它的“**策略**”。
> - 如果这套习惯让它经常找到奶酪，就加强这个习惯；否则就调整策略。
> 
> **优化方式**
> 
> - 直接学习一个策略函数$\pi (a|s;\theta)$，输出某状态下采取各个动作的概率。
> - 用“策略梯度”算法来优化参数$\theta$，让高奖励的动作概率变大。例如：
> 
> $$
> \nabla_\theta J(\theta)=E[\nabla_\theta\log {\pi_{\theta}(a|s)}·R]
> $$
> 
> - 调整参数方向，使得带来高奖励的动作更有可能被选中。
> 
> **特点总结**
> 
> | 特性         | 描述                                 |
> | ------------ | ------------------------------------ |
> | 优化对象     | 策略本身（Policy）                   |
> | 策略直接学习 | 不依赖值函数，直接优化决策策略       |
> | 算法代表     | Policy Gradient、REINFORCE、PPO、A2C |
> | 适合场景     | 连续动作空间、策略难以从值函数推导时 |

</details>


<details>
<summary>on-policy与off-policy</summary>

<br>

**on-policy**

> 想象你是一个打游戏的 AI 学徒：
> - 你自己操控角色，打一局后回看录像总结哪里做得不好，然后优化自己的操作方式。
> - 自己打的局，自己学；
> - 每一局用的是“你现在的策略”，从中学习也是为了改进这个策略；
> - 比如：SARSA、PPO、REINFORCE。
> 
> | 特性       | 描述                                                          |
> | ---------- | ------------------------------------------------------------- |
> | 策略一致性 | 收集数据用的策略 == 学习/更新的策略                           |
> | 样本利用率 | 一般只能用一次（不适合重放）                                  |
> | 探索方式   | 通常要有策略自带的随机性（比如 epsilon-greedy，或者策略分布） |
> | 优点       | 理论收敛性强，学得稳定                                        |
> | 缺点       | 样本效率低，不能重复利用数据                                  |

**off-policy**

> 还是一个打游戏的 AI 学徒：
> - 你在旁边看大神打游戏（或者过去自己老版本的录像），从他们的操作中提取经验并优化自己的策略。
> - 用的是别人的经验，训练的是自己的策略；
> - 比如 Q-learning 中你可以随便探索（比如随机行动），但训练时总是更新“最优策略”对应的 Q 值；
> - DQN、Q-Learning、DDPG 都是 off-policy 算法。
> 
> | 特性       | 描述                                           |
> | ---------- | ---------------------------------------------- |
> | 策略不一致 | 收集数据的策略 ≠ 学习优化的策略                |
> | 样本利用率 | 可以反复使用历史数据（比如经验回放）           |
> | 探索方式   | 数据可以来自随机策略或旧策略                   |
> | 优点       | 样本效率高，能用“离线数据”训练                 |
> | 缺点       | 学习不稳定，尤其当行为策略与目标策略差别太大时 |

**总结**

> | 对比维度         | On-Policy                  | Off-Policy                     |
> | ---------------- | -------------------------- | ------------------------------ |
> | 数据收集策略     | 当前正在优化的策略         | 可以是旧的、随机的或别人的策略 |
> | 数据使用效率     | 每条数据用一次             | 可以反复使用、经验回放         |
> | 是否需要探索机制 | 是，需要策略有随机性       | 不一定，行为策略可以另设       |
> | 学习的稳定性     | 稳定但效率低               | 效率高但容易发散               |
> | 应用例子         | PPO、REINFORCE、A2C、SARSA | DQN、Q-Learning、DDPG、SAC     |

</details>


<details>
<summary>PPO (Proximal Policy Optimization, 近端策略优化)</summary>

<br>

**背景：为什么要 PPO？**

> 在策略梯度方法（如 REINFORCE）中，直接优化期望回报：
> 
> $$
> J(\theta)=E_{\tau \sim \pi_{\theta}}[\sum_t\log\pi_{\theta}(a_t|s_t)·A_t]
> $$
> 
> 这些方法的基本思想是：通过采样和优化，使策略参数$\theta$让策略$\pi_{\theta}(a|s)$更倾向于带来高回报的动作。
> 
> 当我们对策略进行更新时，如果步长（learning rate）或梯度本身太大，就可能导致 策略突然发生巨大变化。具体原因解释如下：
> 
> **原因1：策略是概率分布，稍微调整参数就可能完全改变行为**
> 
> - 假设当前策略（旧），在某个状态 s： 动作 a1 的概率是 0.9，动作 a2 的概率是 0.1
> - 更新后策略（新）：a1 的概率变成 0.1，a2 的概率变成 0.9
> - 策略在同一个状态下的行为 翻转了，这就是剧烈变化。
> 
> **原因2：REINFORCE / A2C 的目标函数是无约束的**
> 
> - REINFORCE 要最大化：$E_{\tau \sim \pi_{\theta}}[\sum_t\log\pi_{\theta}(a_t|s_t)·A_t]$
> - 这个目标函数没有限制策略变化的幅度，所以在更新参数的时候，哪怕变化很大，也不会受到惩罚。
> 
> **原因3：剧烈变化导致不稳定训练**
> 
> - 如果策略变动太大，当前策略和生成经验的旧策略差距太大；
> - 会导致估计的梯度方向不准确；
> - 一步导致学习“崩掉”或震荡。

**PPO 的核心**

> **PPO 的核心思想**：不让策略每次更新跳太远，采用“剪切（Clipping）”技术做约束。
> 
> **PPO 的核心目标函数**：
> 
> $$
> L_{CLIP}(\theta)=E_t[\min(r_t(\theta)·A_t,clip(r_t(\theta),1-\epsilon,1+\epsilon)·A_t)]
> $$
> 
> 其中：
> - $r_t(\theta)=\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$：新旧策略的概率比
> - $A_t$：优势函数（估计当前行为是否优于平均）
> - clip 约束策略变化在$[1-\epsilon,1+\epsilon]$内
> 
> 这种方式能有效防止策略更新“越界”，让学习更稳定。

**PPO 相比其他方法的优化点（不止一个）**

> **优化1：策略更新幅度控制（Clip）**
> 
> 这是 PPO 的最核心创新点。
> - 以前（如 REINFORCE）策略每次更新可能大幅度波动，训练不稳定；
> - PPO 使用 clip 函数控制策略新旧概率比，保持在合理范围；
> - 优点：稳定、简单、高效，训练不会“崩”。
> 
> **优化2：支持多步轨迹采样（采样效率更高）**
> 
> - 比如 A2C 是每一步交互就更新一次参数（每步都同步）；
> - PPO 可以一次收集一整个 batch（完整或部分 episode），然后统一优化；
> - 好处：减少同步成本，提升样本利用效率。
> 
> **优化3：可以使用广义优势估计（GAE）**
> 
> GAE（Generalized Advantage Estimation）是一种对 Advantage 函数的更稳定估计方式；
> PPO 支持这种估计方法，更准确、更稳定；
> 
> $$
> A_t^{GAE}=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}
> $$
> 
> 其中：$\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$
> 
> **优化4：可以结合多个损失项一起训练（Actor-Critic）**
> 
> PPO 通常使用两个损失函数并行训练：
> - 策略损失（用剪切策略比控制更新）；
> - 值函数损失（预测 V 值）；
> - 有时还加上 entropy loss 促进探索。
> 
> PPO 总损失函数一般如下：
> 
> $$
> L=L_{policy}^{CLIP}+c_1·L_{value}-c_2·Entropy
> $$
> 
> **总结**
> 
> | 优化点                | 是否是 PPO 核心     |
> | --------------------- | ------------------- |
> | 策略 clip 限制更新    | ✅ 绝对核心          |
> | 支持 batch+多步采样   | ✅ 提升效率          |
> | 使用 GAE 增强稳定性   | ✅ 常用搭配          |
> | 值函数 + 策略联合训练 | ✅ Actor-Critic 标配 |
> | 轻量实现（相比 TRPO） | ✅ 实用性极强        |

**PPO是on-policy还是off-policy**

> **PPO 为什么是 on-policy？**
> 
> PPO 是一个策略优化器，它优化的是当前策略$\pi_\theta$
> 
> 虽然它“缓解”了 on-policy 的一些缺点，但它本质仍然是 on-policy，因为：
> - 它的经验采集仍然依赖当前策略$\pi$；
> - 策略更新仍基于当前$\pi$与旧策略$\pi_{old}$的比值；
> - 即使允许小幅偏离，也不允许用完全无关的旧数据。
> 
> **那 PPO 为什么样本效率高？哪来的？**
> 
> > PPO 提高样本效率，主要靠以下机制：
> > 
> > **机制1：重复使用旧数据（小范围）——“软”on-policy**
> > 
> > 虽然 PPO 是 on-policy，但它使用了一个$\pi_{old}$机制：$r_t(\theta)=\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$
> > 
> > - 它允许我们用 多个 epoch 重复利用同一批样本；
> > - 只要保证$\pi$与$\pi_{old}$不偏差太大，就不会影响训练质量；
> > - 这种策略更新方式叫做 “trust region” 近似。
> > 
> > **机制2：Clip 策略让“轻微 off-policy”（容忍策略漂移）安全可行**
> > 
> > - PPO 的 clip 限制保证了即便策略发生了一点变化，学习目标依然是可信的；
> > - 这让它能容忍轻微偏离 on-policy，在保证稳定的前提下使用更多数据。
> > - 轻微 off-policy 并非真正 off-policy，是设计上具备一定的**策略漂移容忍度**（policy shift tolerance），这是其样本利用率提升的核心机制之一。
> > 
> > **机制3：GAE（优势估计）+ 批量轨迹更新**
> > 
> > - PPO 允许收集一整个 batch（可能包含多个 episode）的轨迹；
> > - 用 GAE 估计优势后，一次性进行多轮优化；
> > - 所以样本“利用率”高于 REINFORCE / A2C（这些一次用一次就扔）；
> 
> **PPO 是否是完全的 on-policy？**
> 
> 严格意义上讲：
> - PPO 是 近似 on-policy 或称 soft on-policy 方法；
> - 它介于传统 on-policy 和 off-policy 之间；
> - 但它仍不能使用完全离线的数据或完全旧的策略轨迹；
> - 如果把旧经验存到 replay buffer、持续用很久，那就彻底违反 on-policy 假设了（比如 DQN 才这么干）。

</details>


<details>
<summary>loss不断下降，但reward一直震荡，是什么原因</summary>

<br>

**情况 1：模型在过拟合 Critic，但策略没变好**

> Loss 下降是因为模型在拟合 value function（比如 TD 误差），但策略本身并没有学到更多有价值的行为。
> 
> Loss 是对某个目标的优化，而不是直接最大化 reward
> - 常见 loss（如 policy gradient、value loss）优化的是 估计值（如动作概率、Q 值等），而不是直接优化环境返回的 reward。
> - 训练过程中 loss 降低，表示模型更 confident、收敛性更好，但策略是否更优还需看 reward。
> 
> **解决方法**：
> - 观察 policy 的变化（如 action 分布、熵 entropy）是否真的在改变。
> - 增加 entropy bonus 权重（鼓励 exploration）
> - 用 advantage normalization，避免大 advantage 推动过强的梯度更新。

**情况 2：策略陷入局部最优 / 探索不足**

> 训练过程中 loss 下降但 policy 不再探索，reward 停滞或震荡。
> 
> **解决方法**：
> - 调大 entropy regularization 项（鼓励更多探索）
> - 使用 ε-greedy 或 Gumbel noise 加入随机性
> - 如果是 PPO，尝试调大 clip_epsilon 以避免过度收敛。

**情况 3：环境 reward 本身就 noisy / 非平稳**

> 不是模型的问题，是 reward 天然震荡，比如：
> - 游戏有很多随机性（敌人行为随机）
> - reward sparse（只有达成任务才给分）
> 
> **解决方法**：
> - 平滑 reward 曲线（如滑动平均）来更准确评估变化趋势
> - 分析是否是环境随机性主导而非策略退化
> - 多运行几条 trajectory 看 reward 的 方差是否高

</details>


