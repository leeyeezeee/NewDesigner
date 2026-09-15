**Token cost、构图参数化与训练长度核查（2026-09-15）**

已将 main 从 3c9941b 快进到 origin/main 的 6e9ca44。远程本次主要将 prompt_token_cost_beta 的默认值从 0.1 改为 0.5；本地 README 小修及图表修改已保留。本轮没有修改构图器、奖励公式或训练默认参数。

目前仓库内没有实际运行的训练日志或 checkpoint，因此以下是代码事实和离线数值复现，不能当作某次真实训练失败的唯一归因。合并后 10 项奖励与计数测试全部通过。数值复现使用 CPU PyTorch 2.14、float64，直接提取当前 Graph 的方法执行，不加载 LLM，不是端到端性能实验。

**1. 成本项已经接入，但观察指标需要对应**

实际奖励为 u_k = R_k - beta * T_k / max_j(T_j)，A_k = u_k - mean_j(u_j)，其中 T_k 是同一道题下第 k 个图的正常 rollout prompt tokens，包含最终决策调用，不含 TF/反事实打分调用。成本通过 beta * (c_k - mean(c)) * log pi(G_k) 进入损失，方向正确，梯度测试非零。

然而训练日志中的 avg_communication_tokens 仍是 prompt + completion；avg_edges 仅统计每轮空间边，不包含跨轮边和最终决策边。不能仅由这两个量判断 prompt-token 惩罚失效。应优先读取 graph_rewards.rollout_prompt_tokens 和 batch_gradient_norms.prompt_token_cost，并按每题平均值比较固定评估集，不能比较题目数变化后的累计 token 总量。

减均值后的优势平均值必然是 0，不能用其均值判断没有学习信号。同组图成本相同时，成本优势确实为 0；差异小时信号也小。例如 beta=0.5、同正确性、两个图 T=10000/10200，其成本优势仅约 +0.004902/-0.004902。提高 beta 不会解决没有组内差异或零梯度区的问题；Adam 下也不能把 beta 增加五倍直接解释为参数更新幅度增加五倍。TF 权重默认未设置时解析为 0，只有运行命令显式启用 TF/IG 时才需要讨论其与成本梯度竞争。

代码：experiments/teacher_forcing_reward.py 的 token_aware_graph_utilities、graph_correctness_advantage_edge_loss；experiments/math_dataset_runner.py 的 append_training_step；experiments/edge_training_log.py。

**2. 默认跨轮通信会抵消空间稀疏化的成本收益**

各主要入口默认 num_rounds=2；FullConnected 的 temporal mask 为全 1。即使只启用 optimized_spatial，construct_temporal_connection 仍会加入跨轮边。更关键的是，固定跨轮分支错误地对上一轮到当前轮的边使用当前轮空间图的 check_cycle。它只检查 spatial_successors：删去空间路径，可能使此前被拒绝的跨轮边重新出现。

直接调用当前方法，假定两轮使用同一空间结构，得到：

| 节点数 | 空间结构 | 每轮空间边 | 第二轮跨轮边 | 两轮通信连接总数 |
| --- | --- | ---: | ---: | ---: |
| 4 | 空图 | 0 | 12 | 12 |
| 4 | 链 | 3 | 6 | 12 |
| 4 | 完全 DAG | 6 | 6 | 18 |
| 5 | 空图 | 0 | 20 | 20 |
| 5 | 链 | 4 | 10 | 18 |
| 5 | 完全 DAG | 10 | 10 | 30 |

这些是消息连接数，不是实际 token 数；真实成本还受消息长度影响。但它们已严格证明“空间边更少”不推出“通信更少”。该问题只适用于有跨轮通信且进入固定跨轮分支的配置；若实际命令使用 num_rounds=1，则不是该次实验的原因。

此外，Graph.arun 每轮仍执行所有智能体，connect_decision_node 总会汇总所有智能体的最后输出。空间边策略不能关闭智能体调用、改变轮数或直接裁掉最终汇总输入。这构成成本可控范围的限制，但输出长度本身仍可能随图改变。

代码：GDesigner/graph/graph.py:624、734、879；experiments/math_dataset_runner.py:548。

**3. rank 的确影响表示，但不是“rank 越大探索越强”**

当前流程是 sigmoid(affinity) -> SVD 的左奇异向量 U_r -> U_r W U_r^T -> clamp(1e-6,1-1e-6) -> logit。W 初始化为单位阵。实际 rank 被限制为 min(requested_rank, N-1)，所以默认 4 个节点时为 3，5 个节点时为 4；将命令行 rank 改成 8 并不会提高它们的有效 rank。

初始化时 U_r W U_r^T = U_r U_r^T，是对称投影矩阵，并非保留原邻接矩阵强度的截断 SVD 重构。奇异值被丢弃。当 r=N-1，U_r U_r^T = I-u_N u_N^T，非对角正值只出现在 u_N 分量异号的节点之间，形成初始二部支持偏置；负值被 clamp 到 1e-6，该边对其截断前数值的直接导数为零。其他边的共享参数更新仍可能让它离开这个区间，并非永久无法恢复。W 后续也不被约束为对称，所以不能声称整个训练期间都只能生成二部图。

对 100 个随机 affinity 输入分别复现当前解码器，W=I：

| N | r | 非对角概率落在下限的平均比例 | 环检测前期望空间边数 |
| --- | --- | ---: | ---: |
| 4 | 3 | 37.67% | 1.4864 |
| 5 | 4 | 41.80% | 1.8287 |
| 4 | 4（绕过构造器上限的诊断） | 100% | 0.000012 |
| 5 | 5（绕过构造器上限的诊断） | 100% | 0.000020 |

后两种 full-rank 诊断中，U U^T=I，非对角概率全部被截到下限，期望边数对 affinity 和 W 的梯度范数均为 0。因此不能简单去掉 N-1 上限并增大 rank。以上百分比不是实际训练模型的测量值。

低秩限制作用于截断前连续矩阵，不能等同于只能采样有限几个离散图模板。另有数值稳定性风险：PyTorch 官方说明，依赖 SVD 奇异向量的梯度在重复或接近重复的奇异值处可能不稳定；本轮未证明实际训练出现该故障。[官方说明](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.svd.html)

代码：GDesigner/graph/graph.py:135、483、492、510。

**4. 默认 10 是优化步骤数，不是 epoch**

默认 batch_size=4、graph_sample_count=8、num_iterations=10，所以是 40 个题目位置、320 次图 rollout、10 次 optimizer.step。重复采样 8 个图不能当成 8 次参数更新。

MMLU 使用独立 dev/val，可以在固定评估集下增加优化步骤。AQuA、GSM8K、HumanEval 等入口则把同一个数据序列的前 num_iterations 批训练、剩余批评估。直接改步骤数会改变评估样本，并在小数据集上消灭评估阶段。

本地 AQuA 文件 254 条、batch=4 时仅 63 个完整批次，当前还丢弃尾部 2 条。10 次更新后评估 212 条；改为 100 后没有评估批次，且不会抵达第 100 批的保存/重置分支，最终标作评估的数值可能实际上是训练阶段累计值。HumanEval 本地文件为 161 条、40 个完整批次，也有同类问题。

代码：experiments/run_mmlu.py:48、101；experiments/math_dataset_runner.py:133、141、452；experiments/run_gsm8k.py、experiments/run_humaneval.py 的同类循环。

**建议修改顺序**

1. 固定 train/validation/test 样本，训练循环与评估循环分开；训练可重复遍历 train，保留不足整批的样本。记录真实 optimizer_steps、已见训练样本数及有效 epoch，评估仅使用同一固定集合。只有 validation 用于选择步数/超参数，test 留作最终报告。
2. 明确时间边语义：上一轮 -> 当前轮的边不应使用当前轮空间 DAG 环检测。需要跨轮成本可学习时显式训练 temporal policy；仅研究空间策略时可先做同协议的单轮对照。修正时间边逻辑后须重跑相关基线，因为消息可达范围发生变化。
3. 增加直接 affinity decoder 作为主要对照：复用已有有向双线性 affinity logits，直接交给 Bernoulli(logits=...)，去掉强制 U_r W U_r^T 和概率 hard clamp；保留旧解码器作消融，保留动态图环检测。可加一个从 0 初始化的可学习标量 logit bias，独立控制密度；这是实现建议，不是已验证的最优结构。若仍需要低秩先验，应在可学习 logits 上表达或作为残差/可选正则，而非把带负数的投影矩阵硬当作概率。
4. 上述解码方向有文献依据：AAAI 2022 DiGAE 使用源/目标编码的非对称内积，之后接 sigmoid（式 13）。这里只借鉴解码形式，并非复现其编码器或声称其已验证本项目的 token 优化。[原论文](https://ojs.aaai.org/index.php/AAAI/article/view/20682/20441)
5. 完成固定数据拆分后，将 100 个 optimizer steps 作为比 10 更合理的初始实验预算，在固定验证集比较 10/50/100/200；100 不是经验上已验证最优，也不是 100 epoch。先只改变解码器，固定奖励和预算，再单独改变训练长度，避免同时改变多项后无法归因。beta 保留远程当前值 0.5，并配套 beta=0 的同种子对照。
6. 每步记录 prompt/completion 分项、空间/时间边分项、有效概率下限占比、组内相对 token 跨度，以及 correctness/cost/IG 梯度范数和方向关系。只有收到实际训练命令与日志，才能判定本次究竟是哪项占主导。

复现脚本与机器可读结果位于 tmp/token_cost_audit/probe.py 和 results.json（本地临时文件，不纳入 Git）。
