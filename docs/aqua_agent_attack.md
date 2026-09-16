# AQuA 单 agent 攻击评估

不需要重新训练。该实验冻结已有模型，仅对一个普通 agent 的 system prompt
追加恶意指令。它要求错误答案和错误分析，适配 AQuA 的 A–E 五个选项，保留
各角色原输出格式，包括编程角色的 Python 格式。此模板参考 AgentPrune 附录 G.3，
属于适配版本，并非原样复制其四选一模板。不读取正确答案，也不强制每次攻击成功。

## 在已有运行环境中执行

从 NewDesigner 仓库根目录运行，使用原实验的 Python 环境和 Qwen3-8B 服务。
下列命令假设 `.env` 已配置 `AGENT_BASE_URL`、`AGENT_API_TYPE` 和凭证。
也可在每条评估命令后追加 `--agent_base_url http://localhost:8005/v1 --agent_api_type vllm`，
端口应替换为实际服务地址。脚本不会部署模型或连接 SSH。

```bash
# 先用两题检查服务和 checkpoint；与正式结果分目录
python -m experiments.evaluate_aqua_attack --method edgeig --checkpoint result/checkpoints/aqua_edgeig_beta0.5.pt --limit_questions 2 --output_dir result/aqua_attack_smoke

# 固定结构没有训练过程。默认 5 个 agent、1 轮，与现有 AQuA 结果一致。
python -m experiments.evaluate_aqua_attack --method complete --seeds 42 43 44
python -m experiments.evaluate_aqua_attack --method random --seeds 42 43 44
python -m experiments.evaluate_aqua_attack --method tree --seeds 42 43 44

# 一个已有 checkpoint 上重复三个评估种子，不代表三个独立训练种子。
python -m experiments.evaluate_aqua_attack --method edgeig --checkpoint result/checkpoints/aqua_edgeig_beta0.5.pt --seeds 42 43 44

# 只读取真实、完整、同口径的结果，生成 PDF 和 PNG
python scripts/plot_agent_attack.py
```

每个种子随机选中一个 agent，全程固定；可用 `--attack_agent 0` 指定位置。
使用第 40–251 条记录（从零开始，共 212 题），训练前缀与原实现保持一致；
默认丢弃最后不完整的批次。更改数据划分或轮数后必须使用新的输出目录。
clean / attack 重放相同空间及跨轮边，LLM 输出仍可能具有随机性。
当前最终汇总方式保持 FinalRefer；不是 GPTSwarm 的多数投票实验。

Tree 在此明确采用父节点 `(child - 1) // 2` 的有向二叉树。
这是当前新增的固定基线；若服务器原有 Tree 定义不同，需统一后才能比较。
Random 沿用仓库原有 Random 基线：每个种子以 0.5 概率生成空间候选边
（不含自环）和跨轮候选边，然后固定用于全部问题；空间边仍经过原有的防环检查。
Complete 的空间候选边覆盖所有不同节点，实际执行同样经过防环检查，因此是 DAG。
Complete 和 Tree 多轮时使用默认的全跨轮候选连接；Random 使用上述随机跨轮掩码。

每个方法/种子生成 `.config.json`、`.cases.jsonl`、`.summary.json`。
重复相同命令会跳过已完成的配对问题；配置不同时拒绝混写。
输出包括攻击前后准确率、下降百分点、正确变错误比例、各节点各轮回答、
实际拓扑及 rollout prompt tokens。按本次配对运行重新测量 clean，
不将历史 clean 数值拼到新的 attack 结果上。

## checkpoint 兼容

支持本仓库 v8 和现有 AQuA v7 checkpoint。v7 仅在评估子类中使用原来的
低秩 SVD 解码公式和 checkpoint 内 refinement_weight；不改变训练端的 v8
解码器，也不把旧权重解释为新架构。不兼容的其他模型会报错。
普通角色按原 AQuA 的固定顺序重建；自定义角色/agent 的旧模型不在此简化入口的支持范围内。

本地现存 AQuA v7 模型为 5 个 agent、1 轮；若想评估新的无 rank 模型，
请传入服务器上训练完成的 v8 checkpoint。攻击实验本身不要求重训。

## ARG-Designer / AgentPrune

本地未提供这两个方法的运行代码与 checkpoint，所以本入口不冒充它们的推理器。
将 `experiments/agent_attack.py` 中的 `AttackedLLM` 接到各方法选中节点的 LLM
客户端即可复用攻击（要求客户端使用相同的 `agen/gen(messages, **kwargs)` 接口）。
必须在各自真实推理流程中运行 clean / attack，固定其本来采样的角色及拓扑，
并保存同格式 summary；不能仅重命名 EdgeIG 或固定图结果。

画图脚本支持这些方法的 summary，`method` 分别填写 `ARG-Designer`、`AgentPrune`。
必须携带本评估器 summary 中的协议元数据及相同问题索引、模型、轮数、模板和种子集合，
正确率字段为 `[0,1]` 范围的 `clean_accuracy`、`attack_accuracy`。
真实方法的 agent 数或角色不同可以保留，但需在论文中报告。

## 解释范围

这是正常训练后、推理期间遭遇攻击的系统鲁棒性实验，没有在线优化或 TF 边消融开销。
空间边、默认跨轮消息以及所有 agent 到最终汇总节点的通道均保留原逻辑。
因此它不能单独证明模型已识别或隔离恶意节点。
