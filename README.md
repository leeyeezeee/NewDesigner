# GDesigner

## Overview

We provide the code of our paper. The algorithm implementation code is in `GDesigner` folder, and the experimental code is in `experiments` folder.

## Quick Start

### Install packages

```bash
conda create -n gdesigner python=3.10
conda activate gdesigner
pip install -r requirements.txt
```

### Add API keys in `template.env` and change its name to `.env`

```python
BASE_URL = "" # the BASE_URL of OpenAI LLM backend
API_KEY = "" # for OpenAI LLM backend
```

### Download Datasets

Download MMLU, HumanEval and GSM8K datasets from MMLU, HumanEval and GSM8K. And put them in different folders.

### Run GDesigner on MMLU by running the following scripts

```bash
python experiments/run_mmlu.py --mode FullConnected --batch_size 4 --agent_nums 6 --num_iterations 30 --num_rounds 1 --optimized_spatial
```

The above code verifies the experimental results of the `mmlu` dataset under different topologies.

We also provide experimental code for other datasets and topologies.You can refer to `experiments/run_humaneval.py` and `experiments/run_gsm8k.py`.

For example, if you want to verify the results on the `gsm8k` dataset, you can execute the following command

```bash
python experiments/run_gsm8k.py --mode FullConnected --batch_size 4 --agent_nums 4 --num_iterations 30 --num_rounds 1 --optimized_spatial
```

The optimized spatial topology samples Bernoulli decisions directly from the
existing task-conditioned, directed affinity logits. It does not apply SVD,
low-rank projection, or probability clipping. `--refine_rank` is accepted only
for compatibility and has no effect; `--anchor_reg_weight` and
`--sparsity_reg_weight` must remain 0. Old low-rank checkpoints require
retraining because the decoder's probability mapping has changed.

All six dataset entry points default to **30 optimizer updates**
(`--num_iterations 30`), not 30 epochs. With batch size 4 and 8 graph samples,
this means 120 question positions and 960 graph rollouts. The non-MMLU runners
reserve the first 10 batches for training (`--train_split_batches 10`) and
always evaluate on the same remaining full batches. Thirty updates cycle over
that fixed training prefix three times, so the training/evaluation sample split
matches the original 10-update default. Incomplete trailing batches remain
excluded as before. `--num_iterations` changes the update budget only; it no
longer changes the evaluation subset. MMLU continues to use dev for training
and val for evaluation. If an earlier run used a different training boundary,
set `--train_split_batches` to that historical number of batches.

`--use_graph_tf_reward` enables multi-graph mean-centered graph advantages,
while `--edge_ig_reward_lambda` adds per-edge teacher-forcing information gain.
Within-round downstream edge rewards retain a default discount factor of 0.2.

Graph utility is `u_k = correctness_k - beta * T_k / max_l(T_l)`, with the
maximum taken only over samples of the same question. The graph advantage is
`u_k - mean_l(u_l)`; it is **not** divided by the group standard deviation.
`--prompt_token_cost_beta` defaults to **0.5**. Positive beta enables grouped
training (`--graph_sample_count`, default 8, must be at least 2). Set beta to 0
to disable cost; `--use_graph_tf_reward` still enables centered correctness
advantages in that case. Without either group option or full-graph TF, the
existing single-sample correctness path is retained.

`T_k` counts prompt tokens from normal execution of all rounds and the final
decision agent. Concurrent graphs have separate counters; completion tokens and
extra TF/edge-ablation calls do not enter this penalty. An all-zero token group
has zero cost. Training logs include per-graph token counts, utilities, advantages,
and separate gradient norms for correctness, prompt cost, full-graph TF, and IG.
The existing `avg_communication_tokens` metric still counts prompt + completion.

Edge IG still uses `lambda * tanh(discounted_IG / temperature)` and is summed over
selected edges. The graph log-probability is itself a sum over sampled decisions
(including absent edges sampled with action 0), so its graph advantage
already acts on each decision. No additional edge-count division is applied.
The optional full-graph TF ablation remains separately standardized and defaults
to weight 0; its behavior and the edge-IG settings are unchanged.

Edge values are measured with real edge ablation. The
`--edge_ig_warmup_iterations` setting skips those ablation calls during the
initial training iterations.

Each run creates timestamped training and case records under
`result/<dataset>/`, for example `gsm8k_log_20260828_1200.jsonl` and
`gsm8k_cases_20260828_1200.jsonl`. Runs started in the same minute overwrite
that minute's pair; records from other minutes are left unchanged.
The dataset summary remains the append-only `result/<dataset>.jsonl` file.

Only agent nodes belong to the learned adjacency matrices. After all agent
rounds finish, an external decision node receives every agent's latest answer
and produces the final result. Rank no longer restricts the spatial decoder.

Temporal optimization remains opt-in via `--optimized_temporal`. By default,
temporal logits do not require gradients and are excluded from the optimizer.
The existing fixed temporal branch still checks spatial reachability, so its
realized edge set can vary with the spatial graph even though its parameters
are not learned. This temporal behavior is unchanged.

As in the original G-Designer implementation, the default FullConnected mask
makes every non-self agent direction eligible. Edges are considered
sequentially and rejected when adding one would close a cycle; the realized
graph is then executed in topological order. Training, validation, and testing
all use Bernoulli edge sampling—there is no fixed-probability threshold that
hardens the test graph.

```bash
python experiments/run_gsm8k.py --optimized_spatial --use_graph_tf_reward --edge_ig_reward_lambda 1.0
```

## Acknowledgement

This code refers to [GPTSwarm](https://github.com/metauto-ai/GPTSwarm).
