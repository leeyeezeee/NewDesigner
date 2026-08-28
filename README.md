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
python experiments/run_mmlu.py --mode FullConnected --batch_size 4 --agent_nums 6 --num_iterations 10 --num_rounds 1 --optimized_spatial
```

The above code verifies the experimental results of the `mmlu` dataset under different topologies.

We also provide experimental code for other datasets and topologies.You can refer to `experiments/run_humaneval.py` and `experiments/run_gsm8k.py`.

For example, if you want to verify the results on the `gsm8k` dataset, you can execute the following command

```bash
python experiments/run_gsm8k.py --mode FullConnected --batch_size 4 --agent_nums 4 --num_iterations 10 --num_rounds 1 --optimized_spatial
```

The optimized topology combines G-Designer's low-rank refinement regularization
with IGPO-style teacher forcing. `--use_graph_tf_reward` enables multi-graph
correctness advantages, while `--edge_ig_reward_lambda` adds per-edge
teacher-forcing information gain. Within-round downstream edge rewards use a
default discount factor of 0.2. The refinement rank defaults to 4. The
optional anchor and nuclear-norm penalties default to 0 and can be enabled with
`--anchor_reg_weight` and `--sparsity_reg_weight`:

Real edge ablation remains the default edge-value estimator. The optional
`--use_graph_critic` path instead fits a small, independent two-layer GCN from
the actor's cached pre-GAT role + question features and the sampled adjacency;
it does not invoke the frozen embedding model again. Mean-pooled graph features
pass through a `16 -> 16 -> 1` MLP with an unrestricted scalar output. It uses
one full-graph teacher-forcing score per sampled graph, centers the scores among
graphs sampled for the same question, and estimates a selected edge with
`Q(question, A) - Q(question, A without edge)`. Detached examples are kept in a
256-sample FIFO replay by default and the critic takes four 32-sample MSE updates
after each actor update. The critic learning rate defaults to 0.03. During its
two-iteration warmup the critic is trained but its predicted edge reward does
not affect the actor; real edge-ablation warmup instead skips ablation calls.
`--edge_ig_reward_lambda` is the single edge-TF reward weight in both modes;
`--use_graph_critic` only switches its estimator from real deletion to critic
prediction, so the two estimators cannot be enabled together.

Each run creates timestamped training and case records under
`result/<dataset>/`, for example `gsm8k_log_20260828_1200.jsonl` and
`gsm8k_cases_20260828_1200.jsonl`. Runs started in the same minute overwrite
that minute's pair; records from other minutes are left unchanged.
The dataset summary remains the append-only `result/<dataset>.jsonl` file.

Only agent nodes belong to the learned adjacency matrices. After all agent
rounds finish, an external decision node receives every agent's latest answer
and produces the final result. The effective refinement rank is capped below
the number of agent nodes so the decoder remains genuinely low-rank.

As in the original G-Designer implementation, the default FullConnected mask
makes every non-self agent direction eligible. Edges are considered
sequentially and rejected when adding one would close a cycle; the realized
graph is then executed in topological order. Training, validation, and testing
all use Bernoulli edge sampling—there is no fixed-probability threshold that
hardens the test graph.

```bash
python experiments/run_gsm8k.py --optimized_spatial --use_graph_tf_reward --edge_ig_reward_lambda 1.0
```

```bash
python experiments/run_gsm8k.py --optimized_spatial --use_graph_critic --graph_sample_count 8 --edge_ig_reward_lambda 1.0
```

## Acknowledgement

This code refers to [GPTSwarm](https://github.com/metauto-ai/GPTSwarm).
