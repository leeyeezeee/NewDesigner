"""Paired AQuA agent-attack evaluation. No optimizer, TF scoring, or retraining.

Examples (run from the repository root in its existing Python environment):
  python -m experiments.evaluate_aqua_attack --method complete
  python -m experiments.evaluate_aqua_attack --method random
  python -m experiments.evaluate_aqua_attack --method tree
  python -m experiments.evaluate_aqua_attack --method edgeig --checkpoint result/checkpoints/aqua.pt
"""
import argparse
import asyncio
import copy
import hashlib
import json
import random
import time
from pathlib import Path

from experiments.agent_attack import attack_one_agent, paired_metrics, LIAR_PROMPT
from experiments.agent_backend import add_agent_backend_args, apply_agent_backend_args


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['complete', 'random', 'tree', 'edgeig'], required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--llm_name', default=None, help='Defaults to checkpoint model or Qwen3-8B.')
    parser.add_argument('--num_agents', type=int, default=None)
    parser.add_argument('--num_rounds', type=int, default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--attack_agent', type=int, default=None, help='Zero-based; otherwise randomly select once per seed.')
    parser.add_argument('--dataset_json', type=Path, default=Path('datasets/AQuA/AQuA.jsonl'))
    parser.add_argument('--batch_size', type=int, default=4, help='Historical data split batch size; not concurrency.')
    parser.add_argument('--train_split_batches', type=int, default=10)
    parser.add_argument('--limit_questions', type=int, default=None, help='Smoke tests only; omit for paper results.')
    parser.add_argument('--max_concurrent_graphs', type=int, default=10)
    parser.add_argument('--output_dir', type=Path, default=Path('result/aqua_attack'))
    add_agent_backend_args(parser)
    args = parser.parse_args()
    if args.method == 'edgeig' and args.checkpoint is None:
        parser.error('--method edgeig requires --checkpoint; random weights are not an evaluation.')
    if args.method != 'edgeig' and args.checkpoint is not None:
        parser.error('Fixed baselines do not load a learned checkpoint.')
    if args.batch_size <= 0 or args.train_split_batches <= 0 or args.max_concurrent_graphs <= 0:
        parser.error('Batch size, training prefix and concurrency must be positive.')
    if args.limit_questions is not None and args.limit_questions <= 0:
        parser.error('--limit_questions must be positive.')
    return args


def evaluation_indices(size, batch_size, train_batches, limit=None):
    indices = list(range(batch_size * train_batches, size // batch_size * batch_size))
    if not indices:
        raise ValueError('The selected training prefix leaves no evaluation examples.')
    return indices if limit is None else indices[:limit]


def predefined_masks(method, n, rng):
    """Match the existing Random baseline: one spatial/temporal mask per seed."""
    spatial = [[int(i != j) for j in range(n)] for i in range(n)]
    temporal = [[1 for _ in range(n)] for _ in range(n)]
    if method == 'random':
        spatial = [[rng.randint(0, 1) if i != j else 0 for i in range(n)] for j in range(n)]
        temporal = [[rng.randint(0, 1) for _ in range(n)] for _ in range(n)]
    elif method == 'tree':
        spatial = [[int(j > 0 and (j - 1) // 2 == i) for j in range(n)] for i in range(n)]
    return spatial, temporal


def build_graph(args):
    import torch
    from experiments.attack_eval_graph import AttackEvalGraph, LEGACY_ARCH
    from experiments.checkpoint import load_graph_checkpoint

    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False) if args.checkpoint else {}
    graph_state, saved_args = saved.get('graph', {}), saved.get('args', {})
    if saved and (saved.get('dataset') != 'aqua' or graph_state.get('domain') != 'aqua'):
        raise ValueError('Use an AQuA checkpoint from this repository.')
    names = graph_state.get('agent_names', ['MathSolver_aqua'] * (args.num_agents or 5))
    if args.num_agents is not None and len(names) != args.num_agents:
        raise ValueError('Agent count must match the checkpoint.')
    if len(names) < 2 or any(name != 'MathSolver_aqua' for name in names):
        raise ValueError('This small evaluator supports AQuA MathSolver_aqua graphs with at least two agents.')
    rounds = args.num_rounds if args.num_rounds is not None else saved_args.get('num_rounds', 1)
    if rounds <= 0:
        raise ValueError('Number of rounds must be positive.')
    # Explicit roles avoid the process-global role cycle advancing across seeds.
    roles = ['Math Solver', 'Mathematical Analyst', 'Programming Expert', 'Inspector']
    n = len(names)
    spatial, temporal = predefined_masks(args.method, n, random)
    graph = AttackEvalGraph(
        domain='aqua', llm_name=args.llm_name or graph_state.get('llm_name', 'Qwen3-8B'),
        agent_names=names, decision_method=saved_args.get('decision_method', 'FinalRefer'),
        node_kwargs=[{'role': roles[i % len(roles)]} for i in range(n)],
        optimized_spatial=graph_state.get('optimized_spatial', False),
        optimized_temporal=graph_state.get('optimized_temporal', False),
        fixed_spatial_masks=spatial,
        fixed_temporal_masks=temporal,
    )
    if args.checkpoint:
        architecture = graph_state.get('spatial_policy_architecture')
        if architecture == LEGACY_ARCH:
            graph.spatial_policy_architecture = LEGACY_ARCH
            graph.refine_rank = int(graph_state['refine_rank'])
            graph.legacy_weight = graph_state['refinement_weight'].detach()
            if graph.legacy_weight.shape != (graph.refine_rank, graph.refine_rank):
                raise ValueError('Invalid legacy refinement weight shape.')
        # The normal loader still rejects all other unsupported architectures.
        load_graph_checkpoint(graph, str(args.checkpoint))
    for module in [graph.gat, graph.edge_mlp, graph.spatial_affinity]:
        module.eval()
        module.requires_grad_(False)
    graph.temporal_logits.requires_grad_(False)
    return graph, rounds


async def main(args):
    # Import the training runtime only after argparse, so --help is lightweight.
    import torch
    from datasets.aqua_dataset import aqua_data_process, aqua_get_predict
    from experiments.attack_eval_graph import sample_plan
    from experiments.graph_concurrency import limited_graph_arun, make_graph_semaphore

    apply_agent_backend_args(args)
    data_bytes = args.dataset_json.read_bytes()
    dataset = aqua_data_process([json.loads(line) for line in data_bytes.decode('utf8').splitlines() if line.strip()])
    indices = evaluation_indices(len(dataset), args.batch_size, args.train_split_batches, args.limit_questions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None
    labels = {'complete': 'Complete', 'random': 'Random', 'tree': 'Tree', 'edgeig': 'EdgeIG'}

    for seed in args.seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        graph, rounds = build_graph(args)
        attacker = args.attack_agent if args.attack_agent is not None else random.Random(seed).randrange(graph.num_nodes)
        if not 0 <= attacker < graph.num_nodes:
            raise ValueError('--attack_agent is outside the ordinary agent range.')
        config = {
            'dataset': 'aqua', 'method': labels[args.method], 'seed': seed,
            'attack': 'agentprune_style_liar_aqua_v1', 'attack_prompt': LIAR_PROMPT,
            'attack_agent': attacker, 'num_rounds': rounds,
            'roles': [node.role for node in graph.nodes.values()],
            'llm_name': graph.llm_name, 'checkpoint_sha256': checkpoint_sha,
            'architecture': graph.spatial_policy_architecture,
            'dataset_sha256': hashlib.sha256(data_bytes).hexdigest(),
            'question_indices': indices, 'paired_topology': True,
            'protocol': 'frozen_checkpoint_inference_attack',
        }
        stem = args.output_dir / f'{args.method}_seed{seed}'
        config_path = stem.with_suffix('.config.json')
        cases_path = stem.with_suffix('.cases.jsonl')
        if config_path.exists() and json.loads(config_path.read_text(encoding='utf8')) != config:
            raise ValueError(f'Configuration differs from {config_path}; choose another --output_dir.')
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf8')
        cases = [json.loads(line) for line in cases_path.read_text(encoding='utf8').splitlines() if line] if cases_path.exists() else []
        completed = {item['question_id'] for item in cases}
        if len(completed) != len(cases) or not completed.issubset(set(indices)):
            raise ValueError('Duplicate or unexpected question IDs in resume file.')
        semaphore = make_graph_semaphore(args.max_concurrent_graphs)

        async def run_condition(index, plan, attacked):
            instance = copy.deepcopy(graph)
            instance.replay_plan = plan
            if attacked:
                attack_one_agent(instance, attacker)
            start = time.monotonic()
            answers, _ = await limited_graph_arun(
                semaphore, instance, {'task': dataset[index]['task']}, rounds,
                track_grad=False, record_execution_history=True,
            )
            traces = [{'agent': i, 'role': node.role, 'round': history['round'],
                       'outputs': [str(output) for output in history['outputs']]}
                      for i, node in enumerate(instance.nodes.values())
                      for history in node.execution_history]
            expected = graph.num_nodes * rounds
            if len(traces) != expected or any(not item['outputs'] or not all(x.strip() for x in item['outputs']) for item in traces):
                raise RuntimeError(f'Incomplete rollout for question {index}; resume after fixing backend errors.')
            if not instance.decision_node.outputs or not str(answers[0]).strip():
                raise RuntimeError(f'Missing final answer for question {index}.')
            predicted = aqua_get_predict(str(answers[0]))
            return {'answer': predicted, 'correct': predicted == dataset[index]['answer'],
                    'prompt_tokens': instance.rollout_prompt_tokens,
                    'wall_seconds': time.monotonic() - start, 'agent_outputs': traces}

        # Bound the pending work too; results are written after each completed pair.
        pending = [index for index in indices if index not in completed]
        for offset in range(0, len(pending), args.max_concurrent_graphs):
            async def pair(index):
                plan = sample_plan(graph, dataset[index]['task'], rounds, seed * 1000003 + index)
                clean, attacked = await asyncio.gather(run_condition(index, plan, False), run_condition(index, plan, True))
                return {'question_id': index, 'target': dataset[index]['answer'],
                        'topology': plan, 'clean': clean, 'attack': attacked}
            tasks = [asyncio.create_task(pair(index)) for index in pending[offset:offset + args.max_concurrent_graphs]]
            try:
                for task in asyncio.as_completed(tasks):
                    item = await task
                    with cases_path.open('a', encoding='utf8') as file:
                        file.write(json.dumps(item, ensure_ascii=False) + '\n')
                    cases.append(item)
                    print(f'{labels[args.method]} seed={seed}: {len(cases)}/{len(indices)}', flush=True)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        summary = {**config, **paired_metrics(cases)}
        stem.with_suffix('.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
        print(json.dumps({k: summary[k] for k in ['method', 'seed', 'num_questions', 'clean_accuracy', 'attack_accuracy', 'accuracy_drop_pp']}), flush=True)


if __name__ == '__main__':
    asyncio.run(main(parse_args()))
