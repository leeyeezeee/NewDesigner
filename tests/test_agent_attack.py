import ast
import asyncio
import copy
import json
import random
import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from experiments.agent_attack import AttackedLLM, attack_one_agent, paired_metrics
from experiments.evaluate_aqua_attack import evaluation_indices, predefined_masks
from scripts.plot_agent_attack import load_groups
from test_direct_affinity import Graph, make_graph

ROOT = Path(__file__).resolve().parents[1]


def eval_namespace():
    # Exercise production sampling/replay without importing an LLM or embedding model.
    tree = ast.parse((ROOT / 'experiments/attack_eval_graph.py').read_text(encoding='utf8'))
    tree.body = [item for item in tree.body if not isinstance(item, (ast.Import, ast.ImportFrom))]
    namespace = {'Graph': Graph, 'copy': copy, 'torch': torch}
    Graph.spatial_adj_matrix.fget.__globals__['np'] = np
    exec(compile(tree, 'attack_eval_graph.py', 'exec'), namespace)
    return namespace


class AgentAttackTests(unittest.TestCase):
    def test_fixed_masks_match_existing_random_baseline_and_tree(self):
        # Extract the actual Random branch to catch differences in mask/draw order.
        tree = ast.parse((ROOT / 'experiments/math_dataset_runner.py').read_text(encoding='utf8'))
        branch = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.If) and ast.unparse(node.test) == "mode == 'Random'")
        namespace = {'N': 5, 'random': random.Random(42)}
        exec(compile(ast.Module(body=branch.body, type_ignores=[]), 'random_baseline', 'exec'), namespace)
        masks = predefined_masks('random', 5, random.Random(42))
        self.assertEqual(masks, (namespace['fixed_spatial_masks'], namespace['fixed_temporal_masks']))
        self.assertEqual(masks, predefined_masks('random', 5, random.Random(42)))
        self.assertNotEqual(masks, predefined_masks('random', 5, random.Random(43)))
        spatial, temporal = predefined_masks('tree', 5, random.Random(42))
        self.assertEqual([(i, j) for i in range(5) for j in range(5) if spatial[i][j]],
                         [(0, 1), (0, 2), (1, 3), (1, 4)])
        self.assertEqual(sum(map(sum, temporal)), 25)

    def test_complete_runner_writes_real_pairs_and_resumes_without_new_calls(self):
        from experiments import evaluate_aqua_attack as runner

        class Client:
            async def agen(self, messages, **kwargs):
                return 'B' if 'liar' in messages[0]['content'] else 'A'

        base = SimpleNamespace(
            nodes={str(i): SimpleNamespace(role='role', llm=Client(), execution_history=[]) for i in range(5)},
            num_nodes=5, llm_name='mock', spatial_policy_architecture='test',
            decision_node=SimpleNamespace(outputs=[]),
        )
        calls = []

        async def execute(semaphore, graph, inputs, rounds, **kwargs):
            replies = []
            for node in graph.nodes.values():
                reply = await node.llm.agen([{'role': 'system', 'content': 'Solve.'},
                                            {'role': 'user', 'content': inputs['task']}])
                calls.append(reply)
                replies.append(reply)
                node.execution_history = [{'round': 0, 'outputs': [reply]}]
            answer = 'B' if 'B' in replies else 'A'
            graph.decision_node.outputs = [answer]
            graph.rollout_prompt_tokens = 5
            return [answer], torch.tensor(0.)

        dependencies = {
            'experiments.attack_eval_graph': SimpleNamespace(sample_plan=lambda *a: [{'spatial': [[0, 1]], 'temporal': []}]),
            'experiments.graph_concurrency': SimpleNamespace(limited_graph_arun=execute, make_graph_semaphore=lambda n: None),
        }
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / 'aqua.jsonl'
            dataset.write_text('\n'.join(json.dumps(dict(question=str(i), options=['A) 1', 'B) 2'],
                                                         rationale='', correct='A')) for i in range(44)), encoding='utf8')
            argv = ['evaluate', '--method', 'complete', '--dataset_json', str(dataset),
                    '--output_dir', str(Path(folder) / 'out'), '--limit_questions', '2']
            with patch.object(sys, 'argv', argv):
                args = runner.parse_args()
            with patch.dict(sys.modules, dependencies), patch.object(runner, 'build_graph', return_value=(base, 1)):
                asyncio.run(runner.main(args))
                self.assertEqual(len(calls), 20)
                asyncio.run(runner.main(args))
                self.assertEqual(len(calls), 20)
            summary = json.loads((args.output_dir / 'complete_seed42.summary.json').read_text(encoding='utf8'))
            self.assertEqual(summary['clean_accuracy'], 1)
            self.assertEqual(summary['attack_accuracy'], 0)
            self.assertEqual(summary['question_indices'], [40, 41])

    def test_attack_is_local_preserves_messages_and_all_five_choices(self):
        class Client:
            def gen(self, messages, **kwargs):
                return messages, kwargs

            async def agen(self, messages, **kwargs):
                return self.gen(messages, **kwargs)

        client = Client()
        graph = SimpleNamespace(nodes={str(i): SimpleNamespace(llm=client, role='normal') for i in range(3)},
                                decision_node=SimpleNamespace(llm=client))
        attack_one_agent(graph, 1)
        original = [{'role': 'system', 'content': 'Use Python code.'},
                    {'role': 'user', 'content': 'Question plus spatial and temporal messages.'}]
        attacked, kwargs = asyncio.run(graph.nodes['1'].llm.agen(original, temperature=.2))
        self.assertEqual(original[0]['content'], 'Use Python code.')
        self.assertEqual(attacked[1], original[1])
        self.assertIn('A, B, C, D, E', attacked[0]['content'])
        self.assertIn('Use Python code.', attacked[0]['content'])
        self.assertEqual(kwargs, {'temperature': .2})
        self.assertIs(graph.nodes['0'].llm, client)
        self.assertIs(graph.decision_node.llm, client)
        self.assertEqual(graph.nodes['1'].role, 'normal')
        self.assertEqual(graph.nodes['1'].llm.gen(original)[0], attacked)
        with self.assertRaises(ValueError):
            attack_one_agent(graph, 3)

    def test_replay_reproduces_both_rounds_and_leaves_rng_and_base_untouched(self):
        ns = eval_namespace()
        graph = make_graph(n=4, optimized_temporal=True)
        graph.__class__ = ns['AttackEvalGraph']
        for module in [graph.gat, graph.edge_mlp, graph.spatial_affinity]:
            module.eval()
        state = torch.random.get_rng_state().clone()
        plan = ns['sample_plan'](graph, 'question', 2, 93)
        self.assertEqual(plan, ns['sample_plan'](graph, 'question', 2, 93))
        torch.testing.assert_close(state, torch.random.get_rng_state())
        self.assertEqual(graph.num_edges, 0)
        self.assertEqual(plan[0]['temporal'], [])
        left, right = copy.deepcopy(graph), copy.deepcopy(graph)
        for instance in [left, right]:
            instance.replay_plan = plan
            for r in range(2):
                instance.construct_spatial_connection(r, track_grad=False)
                instance.construct_temporal_connection(r, track_grad=False)
                for kind in ['spatial', 'temporal']:
                    actual = getattr(instance, kind + '_adj_matrix')
                    expected = np.zeros((4, 4))
                    for s, t in plan[r][kind]:
                        expected[s, t] = 1
                    np.testing.assert_array_equal(actual, expected)

    def test_legacy_decoder_known_rank_one_projection(self):
        ns = eval_namespace()
        # A constant sketch has rank-one projector 11^T / 3.
        probabilities = ns['legacy_probabilities'](
            torch.zeros(3, 3), 1 - torch.eye(3), torch.tensor([[2.]]), 1)
        expected = torch.full((3, 3), 2 / 3)
        expected.fill_diagonal_(1e-6)
        torch.testing.assert_close(probabilities, expected)
        graph = make_graph(n=3)
        graph.__class__ = ns['AttackEvalGraph']
        raw = torch.tensor([[0., 2., -3.], [4., 0., 1.], [-1., 3., 0.]])
        torch.testing.assert_close(graph._affinity_spatial_logits(raw), raw.flatten())

    def test_historical_aqua_split_and_paired_metric_denominator(self):
        self.assertEqual(evaluation_indices(254, 4, 10), list(range(40, 252)))
        self.assertEqual(evaluation_indices(254, 4, 10, 2), [40, 41])
        cases = [{'clean': {'correct': a}, 'attack': {'correct': b}}
                 for a, b in [(True, False), (True, True), (False, True), (False, False)]]
        scores = paired_metrics(cases)
        self.assertEqual(scores['clean_accuracy'], .5)
        self.assertEqual(scores['accuracy_drop_pp'], 0)
        self.assertEqual(scores['correct_to_wrong_rate'], .5)

    def test_plot_rejects_unmatched_protocols_or_incomplete_results(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / f'{i}.json' for i in range(2)]
            row = dict(dataset='aqua', dataset_sha256='test', question_indices=[40, 41],
                       attack='liar', attack_prompt='prompt', llm_name='model', num_rounds=1,
                       protocol='frozen', paired_topology=True, num_questions=2,
                       clean_accuracy=.5, attack_accuracy=0, seed=42)
            paths[0].write_text(json.dumps(dict(row, method='Complete')), encoding='utf8')
            paths[1].write_text(json.dumps(dict(row, method='Tree')), encoding='utf8')
            self.assertEqual(len(load_groups(paths)), 2)
            for changed in [dict(num_questions=1), dict(question_indices=[41, 42]), dict(seed=43)]:
                paths[1].write_text(json.dumps(dict(row, method='Tree', **changed)), encoding='utf8')
                with self.assertRaises(ValueError):
                    load_groups(paths)


if __name__ == '__main__':
    unittest.main()
