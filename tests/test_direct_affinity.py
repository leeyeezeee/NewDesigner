"""Exercise real Graph methods with local stubs for LLM/GNN infrastructure.

Loading the class AST avoids importing optional remote-model dependencies.
The constructor, affinity path, samplers and checkpoint code are unchanged
production methods; only agent creation, embeddings and the GAT are stubbed.
"""
import argparse
import ast
import math
import tempfile
import unittest
from abc import ABC
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from experiments.checkpoint import load_graph_checkpoint, save_graph_checkpoint
from experiments.refinement_loss import refinement_regularization_loss
from experiments.training_schedule import add_training_split_args
from experiments.teacher_forcing_reward import (
    add_full_graph_reward_ablation_args, add_teacher_forcing_reward_args,
)

ROOT = Path(__file__).resolve().parents[1]


class LocalNode:
    def __init__(self, node_id):
        self.id = node_id
        self.spatial_successors = []
        self.spatial_predecessors = []
        self.temporal_successors = []
        self.temporal_predecessors = []

    def add_successor(self, other, kind):
        getattr(self, kind + '_successors').append(other)
        getattr(other, kind + '_predecessors').append(self)


class LocalEncoder(torch.nn.Module):
    def __init__(self, in_channels, *, out_channels, **kwargs):
        super().__init__()
        self.projection = torch.nn.Linear(in_channels, out_channels)

    def forward(self, features, edge_index, return_diagnostics):
        output = self.projection(features)
        weights = output.new_ones((edge_index.shape[1], 1))
        diagnostics = {'initial': output, 'layer1': output, 'layer2': output,
                       'initial_residual_weight': .1}
        for layer in [1, 2]:
            diagnostics[f'attention{layer}_edge_index'] = edge_index
            diagnostics[f'attention{layer}_weights'] = weights
        return output, diagnostics


def load_graph_class():
    tree = ast.parse((ROOT / 'GDesigner/graph/graph.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Graph')
    module = ast.Module(body=[
        ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
        cls,
    ], type_ignores=[])
    namespace = {
        'ABC': ABC, 'torch': torch, 'math': math, 'F': F,
        'spectral_norm': spectral_norm,
        'shortuuid': SimpleNamespace(ShortUUID=lambda: SimpleNamespace(random=lambda **kw: 'test')),
        'PromptSetRegistry': SimpleNamespace(get=lambda _: None),
        'AgentRegistry': SimpleNamespace(get=lambda *a, **kw: LocalNode('decision')),
        'InitialResidualGATv2Encoder': LocalEncoder,
        'MLP': lambda input_dim, hidden_dim, output_dim: torch.nn.Linear(input_dim, output_dim),
    }
    exec(compile(ast.fix_missing_locations(module), 'GDesigner/graph/graph.py', 'exec'), namespace)
    return namespace['Graph']


Graph = load_graph_class()


def make_graph(n=4, rank=4, optimized_temporal=False):
    def init_nodes(graph):
        graph.nodes = {str(i): LocalNode(str(i)) for i in range(len(graph.agent_names))}

    with patch.object(Graph, 'init_nodes', init_nodes), \
         patch.object(Graph, 'construct_adj_matrix', lambda g: torch.arange(n).repeat(2, 1)), \
         patch.object(Graph, 'construct_features', lambda g: torch.eye(n)):
        graph = Graph('test', 'test-model', ['agent'] * n, 'decision',
                      optimized_spatial=True, optimized_temporal=optimized_temporal,
                      refine_rank=rank)
    graph.construct_new_features = lambda _: torch.cat((torch.eye(n), torch.ones(n, n)), dim=1)
    return graph


class DirectAffinityTests(unittest.TestCase):
    def test_logits_preserve_direction_extremes_and_gradients(self):
        graph = make_graph(n=2)
        raw = torch.tensor([[0., -30.], [30., 0.]], requires_grad=True)
        logits = graph._affinity_spatial_logits(raw)
        torch.testing.assert_close(logits, raw.flatten())
        probs = graph.spatial_edge_probabilities
        self.assertLess(probs[1].item(), 1e-6)
        self.assertGreater(probs[2].item(), 1 - 1e-6)
        self.assertEqual(probs[0].item(), 0.)
        # Unlikely actions retain useful gradients even beyond the old clamp.
        loss = -torch.distributions.Bernoulli(logits=logits[[1, 2]]).log_prob(torch.tensor([1., 0.])).sum()
        grad = torch.autograd.grad(loss, raw)[0]
        self.assertLess(grad[0, 1].item(), -.99)
        self.assertGreater(grad[1, 0].item(), .99)

    def test_affinity_pipeline_uses_no_svd_or_refinement_parameters(self):
        for rank in [1, 4, 20]:
            torch.manual_seed(7)
            graph = make_graph(rank=rank)
            self.assertEqual(graph.refine_rank, 0)
            self.assertFalse(hasattr(graph, 'refinement_weight'))
            with patch('torch.linalg.svd', side_effect=AssertionError('SVD called')), \
                 patch('torch.linalg.svdvals', side_effect=AssertionError('SVD called')):
                graph.prepare_spatial_logits('question')
                loss = graph.spatial_logits.square().sum()
                loss.backward()
                reg = refinement_regularization_loss([graph], loss)
            self.assertTrue(all(value.item() == 0 for value in reg))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in graph.spatial_parameters()))
            self.assertGreater(sum(p.grad.abs().sum().item() for p in graph.spatial_parameters()), 0)
            if rank == 1:
                expected = graph.spatial_logits.detach()
            else:
                torch.testing.assert_close(graph.spatial_logits, expected)
        with self.assertRaisesRegex(ValueError, 'must both be 0'):
            refinement_regularization_loss([graph], loss, anchor_reg_weight=.1)

    def test_masks_cycle_checks_and_absent_edge_gradients(self):
        graph = make_graph(n=3)
        raw = torch.zeros(3, 3, requires_grad=True)
        graph.spatial_logits = graph._affinity_spatial_logits(raw)
        # With all candidate draws accepted, cycle checks still yield a DAG.
        with patch.object(torch.distributions.Bernoulli, 'sample', lambda d: torch.ones_like(d.logits)):
            log_prob = graph.construct_spatial_connection()
        self.assertEqual(graph.num_edges, 3)
        self.assertEqual(len(graph._spatial_topological_order()), 3)
        self.assertEqual(len(graph.edge_log_probs), 3)
        self.assertAlmostEqual(log_prob.item(), -3 * math.log(2), places=5)
        # Absent decisions also receive the graph-level reward gradient.
        graph.spatial_logits = graph._affinity_spatial_logits(raw)
        with patch.object(torch.distributions.Bernoulli, 'sample', lambda d: torch.zeros_like(d.logits)):
            log_prob = graph.construct_spatial_connection()
        self.assertEqual(graph.num_edges, 0)
        grad = torch.autograd.grad(log_prob, raw)[0]
        torch.testing.assert_close(grad, -.5 * (1 - torch.eye(3)))

    def test_temporal_policy_is_fixed_unless_explicitly_enabled(self):
        fixed = make_graph(n=3)
        self.assertFalse(fixed.temporal_logits.requires_grad)
        self.assertNotIn(id(fixed.temporal_logits), {id(p) for p in fixed.spatial_parameters()})
        with patch.object(torch.distributions.Bernoulli, 'sample', side_effect=AssertionError('fixed temporal sampled')):
            value = fixed.construct_temporal_connection(round=1)
        self.assertFalse(value.requires_grad)
        self.assertEqual(sum(len(n.temporal_successors) for n in fixed.nodes.values()), 6)
        learned = make_graph(n=3, optimized_temporal=True)
        self.assertTrue(learned.temporal_logits.requires_grad)
        with patch.object(torch.distributions.Bernoulli, 'sample', lambda d: torch.zeros_like(d.logits)):
            value = learned.construct_temporal_connection(round=1)
        grad = torch.autograd.grad(value, learned.temporal_logits)[0]
        self.assertGreater(grad.abs().sum().item(), 0.)

    def test_checkpoint_roundtrip_and_old_decoder_rejection(self):
        source = make_graph()
        optimizer = torch.optim.Adam(source.spatial_parameters())
        source.prepare_spatial_logits('q')
        source.spatial_logits.square().mean().backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'graph.pt')
            save_graph_checkpoint(source, path, dataset='test', optimizer=optimizer)
            target = make_graph(rank=20)
            target_optimizer = torch.optim.Adam(target.spatial_parameters())
            checkpoint = load_graph_checkpoint(target, path, load_optimizer=target_optimizer)
            self.assertNotIn('refinement_weight', checkpoint['graph'])
            self.assertEqual(len(target_optimizer.state), len(optimizer.state))
            for graph in [source, target]:
                graph.gat.eval()
                graph.edge_mlp.eval()
                graph.spatial_affinity.eval()
                graph.prepare_spatial_logits('q', track_grad=False)
            torch.testing.assert_close(source.spatial_logits, target.spatial_logits)
            for old_version in ['initial_residual_gatv2_affinity_low_rank_dynamic_cycle_external_decision_v7', None]:
                checkpoint['graph']['spatial_policy_architecture'] = old_version
                torch.save(checkpoint, path)
                with self.assertRaisesRegex(ValueError, 'incompatible'):
                    load_graph_checkpoint(target, path)

    def test_all_dataset_cli_defaults_and_overrides(self):
        for name in ['mmlu', 'gsm8k', 'humaneval', 'aqua', 'multiarith', 'svamp']:
            tree = ast.parse((ROOT / f'experiments/run_{name}.py').read_text(encoding='utf-8'))
            parse = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_args')
            namespace = {'argparse': argparse, 'os': SimpleNamespace(makedirs=lambda *a, **kw: None),
                         'GDesigner_ROOT': ROOT,
                         'add_training_split_args': add_training_split_args,
                         'add_teacher_forcing_reward_args': add_teacher_forcing_reward_args,
                         'add_full_graph_reward_ablation_args': add_full_graph_reward_ablation_args}
            exec(compile(ast.Module(body=[parse], type_ignores=[]), f'run_{name}.py', 'exec'), namespace)
            with patch('sys.argv', ['run']):
                args = namespace['parse_args']()
            self.assertEqual(args.num_iterations, 30, name)
            self.assertFalse(args.optimized_temporal, name)
            if name != 'mmlu':
                self.assertEqual(args.train_split_batches, 10, name)
            with patch('sys.argv', ['run', '--num_iterations', '7', '--optimized_temporal']):
                args = namespace['parse_args']()
            self.assertEqual(args.num_iterations, 7, name)
            self.assertTrue(args.optimized_temporal, name)


if __name__ == '__main__':
    unittest.main()
