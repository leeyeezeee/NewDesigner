"""Frozen topology replay and explicit v7 checkpoint support, only for evaluation."""
import copy

import torch

from GDesigner.graph.graph import Graph

LEGACY_ARCH = 'initial_residual_gatv2_affinity_low_rank_dynamic_cycle_external_decision_v7'


def legacy_probabilities(raw, mask, weight, rank, temperature=1.0):
    """Exact inference decoder from graph.py at 6e9ca44 (no training losses)."""
    sketch = torch.sigmoid(raw / max(float(temperature), 1e-6))
    basis = torch.linalg.svd(sketch, full_matrices=False)[0][:, :rank]
    refined = basis @ weight[:rank, :rank].to(raw) @ basis.t()
    return (refined * mask.to(raw)).clamp(1e-6, 1 - 1e-6)


class AttackEvalGraph(Graph):
    replay_plan = None
    legacy_weight = None

    def _affinity_spatial_logits(self, raw):
        if self.legacy_weight is None or not self.optimized_spatial:
            return super()._affinity_spatial_logits(raw)
        probabilities = legacy_probabilities(
            raw, self.spatial_masks.reshape(raw.shape), self.legacy_weight,
            self.refine_rank, self.spatial_sampling_temperature,
        )
        self.spatial_edge_probabilities = probabilities.reshape(-1)
        return torch.logit(probabilities).reshape(-1)

    def prepare_spatial_logits(self, task, track_grad=True):
        if self.replay_plan is None:
            return super().prepare_spatial_logits(task, track_grad=False)

    def _replay_connections(self, kind, round_idx):
        if kind == 'spatial':
            self.clear_spatial_connection()
        else:
            self.clear_temporal_connection()
        nodes = list(self.nodes.values())
        for source, target in self.replay_plan[round_idx][kind]:
            nodes[source].add_successor(nodes[target], kind)
        return torch.tensor(0.0)

    def construct_spatial_connection(self, round=0, **kwargs):
        if self.replay_plan is not None:
            return self._replay_connections('spatial', round)
        return super().construct_spatial_connection(round, **kwargs)

    def construct_temporal_connection(self, round=0, **kwargs):
        if self.replay_plan is not None:
            return self._replay_connections('temporal', round)
        return super().construct_temporal_connection(round, **kwargs)


def sample_plan(base, task, rounds, seed):
    graph = copy.deepcopy(base)
    plan = []
    # No await inside this scope: graph sampling cannot race with other tasks.
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(seed)
        graph.prepare_spatial_logits(task, track_grad=False)
        for round_idx in range(rounds):
            graph.construct_spatial_connection(round_idx, track_grad=False)
            graph.construct_temporal_connection(round_idx, track_grad=False)
            plan.append({kind: [list(pair) for pair in zip(*matrix.nonzero())]
                         for kind, matrix in [
                             ('spatial', graph.spatial_adj_matrix),
                             ('temporal', graph.temporal_adj_matrix)]})
    # Convert numpy integer coordinates to ordinary JSON integers.
    return [{kind: [[int(s), int(t)] for s, t in edges]
             for kind, edges in item.items()} for item in plan]
