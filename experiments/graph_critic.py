"""Optional lightweight graph critic for counterfactual edge rewards.

The default training path still measures edge value by executing real edge
ablations.  This module is an opt-in alternative: it learns a question-
conditioned value for an already sampled graph, then estimates an edge's value
with two cheap critic forwards, ``Q(q, A) - Q(q, A without e)``.
"""

import math
import random
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse

from GDesigner.utils.ig_scorer import FinalAnswerScorer, TargetSpec


class GraphValueCritic(torch.nn.Module):
    """G-Designer-style two-layer GCN followed by scalar graph pooling.

    Unlike the actor, the critic starts from the raw role + question features.
    Its first hidden width and 0.5 dropout follow G-Designer's original GCN;
    no actor GAT representation or parameters are shared.
    """

    architecture = "raw_role_question_gcn_768_16_16_mean_mlp_16_1_v2"

    def __init__(
        self,
        input_channels: int = 768,
        hidden_channels: int = 16,
        output_channels: int = 16,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.output_channels = int(output_channels)
        self.dropout = float(dropout)
        self.conv1 = GCNConv(self.input_channels, self.hidden_channels)
        self.conv2 = GCNConv(self.hidden_channels, self.output_channels)
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(self.output_channels, self.output_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(self.output_channels, 1),
        )

    def reset_parameters(self) -> None:
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        for module in self.value_head:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        if node_features.ndim != 2:
            raise ValueError("Critic node_features must have shape [N, D].")
        if node_features.size(1) != self.input_channels:
            raise ValueError(
                f"Critic expected {self.input_channels} input features, "
                f"received {node_features.size(1)}."
            )
        if adjacency.shape != (node_features.size(0), node_features.size(0)):
            raise ValueError(
                "Critic adjacency must have shape [N, N] matching the nodes."
            )
        adjacency = adjacency.to(
            device=node_features.device,
            dtype=node_features.dtype,
        )
        edge_index, edge_weight = dense_to_sparse(adjacency)
        hidden = F.relu(
            self.conv1(node_features, edge_index, edge_weight=edge_weight)
        )
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = self.conv2(hidden, edge_index, edge_weight=edge_weight)
        graph_embedding = hidden.mean(dim=0)
        return self.value_head(graph_embedding).squeeze(-1)


class GraphCriticReplayBuffer:
    """Bounded FIFO replay of detached graph-regression examples."""

    def __init__(self, capacity: int = 256) -> None:
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("Graph critic replay capacity must be positive.")
        self.capacity = capacity
        self._samples = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._samples)

    def add(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        target: float,
    ) -> None:
        self._samples.append({
            "node_features": node_features.detach().float().cpu(),
            "adjacency": adjacency.detach().float().cpu(),
            "target": float(target),
        })

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("Graph critic replay batch size must be positive.")
        if not self._samples:
            return []
        return random.sample(list(self._samples), min(batch_size, len(self._samples)))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "samples": list(self._samples),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        saved_capacity = int(state.get("capacity", self.capacity))
        if saved_capacity != self.capacity:
            raise ValueError(
                "Checkpoint graph critic replay capacity "
                f"{saved_capacity} does not match configured capacity {self.capacity}."
            )
        self._samples.clear()
        for sample in state.get("samples", []):
            self.add(
                sample["node_features"],
                sample["adjacency"],
                sample["target"],
            )


def build_graph_critic(
    graph,
    *,
    learning_rate: float = 0.03,
) -> Tuple[GraphValueCritic, torch.optim.Optimizer]:
    input_channels = int(graph.features.size(1) * 2)
    critic = GraphValueCritic(input_channels=input_channels)
    optimizer = torch.optim.Adam(critic.parameters(), lr=float(learning_rate))
    return critic, optimizer


def realized_spatial_adjacency(graph) -> torch.Tensor:
    """Return one graph for the query; multi-round edges become frequencies."""
    snapshots = list(getattr(graph, "realized_spatial_adjacencies", []))
    if snapshots:
        return torch.stack([
            snapshot.detach().float().cpu()
            for snapshot in snapshots
        ]).mean(dim=0)
    return torch.as_tensor(graph.spatial_adj_matrix, dtype=torch.float32)


def _critic_features(
    graph,
    task: str,
    cache: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    if cache is not None and task in cache:
        return cache[task]
    cached_task = getattr(graph, "task_conditioned_feature_task", None)
    cached_features = getattr(graph, "task_conditioned_features", None)
    if cached_task != task or not isinstance(cached_features, torch.Tensor):
        raise RuntimeError(
            "The graph critic requires the raw pre-GAT actor features for the "
            "current task. Run graph.prepare_spatial_logits(task) before "
            "critic scoring; the critic will not call the frozen sentence "
            "encoder independently."
        )
    features = cached_features.detach().float().cpu()
    if cache is not None:
        cache[task] = features
    return features


async def score_full_graph_teacher_forcing(
    graph,
    input_data: Dict[str, Any],
    *,
    target_spec: TargetSpec,
    scorer: FinalAnswerScorer,
) -> float:
    """Score the existing full run without executing a counterfactual graph."""
    if target_spec.mode == "execution":
        result = await scorer.score_outputs(
            graph.decision_node,
            input_data,
            graph.decision_node.outputs[-1:],
            target_spec,
        )
    else:
        result = await scorer.teacher_answer_logprob(
            graph.decision_node,
            input_data,
            graph.decision_node.get_spatial_info(),
            graph.decision_node.get_temporal_info(),
            target_spec,
        )
    score = float(result.score)
    if not math.isfinite(score):
        raise FloatingPointError(f"Non-finite full-graph critic target: {score}.")
    return score


def _selected_spatial_occurrences(graph) -> List[Dict[str, Any]]:
    snapshots = list(getattr(graph, "realized_spatial_adjacencies", []))
    occurrences = []
    for edge_info in getattr(graph, "edge_log_probs", []):
        if edge_info.get("type") != "spatial":
            continue
        if edge_info.get("selector_keep") is False:
            continue
        log_prob = edge_info.get("log_prob")
        if not isinstance(log_prob, torch.Tensor):
            continue
        source_idx = graph.node_id_to_index.get(edge_info.get("source"))
        target_idx = graph.node_id_to_index.get(edge_info.get("target"))
        round_idx = int(edge_info.get("round", 0))
        if source_idx is None or target_idx is None:
            continue
        if snapshots:
            if round_idx < 0 or round_idx >= len(snapshots):
                continue
            if float(snapshots[round_idx][source_idx, target_idx]) <= 0.0:
                continue
        occurrences.append(edge_info)
    return occurrences


def critic_counterfactual_edge_loss(
    critic: GraphValueCritic,
    graph_groups: Sequence[Sequence[Any]],
    input_groups: Sequence[Dict[str, Any]],
    reference_loss: torch.Tensor,
    *,
    reward_lambda: float,
    tanh_temperature: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """REINFORCE selected edges using critic-predicted deletion effects."""
    if reward_lambda == 0.0:
        return reference_loss.new_tensor(0.0), {
            "edge_count": 0.0,
            "mean_predicted_gain": 0.0,
        }
    if tanh_temperature <= 0.0:
        raise ValueError("Critic tanh temperature must be positive.")

    was_training = critic.training
    critic.eval()
    feature_cache: Dict[str, torch.Tensor] = {}
    coefficient_groups: List[
        List[List[Tuple[torch.Tensor, torch.Tensor]]]
    ] = []
    gains: List[float] = []
    with torch.no_grad():
        for graph_group, input_data in zip(graph_groups, input_groups):
            task = str(input_data["task"])
            features = _critic_features(graph_group[0], task, feature_cache)
            graph_coefficients = []
            for graph in graph_group:
                coefficients_and_log_probs = []
                adjacency = realized_spatial_adjacency(graph)
                full_value = critic(features, adjacency)
                round_count = max(
                    1, len(getattr(graph, "realized_spatial_adjacencies", []))
                )
                for edge_info in _selected_spatial_occurrences(graph):
                    source_idx = graph.node_id_to_index[edge_info["source"]]
                    target_idx = graph.node_id_to_index[edge_info["target"]]
                    counterfactual = adjacency.clone()
                    counterfactual[source_idx, target_idx] = torch.clamp(
                        counterfactual[source_idx, target_idx] - 1.0 / round_count,
                        min=0.0,
                    )
                    gain = full_value - critic(features, counterfactual)
                    normalized_gain = torch.tanh(gain / tanh_temperature)
                    gains.append(float(gain.item()))
                    coefficients_and_log_probs.append(
                        (
                            normalized_gain.detach().to(
                                edge_info["log_prob"].device
                            ),
                            edge_info["log_prob"],
                        )
                    )
                graph_coefficients.append(coefficients_and_log_probs)
            coefficient_groups.append(graph_coefficients)
    if was_training:
        critic.train()
    edge_count = sum(
        len(graph_coefficients)
        for group in coefficient_groups
        for graph_coefficients in group
    )
    if edge_count == 0:
        return reference_loss.new_tensor(0.0), {
            "edge_count": 0.0,
            "mean_predicted_gain": 0.0,
        }
    zero = reference_loss.new_tensor(0.0)
    group_losses = []
    for group in coefficient_groups:
        graph_losses = []
        for graph_coefficients in group:
            terms = [
                -float(reward_lambda) * coefficient * log_prob
                for coefficient, log_prob in graph_coefficients
            ]
            graph_losses.append(torch.stack(terms).sum() if terms else zero)
        if graph_losses:
            group_losses.append(torch.stack(graph_losses).mean())
    loss = torch.stack(group_losses).mean() if group_losses else zero
    return loss, {
        "edge_count": float(edge_count),
        "mean_predicted_gain": float(sum(gains) / len(gains)),
    }


def add_graph_critic_examples(
    replay_buffer: GraphCriticReplayBuffer,
    graph_groups: Sequence[Sequence[Any]],
    input_groups: Sequence[Dict[str, Any]],
    score_groups: Sequence[Sequence[float]],
) -> int:
    """Center each question's scores, then append detached samples to replay."""
    feature_cache: Dict[str, torch.Tensor] = {}
    added = 0
    for graph_group, input_data, raw_scores in zip(
        graph_groups,
        input_groups,
        score_groups,
    ):
        if len(graph_group) != len(raw_scores):
            raise ValueError("Critic graph and target group lengths do not match.")
        task = str(input_data["task"])
        features = _critic_features(graph_group[0], task, feature_cache)
        score_tensor = features.new_tensor(list(raw_scores))
        centered_targets = score_tensor - score_tensor.mean()
        for graph, target in zip(graph_group, centered_targets):
            replay_buffer.add(
                features,
                realized_spatial_adjacency(graph),
                float(target.item()),
            )
            added += 1
    return added


def train_graph_critic(
    critic: GraphValueCritic,
    optimizer: torch.optim.Optimizer,
    replay_buffer: GraphCriticReplayBuffer,
    *,
    batch_size: int = 32,
    updates: int = 4,
) -> Dict[str, float]:
    """Fit centered graph TF values with repeated replay-sampled MSE steps."""
    updates = int(updates)
    if updates <= 0:
        raise ValueError("Graph critic updates per iteration must be positive.")
    if len(replay_buffer) == 0:
        return {
            "loss": 0.0,
            "target_std": 0.0,
            "num_graphs": 0.0,
            "buffer_size": 0.0,
            "updates": 0.0,
        }

    critic.train()
    losses: List[float] = []
    sampled_targets: List[torch.Tensor] = []
    sampled_graphs = 0
    for _ in range(updates):
        batch = replay_buffer.sample(batch_size)
        predictions = [
            critic(sample["node_features"], sample["adjacency"])
            for sample in batch
        ]
        prediction_tensor = torch.stack(predictions)
        target_tensor = prediction_tensor.new_tensor(
            [float(sample["target"]) for sample in batch]
        )
        loss = F.mse_loss(prediction_tensor, target_tensor)
        if not torch.isfinite(loss):
            raise FloatingPointError("Graph critic MSE loss is non-finite.")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            critic.parameters(), max_norm=1.0, error_if_nonfinite=True
        )
        optimizer.step()
        losses.append(float(loss.detach().item()))
        sampled_targets.append(target_tensor.detach().cpu())
        sampled_graphs += len(batch)

    all_targets = torch.cat(sampled_targets)
    return {
        "loss": sum(losses) / len(losses),
        "target_std": float(all_targets.std(unbiased=False).item()),
        "num_graphs": float(sampled_graphs),
        "buffer_size": float(len(replay_buffer)),
        "updates": float(updates),
    }
