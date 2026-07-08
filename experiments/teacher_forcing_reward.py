from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

from GDesigner.utils.ig_scorer import FinalAnswerScorer, ScoreResult, TargetSpec
from GDesigner.utils.uncertainty import edge_key


def _graph_output_info(graph) -> Dict[str, Dict[str, Any]]:
    output_info: Dict[str, Dict[str, Any]] = {}
    for node_id, node in graph.nodes.items():
        node_outputs = getattr(node, "outputs", [])
        if isinstance(node_outputs, list):
            if not node_outputs:
                continue
            node_output = node_outputs[-1]
        else:
            node_output = node_outputs
        output_info[node_id] = {
            "role": getattr(node, "role", ""),
            "output": node_output,
        }
    return output_info


def add_teacher_forcing_reward_args(parser) -> None:
    parser.add_argument(
        "--use_graph_tf_reward",
        action="store_true",
        help=(
            "Use multi-sampled graph teacher-forcing scores as graph-level "
            "weights and final-agent teacher-answer logprob IG gains as per-edge update weights."
        ),
    )
    parser.add_argument(
        "--graph_sample_count",
        type=int,
        default=5,
        help="Number of communication graphs sampled per training example.",
    )
    parser.add_argument(
        "--graph_softmax_temperature",
        type=float,
        default=1.0,
        help="Temperature for softmax over graph-level teacher-forcing scores.",
    )
    parser.add_argument(
        "--edge_tanh_temperature",
        type=float,
        default=1.0,
        help="Temperature for tanh normalization of final-agent edge teacher-answer logprob IG gains.",
    )


async def graph_teacher_forcing_score(
    scorer: FinalAnswerScorer,
    graph,
    input_data: Dict[str, Any],
    outputs: Iterable[Any],
    target_spec: TargetSpec,
) -> ScoreResult:
    if target_spec.mode != "execution":
        return await scorer.teacher_answer_logprob(
            graph.decision_node,
            input_data,
            _graph_output_info(graph),
            {},
            target_spec,
        )

    return await scorer.score_outputs(
        graph.decision_node,
        input_data,
        outputs,
        target_spec,
        cluster_labels=None,
    )


def graph_softmax_weights(scores: Sequence[float], temperature: float) -> List[float]:
    if not scores:
        return []
    temperature = max(float(temperature), 1e-6)
    score_tensor = torch.tensor(scores, dtype=torch.float32)
    return torch.softmax(score_tensor / temperature, dim=0).tolist()


def teacher_forcing_edge_loss(
    graph_groups: Sequence[Sequence[Any]],
    score_groups: Sequence[Sequence[float]],
    edge_detail_groups: Sequence[Sequence[Dict[str, Dict[str, Any]]]],
    reference_loss: torch.Tensor,
    *,
    graph_softmax_temperature: float = 1.0,
    edge_tanh_temperature: float = 1.0,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Build the edge-level policy-gradient loss from graph and edge weights."""
    zero = reference_loss.new_tensor(0.0)
    group_losses: List[torch.Tensor] = []
    summaries: List[Dict[str, Any]] = []
    edge_tanh_temperature = max(float(edge_tanh_temperature), 1e-6)

    for graphs, scores, edge_details_list in zip(
        graph_groups,
        score_groups,
        edge_detail_groups,
    ):
        weights = graph_softmax_weights(scores, graph_softmax_temperature)
        terms: List[torch.Tensor] = []
        weighted_edge_gains: List[float] = []
        used_edges = 0

        for graph_weight, graph, edge_details in zip(weights, graphs, edge_details_list):
            for edge_info in getattr(graph, "edge_log_probs", []):
                log_prob = edge_info.get("log_prob")
                if not torch.is_tensor(log_prob):
                    continue
                detail = edge_details.get(edge_key(edge_info), {})
                if "ig_gain" not in detail:
                    continue

                normalized_gain = torch.tanh(
                    log_prob.new_tensor(float(detail["ig_gain"]) / edge_tanh_temperature)
                )
                coefficient = log_prob.new_tensor(float(graph_weight)) * normalized_gain
                terms.append(-(coefficient * log_prob))
                weighted_edge_gains.append(float(coefficient.detach().cpu().item()))
                used_edges += 1

        group_losses.append(torch.sum(torch.stack(terms)) if terms else zero)
        summaries.append({
            "graph_scores": [float(score) for score in scores],
            "graph_weights": [float(weight) for weight in weights],
            "used_edges": used_edges,
            "avg_weighted_edge_gain": (
                sum(weighted_edge_gains) / len(weighted_edge_gains)
                if weighted_edge_gains
                else 0.0
            ),
        })

    if not group_losses:
        return zero, summaries
    return torch.mean(torch.stack(group_losses)), summaries
