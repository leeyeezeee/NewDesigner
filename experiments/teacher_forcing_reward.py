import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

from GDesigner.utils.ig_scorer import FinalAnswerScorer, ScoreResult, TargetSpec
from GDesigner.utils.uncertainty import edge_key


_DEFAULT_TF_EDGE_IG_REWARD_LAMBDA = 1.0


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
            "Use multi-sampled final-answer correctness to build graph-level "
            "advantages. Per-edge IG can be added with --edge_ig_reward_lambda."
        ),
    )
    parser.add_argument(
        "--use_graph_correctness_advantage",
        action="store_true",
        help=(
            "Deprecated alias for --use_graph_tf_reward."
        ),
    )
    parser.add_argument(
        "--graph_sample_count",
        type=int,
        default=5,
        help="Number of communication graphs sampled per training example.",
    )
    parser.add_argument(
        "--max_concurrent_graphs",
        type=int,
        default=10,
        help=(
            "Maximum number of realized graphs to execute concurrently per batch. "
            "Use 0 or a negative value for unlimited concurrency."
        ),
    )
    parser.add_argument(
        "--graph_softmax_temperature",
        type=float,
        default=1.0,
        help="Deprecated; graph-level softmax weighting is no longer used by --use_graph_tf_reward.",
    )
    parser.add_argument(
        "--edge_tanh_temperature",
        type=float,
        default=1.0,
        help="Temperature for tanh normalization of final-agent edge teacher-answer logprob IG gains.",
    )
    parser.add_argument(
        "--edge_ig_reward_lambda",
        type=float,
        default=None,
        help=(
            "Coefficient for the fine-grained per-edge information-gain reward. "
            "If omitted, correctness-advantage training leaves this extra term disabled."
        ),
    )
    parser.add_argument(
        "--graph_advantage_epsilon",
        type=float,
        default=1e-6,
        help="Small constant used when normalizing graph advantages.",
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


def _edge_ig_coefficient(
    log_prob: torch.Tensor,
    detail: Dict[str, Any],
    *,
    edge_tanh_temperature: float,
    edge_ig_reward_lambda: float,
) -> torch.Tensor:
    normalized_gain = torch.tanh(
        log_prob.new_tensor(float(detail["ig_gain"]) / edge_tanh_temperature)
    )
    return log_prob.new_tensor(float(edge_ig_reward_lambda)) * normalized_gain


def _correctness_advantages(
    correctness_scores: Sequence[float],
    *,
    advantage_epsilon: float,
) -> Tuple[float, List[float], float, float]:
    if not correctness_scores:
        return 0.0, [], 0.0, 0.0

    scores = [float(score) for score in correctness_scores]
    baseline = sum(scores) / len(scores)
    advantages = [score - baseline for score in scores]
    variance = sum(advantage * advantage for advantage in advantages) / len(advantages)
    std = math.sqrt(variance)
    if std > float(advantage_epsilon):
        advantages = [advantage / std for advantage in advantages]
    return baseline, advantages, variance, std


def edge_information_gain_loss(
    graph,
    edge_details: Dict[str, Dict[str, Any]],
    reference_loss: torch.Tensor,
    *,
    edge_tanh_temperature: float = 1.0,
    edge_ig_reward_lambda: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Build a fine-grained per-edge policy-gradient loss from IG gains."""
    zero = reference_loss.new_tensor(0.0)
    edge_tanh_temperature = max(float(edge_tanh_temperature), 1e-6)
    edge_ig_reward_lambda = float(edge_ig_reward_lambda)
    if edge_ig_reward_lambda == 0.0:
        return zero, {"used_edges": 0, "avg_edge_ig_coefficient": 0.0}

    terms: List[torch.Tensor] = []
    coefficients: List[float] = []
    for edge_info in getattr(graph, "edge_log_probs", []):
        log_prob = edge_info.get("log_prob")
        if not torch.is_tensor(log_prob):
            continue
        detail = edge_details.get(edge_key(edge_info), {})
        if "ig_gain" not in detail:
            continue

        coefficient = _edge_ig_coefficient(
            log_prob,
            detail,
            edge_tanh_temperature=edge_tanh_temperature,
            edge_ig_reward_lambda=edge_ig_reward_lambda,
        )
        terms.append(-(coefficient * log_prob))
        coefficients.append(float(coefficient.detach().cpu().item()))

    return (
        torch.sum(torch.stack(terms)) if terms else zero,
        {
            "used_edges": len(terms),
            "avg_edge_ig_coefficient": (
                sum(coefficients) / len(coefficients)
                if coefficients
                else 0.0
            ),
        },
    )


def graph_correctness_advantage_edge_loss(
    graph_groups: Sequence[Sequence[Any]],
    correctness_groups: Sequence[Sequence[float]],
    edge_detail_groups: Sequence[Sequence[Dict[str, Dict[str, Any]]]],
    reference_loss: torch.Tensor,
    *,
    edge_tanh_temperature: float = 1.0,
    edge_ig_reward_lambda: float = 0.0,
    advantage_epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Build edge losses from graph-level correctness advantages plus edge IG."""
    zero = reference_loss.new_tensor(0.0)
    group_losses: List[torch.Tensor] = []
    summaries: List[Dict[str, Any]] = []
    edge_tanh_temperature = max(float(edge_tanh_temperature), 1e-6)
    edge_ig_reward_lambda = float(edge_ig_reward_lambda)

    for graphs, correctness_scores, edge_details_list in zip(
        graph_groups,
        correctness_groups,
        edge_detail_groups,
    ):
        baseline, advantages, variance, std = _correctness_advantages(
            correctness_scores,
            advantage_epsilon=advantage_epsilon,
        )

        terms: List[torch.Tensor] = []
        coefficients: List[float] = []
        edge_ig_coefficients: List[float] = []
        used_edges = 0

        for graph_advantage, graph, edge_details in zip(
            advantages,
            graphs,
            edge_details_list,
        ):
            for edge_info in getattr(graph, "edge_log_probs", []):
                log_prob = edge_info.get("log_prob")
                if not torch.is_tensor(log_prob):
                    continue

                coefficient = log_prob.new_tensor(float(graph_advantage))
                detail = edge_details.get(edge_key(edge_info), {})
                if edge_ig_reward_lambda != 0.0 and "ig_gain" in detail:
                    edge_ig_coefficient = _edge_ig_coefficient(
                        log_prob,
                        detail,
                        edge_tanh_temperature=edge_tanh_temperature,
                        edge_ig_reward_lambda=edge_ig_reward_lambda,
                    )
                    coefficient = coefficient + edge_ig_coefficient
                    edge_ig_coefficients.append(
                        float(edge_ig_coefficient.detach().cpu().item())
                    )

                terms.append(-(coefficient * log_prob))
                coefficients.append(float(coefficient.detach().cpu().item()))
                used_edges += 1

        group_losses.append(torch.sum(torch.stack(terms)) if terms else zero)
        summaries.append({
            "correctness_scores": [float(score) for score in correctness_scores],
            "correctness_baseline": float(baseline),
            "correctness_variance": float(variance),
            "correctness_std": float(std),
            "standardized_graph_advantage": bool(std > float(advantage_epsilon)),
            "graph_advantages": [float(advantage) for advantage in advantages],
            "used_edges": used_edges,
            "avg_edge_coefficient": (
                sum(coefficients) / len(coefficients)
                if coefficients
                else 0.0
            ),
            "avg_edge_ig_coefficient": (
                sum(edge_ig_coefficients) / len(edge_ig_coefficients)
                if edge_ig_coefficients
                else 0.0
            ),
        })

    if not group_losses:
        return zero, summaries
    return torch.mean(torch.stack(group_losses)), summaries


def teacher_forcing_edge_loss(
    graph_groups: Sequence[Sequence[Any]],
    score_groups: Sequence[Sequence[float]],
    edge_detail_groups: Sequence[Sequence[Dict[str, Dict[str, Any]]]],
    reference_loss: torch.Tensor,
    *,
    graph_softmax_temperature: float = 1.0,
    edge_tanh_temperature: float = 1.0,
    edge_ig_reward_lambda: float = _DEFAULT_TF_EDGE_IG_REWARD_LAMBDA,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Build the edge-level policy-gradient loss from graph and edge weights."""
    zero = reference_loss.new_tensor(0.0)
    group_losses: List[torch.Tensor] = []
    summaries: List[Dict[str, Any]] = []
    edge_tanh_temperature = max(float(edge_tanh_temperature), 1e-6)
    edge_ig_reward_lambda = float(edge_ig_reward_lambda)

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
                coefficient = (
                    log_prob.new_tensor(float(graph_weight))
                    * log_prob.new_tensor(edge_ig_reward_lambda)
                    * normalized_gain
                )
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
