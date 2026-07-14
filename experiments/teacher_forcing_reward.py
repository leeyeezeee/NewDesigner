import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

from experiments.agent_backend import add_agent_backend_args
from GDesigner.utils.ig_scorer import FinalAnswerScorer, ScoreResult, TargetSpec
from GDesigner.utils.uncertainty import edge_key


_DEFAULT_TF_EDGE_IG_REWARD_LAMBDA = 1.0


def add_teacher_forcing_reward_args(parser) -> None:
    add_agent_backend_args(parser)
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
        "--edge_ig_discount_factor",
        type=float,
        default=0.0,
        help=(
            "Discount factor for propagating downstream edge IG rewards within "
            "each realized spatial DAG round. Rewards reset between rounds; 0.0 "
            "exactly preserves immediate-only IG."
        ),
    )
    parser.add_argument(
        "--graph_advantage_epsilon",
        type=float,
        default=1e-6,
        help="Small constant used when normalizing graph advantages.",
    )
    parser.add_argument(
        "--graph_sparsity_lambda",
        type=float,
        default=0.1,
        help=(
            "Correctness-gated penalty on the mean realized spatial-edge "
            "density per round. Must be in [0, 1)."
        ),
    )
async def graph_teacher_forcing_score(
    scorer: FinalAnswerScorer,
    graph,
    input_data: Dict[str, Any],
    outputs: Iterable[Any],
    target_spec: TargetSpec,
) -> ScoreResult:
    if target_spec.mode != "execution":
        history = getattr(graph.decision_node, "execution_history", [])
        if not history:
            raise RuntimeError(
                "The optimized decision node has no execution history for "
                "graph-level teacher-forcing scoring."
            )
        final_history = history[-1]
        return await scorer.teacher_answer_logprob(
            graph.decision_node,
            input_data,
            final_history.get("spatial_info", {}),
            final_history.get("temporal_info", {}),
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
    ig_gain: float,
    *,
    edge_tanh_temperature: float,
    edge_ig_reward_lambda: float,
) -> torch.Tensor:
    normalized_gain = torch.tanh(
        log_prob.new_tensor(float(ig_gain) / edge_tanh_temperature)
    )
    return log_prob.new_tensor(float(edge_ig_reward_lambda)) * normalized_gain


def _edge_ig_record(
    edge_info: Dict[str, Any],
    detail: Dict[str, Any],
    log_prob: torch.Tensor,
    coefficient: torch.Tensor,
    discounted_ig_gain: float,
) -> Dict[str, Any]:
    """Return the raw values needed to verify that edge IG reached the loss."""
    ig_gain = float(detail["ig_gain"])
    log_prob_value = float(log_prob.detach().cpu().item())
    coefficient_value = float(coefficient.detach().cpu().item())
    return {
        "edge_key": edge_key(edge_info),
        "before_score": detail.get(
            "before_teacher_logprob", detail.get("before_answer_score")
        ),
        "after_score": detail.get(
            "after_teacher_logprob", detail.get("after_answer_score")
        ),
        "ig_gain": ig_gain,
        "discounted_ig_gain": float(discounted_ig_gain),
        "log_prob": log_prob_value,
        "ig_coefficient": coefficient_value,
        "ig_loss_term": -(coefficient_value * log_prob_value),
    }


def discounted_edge_ig_gains(
    graph,
    edge_details: Dict[str, Dict[str, Any]],
    discount_factor: float,
) -> Dict[str, float]:
    """Propagate immediate IG within each realized spatial DAG round.

    Rewards reset between rounds. Temporal edges never carry downstream credit,
    because temporal topology is not part of the optimized spatial policy. At a
    spatial branch, downstream returns are averaged uniformly, preventing an
    artificial preference for high-outdegree nodes.
    """
    gamma = float(discount_factor)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(
            f"edge_ig_discount_factor must be in [0, 1], got {gamma}."
        )

    edge_infos: Dict[str, Dict[str, Any]] = {}
    immediate_gains: Dict[str, float] = {}
    for edge_info in getattr(graph, "edge_log_probs", []):
        key = edge_key(edge_info)
        detail = edge_details.get(key)
        if detail is None or "ig_gain" not in detail:
            continue
        edge_infos[key] = edge_info
        immediate_gains[key] = float(detail["ig_gain"])

    # Preserve the previous numerical path exactly when discounting is disabled.
    if gamma == 0.0:
        return immediate_gains

    outgoing_by_state: Dict[Tuple[Any, int], List[str]] = {}
    destination_state: Dict[str, Tuple[Any, int]] = {}
    for key, edge_info in edge_infos.items():
        round_idx = int(edge_info["round"])
        edge_type = edge_info["type"]
        if edge_type != "spatial":
            continue
        source_state = (edge_info["source"], round_idx)
        target_state = (edge_info["target"], round_idx)
        outgoing_by_state.setdefault(source_state, []).append(key)
        destination_state[key] = target_state

    edge_returns: Dict[str, float] = {}
    node_values: Dict[Tuple[Any, int], float] = {}
    visiting_states = set()

    def node_value(state: Tuple[Any, int]) -> float:
        if state in node_values:
            return node_values[state]
        if state in visiting_states:
            raise RuntimeError(
                "Cycle detected while propagating discounted edge IG rewards."
            )
        visiting_states.add(state)
        outgoing = outgoing_by_state.get(state, [])
        if not outgoing:
            value = 0.0
        else:
            value = sum(edge_return(key) for key in outgoing) / len(outgoing)
        visiting_states.remove(state)
        node_values[state] = value
        return value

    def edge_return(key: str) -> float:
        if key not in edge_returns:
            target_state = destination_state.get(key)
            edge_returns[key] = immediate_gains[key]
            if target_state is not None:
                edge_returns[key] += gamma * node_value(target_state)
        return edge_returns[key]

    for key in edge_infos:
        edge_return(key)
    return edge_returns


def _average_edge_ig_logprob_loss(records: Sequence[Dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return sum(float(record["ig_loss_term"]) for record in records) / len(records)


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


def correctness_gated_sparse_graph_reward(
    graph,
    correctness: float,
    sparsity_lambda: float,
) -> Tuple[float, float, float, int]:
    """Reward correct graphs for being sparse without counting repeated rounds.

    The realized edge count is averaged across rounds and temporal edges are
    excluded.  The denominator is the number of valid spatial choices in one
    round, so increasing ``num_rounds`` cannot increase the sparsity penalty.
    """
    valid_edges = int((getattr(graph, "spatial_masks") > 0).sum().item())
    mean_edges = float(getattr(graph, "mean_spatial_edges_per_round", 0.0))
    edge_density = mean_edges / valid_edges if valid_edges > 0 else 0.0
    edge_density = min(max(edge_density, 0.0), 1.0)
    correctness = float(correctness)
    reward = correctness - float(sparsity_lambda) * correctness * edge_density
    return reward, edge_density, mean_edges, valid_edges


def edge_information_gain_loss(
    graph,
    edge_details: Dict[str, Dict[str, Any]],
    reference_loss: torch.Tensor,
    *,
    edge_tanh_temperature: float = 1.0,
    edge_ig_reward_lambda: float = 0.0,
    edge_ig_discount_factor: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Build a fine-grained per-edge policy-gradient loss from IG gains."""
    zero = reference_loss.new_tensor(0.0)
    edge_tanh_temperature = max(float(edge_tanh_temperature), 1e-6)
    edge_ig_reward_lambda = float(edge_ig_reward_lambda)
    if edge_ig_reward_lambda == 0.0:
        return zero, {"used_edges": 0, "avg_edge_ig_coefficient": 0.0}
    sampled_edges = len(getattr(graph, "edge_log_probs", []))
    if sampled_edges == 0:
        return zero, {
            "used_edges": 0,
            "sampled_edges": 0,
            "missing_log_prob": 0,
            "missing_detail": 0,
            "missing_ig_gain": 0,
            "edge_ig_records": [],
            "avg_edge_ig_logprob_loss": 0.0,
            "edge_ig_discount_factor": float(edge_ig_discount_factor),
            "avg_edge_ig_coefficient": 0.0,
        }
    discounted_gains = discounted_edge_ig_gains(
        graph, edge_details, edge_ig_discount_factor
    )

    terms: List[torch.Tensor] = []
    coefficients: List[float] = []
    records: List[Dict[str, Any]] = []
    missing_log_prob = 0
    missing_detail = 0
    missing_ig_gain = 0
    for edge_info in getattr(graph, "edge_log_probs", []):
        log_prob = edge_info.get("log_prob")
        if not torch.is_tensor(log_prob):
            missing_log_prob += 1
            continue
        detail = edge_details.get(edge_key(edge_info))
        if detail is None:
            missing_detail += 1
            continue
        if "ig_gain" not in detail:
            missing_ig_gain += 1
            continue

        coefficient = _edge_ig_coefficient(
            log_prob,
            discounted_gains[edge_key(edge_info)],
            edge_tanh_temperature=edge_tanh_temperature,
            edge_ig_reward_lambda=edge_ig_reward_lambda,
        )
        terms.append(-(coefficient * log_prob))
        coefficients.append(float(coefficient.detach().cpu().item()))
        records.append(_edge_ig_record(
            edge_info,
            detail,
            log_prob,
            coefficient,
            discounted_gains[edge_key(edge_info)],
        ))

    avg_edge_ig_logprob_loss = _average_edge_ig_logprob_loss(records)
    print("average edge IG logprob loss:", avg_edge_ig_logprob_loss)

    return (
        torch.sum(torch.stack(terms)) if terms else zero,
        {
            "used_edges": len(terms),
            "sampled_edges": sampled_edges,
            "missing_log_prob": missing_log_prob,
            "missing_detail": missing_detail,
            "missing_ig_gain": missing_ig_gain,
            "edge_ig_records": records,
            "avg_edge_ig_logprob_loss": avg_edge_ig_logprob_loss,
            "edge_ig_discount_factor": float(edge_ig_discount_factor),
            "avg_edge_ig_coefficient": (
                sum(coefficients) / len(coefficients)
                if coefficients
                else 0.0
            ),
        },
    )


def graph_correctness_advantage_edge_loss(
    graph_groups: Sequence[Sequence[Any]],
    graph_log_prob_groups: Sequence[Sequence[torch.Tensor]],
    correctness_groups: Sequence[Sequence[float]],
    edge_detail_groups: Sequence[Sequence[Dict[str, Dict[str, Any]]]],
    reference_loss: torch.Tensor,
    *,
    edge_tanh_temperature: float = 1.0,
    edge_ig_reward_lambda: float = 0.0,
    edge_ig_discount_factor: float = 0.0,
    advantage_epsilon: float = 1e-6,
    graph_sparsity_lambda: float = 0.1,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Combine correctness-gated sparse graph rewards and selected-edge IG."""
    zero = reference_loss.new_tensor(0.0)
    group_losses: List[torch.Tensor] = []
    summaries: List[Dict[str, Any]] = []
    edge_tanh_temperature = max(float(edge_tanh_temperature), 1e-6)
    edge_ig_reward_lambda = float(edge_ig_reward_lambda)
    graph_sparsity_lambda = float(graph_sparsity_lambda)
    if not 0.0 <= graph_sparsity_lambda < 1.0:
        raise ValueError(
            "graph_sparsity_lambda must be in [0, 1), got "
            f"{graph_sparsity_lambda}."
        )
    if not (
        len(graph_groups)
        == len(graph_log_prob_groups)
        == len(correctness_groups)
        == len(edge_detail_groups)
    ):
        raise ValueError(
            "Graph reward batches must contain the same number of groups."
        )

    for graphs, graph_log_probs, correctness_scores, edge_details_list in zip(
        graph_groups,
        graph_log_prob_groups,
        correctness_groups,
        edge_detail_groups,
    ):
        if not (
            len(graphs)
            == len(graph_log_probs)
            == len(correctness_scores)
            == len(edge_details_list)
        ):
            raise ValueError(
                "Each graph reward group must contain the same number of graphs, "
                "full graph log-probs, correctness scores, and edge-detail maps."
            )
        graph_rewards: List[float] = []
        spatial_edge_densities: List[float] = []
        mean_spatial_edges: List[float] = []
        valid_spatial_edges: List[int] = []
        for graph, correctness in zip(graphs, correctness_scores):
            reward, density, mean_edges, valid_edges = (
                correctness_gated_sparse_graph_reward(
                    graph,
                    correctness,
                    graph_sparsity_lambda,
                )
            )
            graph_rewards.append(reward)
            spatial_edge_densities.append(density)
            mean_spatial_edges.append(mean_edges)
            valid_spatial_edges.append(valid_edges)
        baseline, advantages, variance, std = _correctness_advantages(
            graph_rewards,
            advantage_epsilon=advantage_epsilon,
        )

        per_graph_losses: List[torch.Tensor] = []
        graph_reward_loss_values: List[float] = []
        edge_ig_loss_values: List[float] = []
        edge_ig_coefficients: List[float] = []
        edge_ig_records: List[Dict[str, Any]] = []
        sampled_edges = 0
        missing_log_prob = 0
        missing_detail = 0
        missing_ig_gain = 0
        used_edges = 0

        for graph_advantage, graph, graph_log_prob, edge_details in zip(
            advantages,
            graphs,
            graph_log_probs,
            edge_details_list,
        ):
            if not torch.is_tensor(graph_log_prob):
                raise TypeError("Full graph log-prob must be a torch.Tensor.")
            graph_advantage_tensor = graph_log_prob.new_tensor(float(graph_advantage))
            graph_reward_loss = -(graph_advantage_tensor * graph_log_prob)

            discounted_gains = discounted_edge_ig_gains(
                graph, edge_details, edge_ig_discount_factor
            )
            graph_edge_ig_terms: List[torch.Tensor] = []
            for edge_info in getattr(graph, "edge_log_probs", []):
                sampled_edges += 1
                log_prob = edge_info.get("log_prob")
                if not torch.is_tensor(log_prob):
                    missing_log_prob += 1
                    continue

                detail = edge_details.get(edge_key(edge_info))
                if detail is None:
                    missing_detail += 1
                    detail = {}
                elif "ig_gain" not in detail:
                    missing_ig_gain += 1
                if edge_ig_reward_lambda != 0.0 and "ig_gain" in detail:
                    edge_ig_coefficient = _edge_ig_coefficient(
                        log_prob,
                        discounted_gains[edge_key(edge_info)],
                        edge_tanh_temperature=edge_tanh_temperature,
                        edge_ig_reward_lambda=edge_ig_reward_lambda,
                    )
                    edge_ig_coefficients.append(
                        float(edge_ig_coefficient.detach().cpu().item())
                    )
                    edge_ig_records.append(
                        _edge_ig_record(
                            edge_info,
                            detail,
                            log_prob,
                            edge_ig_coefficient,
                            discounted_gains[edge_key(edge_info)],
                        )
                    )
                    graph_edge_ig_terms.append(-(edge_ig_coefficient * log_prob))
                    used_edges += 1

            edge_ig_loss = (
                torch.sum(torch.stack(graph_edge_ig_terms))
                if graph_edge_ig_terms
                else zero
            )

            per_graph_loss = graph_reward_loss + edge_ig_loss
            per_graph_losses.append(per_graph_loss)
            graph_reward_loss_values.append(
                float(graph_reward_loss.detach().cpu().item())
            )
            edge_ig_loss_values.append(float(edge_ig_loss.detach().cpu().item()))

        avg_edge_ig_logprob_loss = _average_edge_ig_logprob_loss(edge_ig_records)
        if edge_ig_records:
            print("average edge IG logprob loss:", avg_edge_ig_logprob_loss)

        group_losses.append(
            torch.mean(torch.stack(per_graph_losses)) if per_graph_losses else zero
        )
        summaries.append({
            "correctness_scores": [float(score) for score in correctness_scores],
            "graph_rewards": graph_rewards,
            "spatial_edge_densities": spatial_edge_densities,
            "mean_spatial_edges_per_round": mean_spatial_edges,
            "valid_spatial_edges_per_round": valid_spatial_edges,
            "graph_sparsity_lambda": graph_sparsity_lambda,
            "graph_reward_baseline": float(baseline),
            "graph_reward_variance": float(variance),
            "graph_reward_std": float(std),
            "standardized_graph_advantage": bool(std > float(advantage_epsilon)),
            "graph_advantages": [float(advantage) for advantage in advantages],
            "used_edges": used_edges,
            "sampled_edges": sampled_edges,
            "missing_log_prob": missing_log_prob,
            "missing_detail": missing_detail,
            "missing_ig_gain": missing_ig_gain,
            "edge_ig_records": edge_ig_records,
            "avg_edge_ig_logprob_loss": avg_edge_ig_logprob_loss,
            "edge_ig_discount_factor": float(edge_ig_discount_factor),
            "avg_edge_ig_coefficient": (
                sum(edge_ig_coefficients) / len(edge_ig_coefficients)
                if edge_ig_coefficients
                else 0.0
            ),
            "avg_graph_reward_loss": (
                sum(graph_reward_loss_values) / len(graph_reward_loss_values)
                if graph_reward_loss_values
                else 0.0
            ),
            "avg_graph_edge_ig_loss": (
                sum(edge_ig_loss_values) / len(edge_ig_loss_values)
                if edge_ig_loss_values
                else 0.0
            ),
        })

    if not group_losses:
        return zero, summaries
    edge_counts = [
        count
        for summary in summaries
        for count in summary["mean_spatial_edges_per_round"]
    ]
    edge_densities = [
        density
        for summary in summaries
        for density in summary["spatial_edge_densities"]
    ]
    print(
        "graph loss components: "
        f"correctness_sparsity={sum(s['avg_graph_reward_loss'] for s in summaries) / len(summaries):.6f}, "
        f"edge_ig={sum(s['avg_graph_edge_ig_loss'] for s in summaries) / len(summaries):.6f}, "
        f"avg_edges={sum(edge_counts) / len(edge_counts) if edge_counts else 0.0:.2f}, "
        f"edge_density={sum(edge_densities) / len(edge_densities) if edge_densities else 0.0:.6f}"
    )
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
    edge_ig_discount_factor: float = 0.0,
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
            discounted_gains = discounted_edge_ig_gains(
                graph, edge_details, edge_ig_discount_factor
            )
            for edge_info in getattr(graph, "edge_log_probs", []):
                log_prob = edge_info.get("log_prob")
                if not torch.is_tensor(log_prob):
                    continue
                detail = edge_details.get(edge_key(edge_info), {})
                if "ig_gain" not in detail:
                    continue

                normalized_gain = torch.tanh(
                    log_prob.new_tensor(
                        discounted_gains[edge_key(edge_info)]
                        / edge_tanh_temperature
                    )
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
