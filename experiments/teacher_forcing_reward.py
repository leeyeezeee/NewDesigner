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
        default=8,
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
            "each independently sampled spatial DAG round. Propagation does "
            "not cross rounds; 0.0 keeps only each round's immediate IG."
        ),
    )
    parser.add_argument(
        "--graph_advantage_epsilon",
        type=float,
        default=1e-6,
        help="Small constant used when normalizing graph advantages.",
    )
    parser.add_argument(
        "--graph_token_cost_lambda",
        type=float,
        default=0.4,
        help=(
            "Weight of the centered OPTIMA-style graph token cost in the graph "
            "advantage. Correctness is standardized separately."
        ),
    )
    parser.add_argument(
        "--edge_ig_warmup_iterations",
        type=int,
        default=2,
        help=(
            "Number of initial training iterations that skip edge IG "
            "counterfactual calls and loss."
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
    ig_gain = float(ig_gain)
    edge_tanh_temperature = float(edge_tanh_temperature)
    edge_ig_reward_lambda = float(edge_ig_reward_lambda)
    if not all(math.isfinite(value) for value in (
        ig_gain,
        edge_tanh_temperature,
        edge_ig_reward_lambda,
    )):
        raise FloatingPointError(
            "Edge IG coefficient received a non-finite value: "
            f"ig_gain={ig_gain}, temperature={edge_tanh_temperature}, "
            f"lambda={edge_ig_reward_lambda}."
        )
    if edge_tanh_temperature <= 0.0:
        raise ValueError(
            "edge_tanh_temperature must be positive, got "
            f"{edge_tanh_temperature}."
        )
    normalized_gain = torch.tanh(
        log_prob.new_tensor(ig_gain / edge_tanh_temperature)
    )
    return log_prob.new_tensor(edge_ig_reward_lambda) * normalized_gain


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
        "type": edge_info.get("type"),
        "round": edge_info.get("round"),
        "source": edge_info.get("source"),
        "target": edge_info.get("target"),
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
        "edge_token_count": int(detail.get("edge_token_count", 0)),
        "graph_total_edge_tokens": int(
            detail.get("graph_total_edge_tokens", 0)
        ),
        "edge_token_cost": float(detail.get("edge_token_cost", 0.0)),
        "raw_ig_gain": float(detail.get("raw_ig_gain", detail["ig_gain"])),
    }


def discounted_edge_ig_gains(
    graph,
    edge_details: Dict[str, Dict[str, Any]],
    discount_factor: float,
) -> Dict[str, float]:
    """Propagate immediate IG within each independently sampled spatial round.

    Downstream propagation resets between rounds. Temporal edges never carry
    downstream credit. At a spatial branch, downstream returns are averaged
    uniformly, preventing an artificial preference for high-outdegree nodes.
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


def _standardized_advantages(
    reward_scores: Sequence[float],
    *,
    advantage_epsilon: float,
) -> Tuple[float, List[float], float, float]:
    if not reward_scores:
        return 0.0, [], 0.0, 0.0

    scores = [float(score) for score in reward_scores]
    baseline = sum(scores) / len(scores)
    advantages = [score - baseline for score in scores]
    variance = sum(advantage * advantage for advantage in advantages) / len(advantages)
    std = math.sqrt(variance)
    if std > float(advantage_epsilon):
        advantages = [advantage / std for advantage in advantages]
    return baseline, advantages, variance, std


def mean_valid_spatial_edge_probability(
    graph,
    reference_loss: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    """Return the expected one-round spatial-edge density.

    The probability matrix is generated once per task and reused by every
    communication round. Only valid spatial entries participate; sampled edge
    counts and temporal edges do not enter this differentiable regularizer.
    """
    spatial_logits = getattr(graph, "spatial_logits", None)
    spatial_masks = getattr(graph, "spatial_masks", None)
    if not torch.is_tensor(spatial_logits) or not torch.is_tensor(spatial_masks):
        return reference_loss.new_tensor(0.0), 0

    flat_logits = spatial_logits.reshape(-1)
    valid_mask = spatial_masks.reshape(-1).to(device=flat_logits.device) > 0
    if flat_logits.numel() != valid_mask.numel():
        raise ValueError(
            "Spatial logits and masks must have the same number of entries; "
            f"received {flat_logits.numel()} and {valid_mask.numel()}."
        )
    valid_edges = int(valid_mask.sum().item())
    if valid_edges == 0:
        return flat_logits.sum() * 0.0, 0
    temperature = max(
        float(getattr(graph, "spatial_sampling_temperature", 1.0)),
        1e-6,
    )
    return torch.sigmoid(flat_logits[valid_mask] / temperature).mean(), valid_edges


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


def _unique_trainable_graph_parameters(
    graph_groups: Sequence[Sequence[Any]],
) -> List[torch.nn.Parameter]:
    """Collect the shared topology parameters once for gradient diagnostics."""
    parameters: List[torch.nn.Parameter] = []
    seen: set[int] = set()
    for graphs in graph_groups:
        for graph in graphs:
            candidates: List[torch.nn.Parameter] = []
            for module_name in ("gat",):
                module = getattr(graph, module_name, None)
                if isinstance(module, torch.nn.Module):
                    candidates.extend(module.parameters())
            spatial_parameters = getattr(graph, "spatial_parameters", None)
            if callable(spatial_parameters):
                candidates.extend(spatial_parameters())
            temporal_logits = getattr(graph, "temporal_logits", None)
            if isinstance(temporal_logits, torch.nn.Parameter):
                candidates.append(temporal_logits)

            for parameter in candidates:
                if not parameter.requires_grad or id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                parameters.append(parameter)
    return parameters


def _loss_gradient_l2_norm(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
) -> float:
    """Measure a loss component without accumulating parameter gradients."""
    if not torch.is_tensor(loss) or not loss.requires_grad or not parameters:
        return 0.0
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squared_norm = loss.new_tensor(0.0, dtype=torch.float32)
    for gradient in gradients:
        if gradient is not None:
            squared_norm = squared_norm + gradient.detach().float().pow(2).sum()
    return float(torch.sqrt(squared_norm).cpu().item())


def graph_correctness_advantage_edge_loss(
    graph_groups: Sequence[Sequence[Any]],
    graph_log_prob_groups: Sequence[Sequence[torch.Tensor]],
    correctness_groups: Sequence[Sequence[float]],
    edge_detail_groups: Sequence[Sequence[Dict[str, Dict[str, Any]]]],
    reference_loss: torch.Tensor,
    *,
    graph_token_groups: Sequence[Sequence[float]] | None = None,
    graph_token_cost_lambda: float = 0.4,
    edge_tanh_temperature: float = 1.0,
    edge_ig_reward_lambda: float = 0.0,
    edge_ig_discount_factor: float = 0.0,
    advantage_epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Combine standardized correctness, centered prompt-token cost, and edge IG."""
    zero = reference_loss.new_tensor(0.0)
    group_losses: List[torch.Tensor] = []
    group_graph_reward_losses: List[torch.Tensor] = []
    group_correctness_losses: List[torch.Tensor] = []
    group_token_losses: List[torch.Tensor] = []
    group_edge_ig_losses: List[torch.Tensor] = []
    summaries: List[Dict[str, Any]] = []
    edge_tanh_temperature = max(float(edge_tanh_temperature), 1e-6)
    edge_ig_reward_lambda = float(edge_ig_reward_lambda)
    graph_token_cost_lambda = float(graph_token_cost_lambda)
    if graph_token_cost_lambda < 0.0:
        raise ValueError(
            "graph_token_cost_lambda must be non-negative, "
            f"got {graph_token_cost_lambda}."
        )
    if graph_token_groups is None:
        graph_token_groups = [
            [0.0 for _ in correctness_scores]
            for correctness_scores in correctness_groups
        ]
    if not (
        len(graph_groups)
        == len(graph_log_prob_groups)
        == len(correctness_groups)
        == len(edge_detail_groups)
        == len(graph_token_groups)
    ):
        raise ValueError(
            "Graph reward batches must contain the same number of groups."
        )

    for graphs, graph_log_probs, correctness_scores, edge_details_list, graph_tokens in zip(
        graph_groups,
        graph_log_prob_groups,
        correctness_groups,
        edge_detail_groups,
        graph_token_groups,
    ):
        if not (
            len(graphs)
            == len(graph_log_probs)
            == len(correctness_scores)
            == len(edge_details_list)
            == len(graph_tokens)
        ):
            raise ValueError(
                "Each graph reward group must contain the same number of graphs, "
                "full graph log-probs, correctness scores, token counts, and "
                "edge-detail maps."
            )
        graph_token_counts = [max(0.0, float(count)) for count in graph_tokens]
        max_graph_tokens = max(graph_token_counts, default=0.0)
        normalized_graph_token_costs = (
            [count / max_graph_tokens for count in graph_token_counts]
            if max_graph_tokens > 0.0
            else [0.0 for _ in graph_token_counts]
        )
        graph_rewards = [
            float(correctness)
            - graph_token_cost_lambda * normalized_token_cost
            for correctness, normalized_token_cost in zip(
                correctness_scores,
                normalized_graph_token_costs,
            )
        ]
        correctness_baseline, correctness_advantages, correctness_variance, correctness_std = (
            _standardized_advantages(
                correctness_scores,
                advantage_epsilon=advantage_epsilon,
            )
        )
        token_cost_baseline = (
            sum(normalized_graph_token_costs) / len(normalized_graph_token_costs)
            if normalized_graph_token_costs
            else 0.0
        )
        centered_token_costs = [
            cost - token_cost_baseline
            for cost in normalized_graph_token_costs
        ]
        token_advantages = [
            -graph_token_cost_lambda * centered_cost
            for centered_cost in centered_token_costs
        ]
        advantages = [
            correctness_advantage + token_advantage
            for correctness_advantage, token_advantage in zip(
                correctness_advantages,
                token_advantages,
            )
        ]
        graph_reward_baseline = (
            sum(graph_rewards) / len(graph_rewards) if graph_rewards else 0.0
        )
        centered_graph_rewards = [
            reward - graph_reward_baseline for reward in graph_rewards
        ]
        graph_reward_variance = (
            sum(value * value for value in centered_graph_rewards)
            / len(centered_graph_rewards)
            if centered_graph_rewards
            else 0.0
        )
        graph_reward_std = math.sqrt(graph_reward_variance)
        spatial_probability_means: List[torch.Tensor] = []
        valid_spatial_edges: List[int] = []
        mean_spatial_edges = [
            float(getattr(graph, "mean_spatial_edges_per_round", 0.0))
            for graph in graphs
        ]
        for graph in graphs:
            probability_mean, valid_edges = mean_valid_spatial_edge_probability(
                graph,
                reference_loss,
            )
            spatial_probability_means.append(probability_mean)
            valid_spatial_edges.append(valid_edges)
        per_graph_losses: List[torch.Tensor] = []
        graph_reward_losses: List[torch.Tensor] = []
        correctness_losses: List[torch.Tensor] = []
        token_losses: List[torch.Tensor] = []
        graph_edge_ig_losses: List[torch.Tensor] = []
        graph_reward_loss_values: List[float] = []
        edge_ig_loss_values: List[float] = []
        edge_ig_coefficients: List[float] = []
        edge_ig_records: List[Dict[str, Any]] = []
        sampled_edges = 0
        missing_log_prob = 0
        missing_detail = 0
        missing_ig_gain = 0
        used_edges = 0

        for (
            graph_advantage,
            correctness_advantage,
            token_advantage,
            graph,
            graph_log_prob,
            edge_details,
        ) in zip(
            advantages,
            correctness_advantages,
            token_advantages,
            graphs,
            graph_log_probs,
            edge_details_list,
        ):
            if not torch.is_tensor(graph_log_prob):
                raise TypeError("Full graph log-prob must be a torch.Tensor.")
            graph_advantage_tensor = graph_log_prob.new_tensor(float(graph_advantage))
            graph_reward_loss = -(graph_advantage_tensor * graph_log_prob)
            correctness_loss = -(
                graph_log_prob.new_tensor(float(correctness_advantage))
                * graph_log_prob
            )
            token_loss = -(
                graph_log_prob.new_tensor(float(token_advantage))
                * graph_log_prob
            )

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
            graph_reward_losses.append(graph_reward_loss)
            correctness_losses.append(correctness_loss)
            token_losses.append(token_loss)
            graph_edge_ig_losses.append(edge_ig_loss)
            graph_reward_loss_values.append(
                float(graph_reward_loss.detach().cpu().item())
            )
            edge_ig_loss_values.append(float(edge_ig_loss.detach().cpu().item()))

        avg_edge_ig_logprob_loss = _average_edge_ig_logprob_loss(edge_ig_records)
        if edge_ig_records:
            print("average edge IG logprob loss:", avg_edge_ig_logprob_loss)

        group_total_loss = (
            torch.mean(torch.stack(per_graph_losses)) if per_graph_losses else zero
        )
        group_graph_reward_losses.append(
            torch.mean(torch.stack(graph_reward_losses))
            if graph_reward_losses
            else zero
        )
        group_correctness_losses.append(
            torch.mean(torch.stack(correctness_losses))
            if correctness_losses
            else zero
        )
        group_token_losses.append(
            torch.mean(torch.stack(token_losses))
            if token_losses
            else zero
        )
        group_edge_ig_losses.append(
            torch.mean(torch.stack(graph_edge_ig_losses))
            if graph_edge_ig_losses
            else zero
        )
        group_losses.append(group_total_loss)
        summaries.append({
            "correctness_scores": [float(score) for score in correctness_scores],
            "graph_token_counts": graph_token_counts,
            "normalized_graph_token_costs": normalized_graph_token_costs,
            "max_graph_tokens": float(max_graph_tokens),
            "graph_token_cost_lambda": graph_token_cost_lambda,
            "graph_rewards": graph_rewards,
            "correctness_gate": float(correctness_baseline),
            "correctness_baseline": float(correctness_baseline),
            "correctness_variance": float(correctness_variance),
            "correctness_std": float(correctness_std),
            "standardized_correctness_advantage": bool(
                correctness_std > float(advantage_epsilon)
            ),
            "correctness_advantages": [
                float(advantage) for advantage in correctness_advantages
            ],
            "token_cost_baseline": float(token_cost_baseline),
            "centered_token_costs": [
                float(cost) for cost in centered_token_costs
            ],
            "token_advantages": [
                float(advantage) for advantage in token_advantages
            ],
            "mean_spatial_edge_probabilities": [
                float(value.detach().cpu().item())
                for value in spatial_probability_means
            ],
            "mean_spatial_edges_per_round": mean_spatial_edges,
            "valid_spatial_edges_per_round": valid_spatial_edges,
            "graph_reward_baseline": float(graph_reward_baseline),
            "graph_reward_variance": float(graph_reward_variance),
            "graph_reward_std": float(graph_reward_std),
            "standardized_graph_advantage": False,
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
    edge_probabilities = [
        probability
        for summary in summaries
        for probability in summary["mean_spatial_edge_probabilities"]
    ]
    graph_token_counts = [
        count
        for summary in summaries
        for count in summary["graph_token_counts"]
    ]
    normalized_graph_token_costs = [
        cost
        for summary in summaries
        for cost in summary["normalized_graph_token_costs"]
    ]
    graph_rewards = [
        reward
        for summary in summaries
        for reward in summary["graph_rewards"]
    ]
    token_advantages = [
        advantage
        for summary in summaries
        for advantage in summary["token_advantages"]
    ]
    graph_reward_objective = torch.mean(torch.stack(group_graph_reward_losses))
    correctness_objective = torch.mean(torch.stack(group_correctness_losses))
    token_objective = torch.mean(torch.stack(group_token_losses))
    edge_ig_objective = torch.mean(torch.stack(group_edge_ig_losses))
    trainable_parameters = _unique_trainable_graph_parameters(graph_groups)
    graph_reward_gradient_norm = _loss_gradient_l2_norm(
        graph_reward_objective,
        trainable_parameters,
    )
    correctness_gradient_norm = _loss_gradient_l2_norm(
        correctness_objective,
        trainable_parameters,
    )
    token_gradient_norm = _loss_gradient_l2_norm(
        token_objective,
        trainable_parameters,
    )
    edge_ig_gradient_norm = _loss_gradient_l2_norm(
        edge_ig_objective,
        trainable_parameters,
    )
    for summary in summaries:
        summary["batch_gradient_norms"] = {
            "graph_reward": graph_reward_gradient_norm,
            "correctness": correctness_gradient_norm,
            "token": token_gradient_norm,
            "edge_ig": edge_ig_gradient_norm,
        }
    print(
        "graph gradient norms: "
        f"graph_reward={graph_reward_gradient_norm:.6f}, "
        f"correctness={correctness_gradient_norm:.6f}, "
        f"token={token_gradient_norm:.6f}, "
        f"edge_ig={edge_ig_gradient_norm:.6f}, "
        f"avg_edges={sum(edge_counts) / len(edge_counts) if edge_counts else 0.0:.2f}, "
        f"mean_edge_probability={sum(edge_probabilities) / len(edge_probabilities) if edge_probabilities else 0.0:.6f}"
    )
    print(
        "graph token reward: "
        f"avg_prompt_tokens={sum(graph_token_counts) / len(graph_token_counts) if graph_token_counts else 0.0:.2f}, "
        f"avg_normalized_cost={sum(normalized_graph_token_costs) / len(normalized_graph_token_costs) if normalized_graph_token_costs else 0.0:.6f}, "
        f"avg_abs_token_advantage={sum(abs(value) for value in token_advantages) / len(token_advantages) if token_advantages else 0.0:.6f}, "
        f"avg_reward={sum(graph_rewards) / len(graph_rewards) if graph_rewards else 0.0:.6f}, "
        f"lambda={graph_token_cost_lambda:.6f}"
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
