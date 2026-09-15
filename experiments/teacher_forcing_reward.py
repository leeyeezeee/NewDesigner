import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from experiments.agent_backend import add_agent_backend_args
from GDesigner.utils.ig_scorer import (
    FinalAnswerScorer,
    TargetSpec,
    edge_key,
)


def add_teacher_forcing_reward_args(parser) -> None:
    add_agent_backend_args(parser)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by Python, NumPy, and PyTorch.",
    )
    parser.add_argument(
        "--use_graph_tf_reward",
        action="store_true",
        help=(
            "Use mean-centered multi-sampled graph utilities. Per-edge "
            "teacher-forcing IG is enabled with --edge_ig_reward_lambda."
        ),
    )
    parser.add_argument(
        "--graph_sample_count",
        type=int,
        default=8,
        help="Number of communication graphs sampled per training example.",
    )
    parser.add_argument(
        "--prompt_token_cost_beta",
        type=float,
        default=0.5,
        help=(
            "Graph utility is correctness minus beta * rollout prompt tokens / "
            "the within-question maximum. Positive beta enables grouped sampling "
            "and requires graph_sample_count >= 2; 0 disables the token penalty."
        ),
    )
    parser.add_argument(
        "--max_concurrent_graphs",
        type=int,
        default=10,
        help="Maximum concurrently executed graph samples; non-positive is unlimited.",
    )
    parser.add_argument(
        "--edge_tanh_temperature",
        type=float,
        default=1.0,
        help="Temperature for tanh normalization of edge teacher-forcing IG.",
    )
    parser.add_argument(
        "--edge_ig_reward_lambda",
        type=float,
        default=None,
        help=(
            "Coefficient for the real edge-deletion teacher-forcing "
            "information-gain reward."
        ),
    )
    parser.add_argument(
        "--edge_ig_discount_factor",
        type=float,
        default=0.2,
        help="Within-round downstream IG discount (default: 0.2; 0 keeps immediate IG only).",
    )
    parser.add_argument(
        "--graph_advantage_epsilon",
        type=float,
        default=1e-6,
        help="Small constant for the optional full-graph TF advantage standardization.",
    )
    parser.add_argument(
        "--edge_ig_warmup_iterations",
        type=int,
        default=2,
        help=(
            "Initial iterations that skip real counterfactual edge-ablation "
            "calls and their edge-IG loss."
        ),
    )
    parser.add_argument(
        "--refine_rank",
        type=int,
        default=4,
        help="Rank of the G-Designer Z W Z^T refinement decoder.",
    )
    parser.add_argument(
        "--anchor_reg_weight",
        type=float,
        default=0.0,
        help="Weight of the G-Designer sketch/anchor Frobenius penalty.",
    )
    parser.add_argument(
        "--sparsity_reg_weight",
        type=float,
        default=0.0,
        help="Weight of the nuclear-norm penalty on refinement matrix W.",
    )


def add_full_graph_reward_ablation_args(parser) -> None:
    """Add the AQuA/GSM8K full-graph TF ablation options."""
    parser.add_argument(
        "--full_graph_tf_reward_lambda",
        type=float,
        default=0.0,
        help=(
            "Weight for the within-question full-graph teacher-forcing "
            "advantage. This assigns one graph reward through the summed "
            "graph log-prob without deleting individual edges."
        ),
    )
    parser.add_argument(
        "--disable_run_logs",
        action="store_true",
        help=(
            "Do not create timestamped training-step or evaluation-case logs. "
            "The aggregate metrics_file is still appended."
        ),
    )


def set_experiment_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_graph_reward_sampling(
    use_graph_tf_reward: bool,
    full_graph_tf_reward_lambda: float,
    prompt_token_cost_beta: float,
    graph_sample_count: int,
) -> bool:
    beta = float(prompt_token_cost_beta)
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("prompt_token_cost_beta must be finite and non-negative.")
    if beta > 0.0 and int(graph_sample_count) < 2:
        raise ValueError("--prompt_token_cost_beta > 0 requires --graph_sample_count >= 2.")
    if full_graph_tf_reward_lambda != 0.0 and int(graph_sample_count) < 2:
        raise ValueError("--full_graph_tf_reward_lambda requires --graph_sample_count >= 2.")
    return bool(use_graph_tf_reward or full_graph_tf_reward_lambda != 0.0 or beta > 0.0)


async def score_full_graph_teacher_forcing(
    graph,
    input_data: Dict[str, Any],
    *,
    target_spec: TargetSpec,
    scorer: FinalAnswerScorer,
) -> float:
    """Score the realized full multi-agent graph against the target."""
    if target_spec.mode == "execution":
        result = await scorer.score_outputs(
            graph.decision_node,
            input_data,
            graph.decision_node.outputs,
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
        raise FloatingPointError(
            f"Non-finite full-graph teacher-forcing score: {score}."
        )
    return score


def experiment_summary_metadata(args: Any, dataset: str) -> Dict[str, Any]:
    """Describe the reward actually enabled by the parsed runtime arguments."""
    edge_lambda_value = getattr(args, "edge_ig_reward_lambda", None)
    edge_lambda = 0.0 if edge_lambda_value is None else float(edge_lambda_value)
    full_graph_tf_lambda = float(
        getattr(args, "full_graph_tf_reward_lambda", 0.0)
    )
    token_beta = float(getattr(args, "prompt_token_cost_beta", 0.5))
    use_group_advantage = resolve_graph_reward_sampling(
        bool(getattr(args, "use_graph_tf_reward", False)),
        full_graph_tf_lambda,
        token_beta,
        int(getattr(args, "graph_sample_count", 8)),
    )
    edge_rewards = []
    if edge_lambda != 0.0:
        edge_rewards.append(
            "execution_score_diff"
            if str(dataset).lower() == "humaneval"
            else "teacher_logprob_diff"
        )
    optimized = bool(getattr(args, "optimized_spatial", False)) or bool(
        getattr(args, "optimized_temporal", False)
    )
    if not optimized:
        method = f"fixed_{getattr(args, 'mode', 'unknown')}"
    elif edge_rewards:
        method = "optimized_graph_edge_ig"
    elif full_graph_tf_lambda != 0.0:
        method = "optimized_graph_full_tf"
    else:
        method = "optimized_graph"
    return {
        "method": method,
        "seed": int(getattr(args, "seed", 42)),
        "reward": {
            "graph": (
                "centered_correctness_minus_prompt_token_cost"
                if use_group_advantage
                else "binary_correctness"
            ),
            "prompt_token_cost_beta": token_beta,
            "prompt_token_cost_normalization": "within_question_max",
            "prompt_token_cost_scope": "regular_rollout_including_final_decision",
            "edge": edge_rewards or ["none"],
            "lambda": edge_lambda,
            "full_graph_tf": (
                "standardized_teacher_logprob"
                if full_graph_tf_lambda != 0.0
                else "none"
            ),
            "full_graph_tf_lambda": full_graph_tf_lambda,
            "discount_factor": float(
                getattr(args, "edge_ig_discount_factor", 0.2)
            ),
        },
    }


def _edge_ig_coefficient(
    log_prob: torch.Tensor,
    ig_gain: float,
    *,
    edge_tanh_temperature: float,
    edge_ig_reward_lambda: float,
) -> torch.Tensor:
    values = (
        float(ig_gain),
        float(edge_tanh_temperature),
        float(edge_ig_reward_lambda),
    )
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError(f"Non-finite edge IG coefficient inputs: {values}.")
    if edge_tanh_temperature <= 0.0:
        raise ValueError("edge_tanh_temperature must be positive.")
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
        "ig_gain": float(detail["ig_gain"]),
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
    """Propagate edge IG inside each sampled spatial DAG round."""
    gamma = float(discount_factor)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"edge_ig_discount_factor must be in [0, 1], got {gamma}.")

    edge_infos: Dict[str, Dict[str, Any]] = {}
    immediate_gains: Dict[str, float] = {}
    for edge_info in getattr(graph, "edge_log_probs", []):
        key = edge_key(edge_info)
        detail = edge_details.get(key)
        if detail is None or "ig_gain" not in detail:
            continue
        edge_infos[key] = edge_info
        immediate_gains[key] = float(detail["ig_gain"])
    if gamma == 0.0:
        return immediate_gains

    outgoing_by_state: Dict[Tuple[Any, int], List[str]] = {}
    destination_state: Dict[str, Tuple[Any, int]] = {}
    for key, edge_info in edge_infos.items():
        if edge_info["type"] != "spatial":
            continue
        round_idx = int(edge_info["round"])
        outgoing_by_state.setdefault(
            (edge_info["source"], round_idx), []
        ).append(key)
        destination_state[key] = (edge_info["target"], round_idx)

    edge_returns: Dict[str, float] = {}
    node_values: Dict[Tuple[Any, int], float] = {}
    visiting_states = set()

    def node_value(state: Tuple[Any, int]) -> float:
        if state in node_values:
            return node_values[state]
        if state in visiting_states:
            raise RuntimeError("Cycle detected while propagating edge IG.")
        visiting_states.add(state)
        outgoing = outgoing_by_state.get(state, [])
        value = (
            sum(edge_return(key) for key in outgoing) / len(outgoing)
            if outgoing
            else 0.0
        )
        visiting_states.remove(state)
        node_values[state] = value
        return value

    def edge_return(key: str) -> float:
        if key not in edge_returns:
            value = immediate_gains[key]
            target_state = destination_state.get(key)
            if target_state is not None:
                value += gamma * node_value(target_state)
            edge_returns[key] = value
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
    baseline, advantages, variance, std = _centered_advantages(reward_scores)
    if std > float(advantage_epsilon):
        advantages = [value / std for value in advantages]
    return baseline, advantages, variance, std


def _centered_advantages(
    reward_scores: Sequence[float],
) -> Tuple[float, List[float], float, float]:
    if not reward_scores:
        return 0.0, [], 0.0, 0.0
    scores = [float(score) for score in reward_scores]
    if not all(math.isfinite(score) for score in scores):
        raise FloatingPointError("Non-finite graph reward.")
    baseline = sum(scores) / len(scores)
    advantages = [score - baseline for score in scores]
    variance = sum(value * value for value in advantages) / len(advantages)
    std = math.sqrt(variance)
    return baseline, advantages, variance, std


def token_aware_graph_utilities(
    correctness_scores: Sequence[float],
    prompt_tokens: Sequence[float],
    beta: float,
) -> Tuple[List[float], List[float]]:
    """Return normalized prompt costs and utilities, without std rescaling."""
    beta = float(beta)
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("prompt_token_cost_beta must be finite and non-negative.")
    if len(correctness_scores) != len(prompt_tokens):
        raise ValueError("Prompt-token counts must align with graph correctness scores.")
    tokens = [float(value) for value in prompt_tokens]
    if not all(math.isfinite(value) and value >= 0.0 for value in tokens):
        raise ValueError("Rollout prompt-token counts must be finite and non-negative.")
    maximum = max(tokens, default=0.0)
    costs = [value / maximum if maximum > 0.0 else 0.0 for value in tokens]
    utilities = [float(score) - beta * cost for score, cost in zip(correctness_scores, costs)]
    if not all(math.isfinite(value) for value in utilities):
        raise FloatingPointError("Non-finite token-aware graph utility.")
    return costs, utilities


def edge_information_gain_loss(
    graph,
    edge_details: Dict[str, Dict[str, Any]],
    reference_loss: torch.Tensor,
    *,
    edge_tanh_temperature: float = 1.0,
    edge_ig_reward_lambda: float = 0.0,
    edge_ig_discount_factor: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Build the selected-edge policy-gradient loss from teacher-forcing IG."""
    zero = reference_loss.new_tensor(0.0)
    edge_ig_reward_lambda = float(edge_ig_reward_lambda)
    if edge_ig_reward_lambda == 0.0:
        return zero, {"used_edges": 0, "avg_edge_ig_coefficient": 0.0}

    edge_infos = list(getattr(graph, "edge_log_probs", []))
    discounted_gains = discounted_edge_ig_gains(
        graph, edge_details, edge_ig_discount_factor
    )
    terms: List[torch.Tensor] = []
    coefficients: List[float] = []
    records: List[Dict[str, Any]] = []
    missing_log_prob = 0
    missing_detail = 0
    missing_ig_gain = 0
    for edge_info in edge_infos:
        log_prob = edge_info.get("log_prob")
        if not torch.is_tensor(log_prob):
            missing_log_prob += 1
            continue
        key = edge_key(edge_info)
        detail = edge_details.get(key)
        if detail is None:
            missing_detail += 1
            continue
        if "ig_gain" not in detail:
            missing_ig_gain += 1
            continue
        coefficient = _edge_ig_coefficient(
            log_prob,
            discounted_gains[key],
            edge_tanh_temperature=edge_tanh_temperature,
            edge_ig_reward_lambda=edge_ig_reward_lambda,
        )
        terms.append(-(coefficient * log_prob))
        coefficients.append(float(coefficient.detach().cpu().item()))
        records.append(_edge_ig_record(
            edge_info, detail, log_prob, coefficient, discounted_gains[key]
        ))

    average_loss = _average_edge_ig_logprob_loss(records)
    return (
        torch.sum(torch.stack(terms)) if terms else zero,
        {
            "used_edges": len(terms),
            "sampled_edges": len(edge_infos),
            "missing_log_prob": missing_log_prob,
            "missing_detail": missing_detail,
            "missing_ig_gain": missing_ig_gain,
            "edge_ig_records": records,
            "avg_edge_ig_logprob_loss": average_loss,
            "edge_ig_discount_factor": float(edge_ig_discount_factor),
            "avg_edge_ig_coefficient": (
                sum(coefficients) / len(coefficients) if coefficients else 0.0
            ),
        },
    )


def _unique_trainable_graph_parameters(
    graph_groups: Sequence[Sequence[Any]],
) -> List[torch.nn.Parameter]:
    parameters: List[torch.nn.Parameter] = []
    seen: set[int] = set()
    for graphs in graph_groups:
        for graph in graphs:
            candidates = list(graph.spatial_parameters())
            temporal_logits = getattr(graph, "temporal_logits", None)
            if isinstance(temporal_logits, torch.nn.Parameter):
                candidates.append(temporal_logits)
            for parameter in candidates:
                if parameter.requires_grad and id(parameter) not in seen:
                    seen.add(id(parameter))
                    parameters.append(parameter)
    return parameters


def _loss_gradient_l2_norm(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
) -> float:
    if not loss.requires_grad or not parameters:
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
    edge_tanh_temperature: float = 1.0,
    edge_ig_reward_lambda: float = 0.0,
    edge_ig_discount_factor: float = 0.2,
    advantage_epsilon: float = 1e-6,
    graph_tf_score_groups: Optional[Sequence[Sequence[float]]] = None,
    full_graph_tf_reward_lambda: float = 0.0,
    prompt_token_cost_beta: float = 0.5,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Combine centered token-aware utility with unchanged graph TF and edge IG."""
    prompt_token_cost_beta = float(prompt_token_cost_beta)
    if not math.isfinite(prompt_token_cost_beta) or prompt_token_cost_beta < 0.0:
        raise ValueError("prompt_token_cost_beta must be finite and non-negative.")
    if not (
        len(graph_groups)
        == len(graph_log_prob_groups)
        == len(correctness_groups)
        == len(edge_detail_groups)
    ):
        raise ValueError("Graph reward batches must contain equal group counts.")
    if graph_tf_score_groups is not None and (
        len(graph_tf_score_groups) != len(graph_groups)
    ):
        raise ValueError(
            "Full-graph TF reward batches must match graph reward groups."
        )
    full_graph_tf_reward_lambda = float(full_graph_tf_reward_lambda)
    if full_graph_tf_reward_lambda != 0.0 and graph_tf_score_groups is None:
        raise ValueError(
            "Full-graph TF scores are required when their reward lambda is non-zero."
        )

    zero = reference_loss.new_tensor(0.0)
    group_losses: List[torch.Tensor] = []
    group_correctness_losses: List[torch.Tensor] = []
    group_token_cost_losses: List[torch.Tensor] = []
    group_graph_tf_losses: List[torch.Tensor] = []
    group_edge_ig_losses: List[torch.Tensor] = []
    summaries: List[Dict[str, Any]] = []
    resolved_tf_score_groups = (
        graph_tf_score_groups
        if graph_tf_score_groups is not None
        else [() for _ in graph_groups]
    )
    for graphs, graph_log_probs, scores, edge_details_list, graph_tf_scores in zip(
        graph_groups,
        graph_log_prob_groups,
        correctness_groups,
        edge_detail_groups,
        resolved_tf_score_groups,
    ):
        if not (
            len(graphs)
            == len(graph_log_probs)
            == len(scores)
            == len(edge_details_list)
        ):
            raise ValueError("Each graph reward group must have aligned samples.")
        if graph_tf_scores and len(graph_tf_scores) != len(graphs):
            raise ValueError(
                "Each full-graph TF score group must align with graph samples."
            )

        if prompt_token_cost_beta > 0.0 and len(graphs) < 2:
            raise ValueError("Token-aware graph advantages require at least two samples.")
        prompt_tokens = [getattr(graph, "rollout_prompt_tokens", None) for graph in graphs]
        if prompt_token_cost_beta > 0.0 and any(value is None for value in prompt_tokens):
            raise ValueError("Missing per-graph rollout prompt-token accounting.")
        normalized_costs, utility_scores = token_aware_graph_utilities(
            scores,
            [0.0 if value is None else value for value in prompt_tokens],
            prompt_token_cost_beta,
        )
        baseline, advantages, variance, std = _centered_advantages(scores)
        cost_baseline, cost_advantages, _, _ = _centered_advantages(normalized_costs)
        utility_baseline, utility_advantages, utility_variance, utility_std = (
            _centered_advantages(utility_scores)
        )
        if graph_tf_scores:
            tf_baseline, tf_advantages, tf_variance, tf_std = (
                _standardized_advantages(
                    graph_tf_scores,
                    advantage_epsilon=advantage_epsilon,
                )
            )
        else:
            tf_baseline, tf_variance, tf_std = (0.0, 0.0, 0.0)
            tf_advantages = [0.0] * len(graphs)
        per_graph_losses = []
        correctness_losses = []
        token_cost_losses = []
        graph_tf_losses = []
        edge_ig_losses = []
        edge_ig_records = []
        edge_ig_coefficients = []
        sampled_edges = 0
        missing_log_prob = 0
        missing_detail = 0
        missing_ig_gain = 0
        for advantage, cost_advantage, tf_advantage, graph, graph_log_prob, edge_details in zip(
            advantages,
            cost_advantages,
            tf_advantages,
            graphs,
            graph_log_probs,
            edge_details_list,
        ):
            if not torch.is_tensor(graph_log_prob):
                raise TypeError("Full graph log-prob must be a torch.Tensor.")
            correctness_loss = -(
                graph_log_prob.new_tensor(float(advantage)) * graph_log_prob
            )
            # Together these are -(u_k - mean(u)) * log pi(G_k). Keep
            # components separate only to measure their gradient magnitudes.
            token_cost_loss = (
                graph_log_prob.new_tensor(prompt_token_cost_beta * float(cost_advantage))
                * graph_log_prob
            )
            graph_tf_loss = -(
                graph_log_prob.new_tensor(
                    full_graph_tf_reward_lambda * float(tf_advantage)
                )
                * graph_log_prob
            )
            ig_loss, ig_summary = edge_information_gain_loss(
                graph,
                edge_details,
                graph_log_prob,
                edge_tanh_temperature=edge_tanh_temperature,
                edge_ig_reward_lambda=edge_ig_reward_lambda,
                edge_ig_discount_factor=edge_ig_discount_factor,
            )
            per_graph_losses.append(correctness_loss + token_cost_loss + graph_tf_loss + ig_loss)
            correctness_losses.append(correctness_loss)
            token_cost_losses.append(token_cost_loss)
            graph_tf_losses.append(graph_tf_loss)
            edge_ig_losses.append(ig_loss)
            edge_ig_records.extend(ig_summary.get("edge_ig_records", []))
            if ig_summary.get("used_edges", 0):
                edge_ig_coefficients.append(
                    float(ig_summary.get("avg_edge_ig_coefficient", 0.0))
                )
            sampled_edges += int(ig_summary.get("sampled_edges", 0))
            missing_log_prob += int(ig_summary.get("missing_log_prob", 0))
            missing_detail += int(ig_summary.get("missing_detail", 0))
            missing_ig_gain += int(ig_summary.get("missing_ig_gain", 0))

        group_losses.append(
            torch.mean(torch.stack(per_graph_losses)) if per_graph_losses else zero
        )
        group_correctness_losses.append(
            torch.mean(torch.stack(correctness_losses)) if correctness_losses else zero
        )
        group_token_cost_losses.append(
            torch.mean(torch.stack(token_cost_losses)) if token_cost_losses else zero
        )
        group_graph_tf_losses.append(
            torch.mean(torch.stack(graph_tf_losses)) if graph_tf_losses else zero
        )
        group_edge_ig_losses.append(
            torch.mean(torch.stack(edge_ig_losses)) if edge_ig_losses else zero
        )
        summaries.append({
            "correctness_scores": [float(score) for score in scores],
            "correctness_baseline": float(baseline),
            "correctness_variance": float(variance),
            "correctness_std": float(std),
            "correctness_advantages": [float(value) for value in advantages],
            "rollout_prompt_tokens": prompt_tokens,
            "normalized_prompt_token_costs": normalized_costs,
            "prompt_token_cost_beta": prompt_token_cost_beta,
            "prompt_token_cost_baseline": cost_baseline,
            "token_aware_utilities": utility_scores,
            "token_aware_baseline": utility_baseline,
            "token_aware_variance": utility_variance,
            "token_aware_std": utility_std,
            "token_aware_advantages": utility_advantages,
            "graph_advantage_normalization": "mean_center_only",
            "full_graph_tf_scores": [
                float(value) for value in graph_tf_scores
            ],
            "full_graph_tf_baseline": float(tf_baseline),
            "full_graph_tf_variance": float(tf_variance),
            "full_graph_tf_std": float(tf_std),
            "full_graph_tf_advantages": [
                float(value) for value in tf_advantages
            ],
            "mean_spatial_edges_per_round": [
                float(getattr(graph, "mean_spatial_edges_per_round", 0.0))
                for graph in graphs
            ],
            "used_edges": len(edge_ig_records),
            "sampled_edges": sampled_edges,
            "missing_log_prob": missing_log_prob,
            "missing_detail": missing_detail,
            "missing_ig_gain": missing_ig_gain,
            "edge_ig_records": edge_ig_records,
            "avg_edge_ig_logprob_loss": _average_edge_ig_logprob_loss(
                edge_ig_records
            ),
            "edge_ig_discount_factor": float(edge_ig_discount_factor),
            "avg_edge_ig_coefficient": (
                sum(edge_ig_coefficients) / len(edge_ig_coefficients)
                if edge_ig_coefficients
                else 0.0
            ),
        })

    if not group_losses:
        return zero, summaries
    parameters = _unique_trainable_graph_parameters(graph_groups)
    correctness_norm = _loss_gradient_l2_norm(
        torch.mean(torch.stack(group_correctness_losses)), parameters
    )
    token_cost_norm = _loss_gradient_l2_norm(
        torch.mean(torch.stack(group_token_cost_losses)), parameters
    )
    graph_tf_norm = _loss_gradient_l2_norm(
        torch.mean(torch.stack(group_graph_tf_losses)), parameters
    )
    edge_ig_norm = _loss_gradient_l2_norm(
        torch.mean(torch.stack(group_edge_ig_losses)), parameters
    )
    for summary in summaries:
        summary["batch_gradient_norms"] = {
            "correctness": correctness_norm,
            "prompt_token_cost": token_cost_norm,
            "full_graph_tf": graph_tf_norm,
            "edge_ig": edge_ig_norm,
        }
    edge_counts = [
        count
        for summary in summaries
        for count in summary["mean_spatial_edges_per_round"]
    ]
    print(
        "graph gradient norms: "
        f"correctness={correctness_norm:.6f}, "
        f"prompt_token_cost={token_cost_norm:.6f}, "
        f"full_graph_tf={graph_tf_norm:.6f}, "
        f"edge_ig={edge_ig_norm:.6f}, "
        f"avg_edges={sum(edge_counts) / len(edge_counts) if edge_counts else 0.0:.2f}"
    )
    return torch.mean(torch.stack(group_losses)), summaries
