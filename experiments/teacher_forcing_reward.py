import math
import random
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from experiments.agent_backend import add_agent_backend_args
from GDesigner.utils.ig_scorer import edge_key


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
            "Use multi-sampled final-answer correctness advantages. Per-edge "
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
        "--use_graph_critic",
        action="store_true",
        help=(
            "Use the optional lightweight GCN critic to approximate selected "
            "edge deletion rewards. Disabled by default; real edge ablation "
            "remains the default implementation."
        ),
    )
    parser.add_argument(
        "--graph_critic_lr",
        type=float,
        default=1e-3,
        help="Learning rate for the independent graph critic.",
    )
    parser.add_argument(
        "--graph_critic_reward_lambda",
        type=float,
        default=0.2,
        help="Coefficient for critic-predicted counterfactual edge rewards.",
    )
    parser.add_argument(
        "--graph_critic_warmup_iterations",
        type=int,
        default=2,
        help="Critic-only fitting iterations before its edge rewards affect the actor.",
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
        help="Coefficient for per-edge teacher-forcing information gain.",
    )
    parser.add_argument(
        "--edge_ig_discount_factor",
        type=float,
        default=0.0,
        help="Within-round downstream IG discount; 0 keeps immediate IG only.",
    )
    parser.add_argument(
        "--graph_advantage_epsilon",
        type=float,
        default=1e-6,
        help="Small constant used when standardizing correctness advantages.",
    )
    parser.add_argument(
        "--edge_ig_warmup_iterations",
        type=int,
        default=2,
        help="Initial iterations that skip counterfactual edge-IG calls and loss.",
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


def set_experiment_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def experiment_summary_metadata(args: Any, dataset: str) -> Dict[str, Any]:
    """Describe the reward actually enabled by the parsed runtime arguments."""
    edge_lambda_value = getattr(args, "edge_ig_reward_lambda", None)
    edge_lambda = 0.0 if edge_lambda_value is None else float(edge_lambda_value)
    use_critic = bool(getattr(args, "use_graph_critic", False))
    use_group_advantage = bool(getattr(args, "use_graph_tf_reward", False)) or use_critic
    edge_rewards = []
    if edge_lambda != 0.0:
        edge_rewards.append(
            "execution_score_diff"
            if str(dataset).lower() == "humaneval"
            else "teacher_logprob_diff"
        )
    if use_critic:
        edge_rewards.append("critic_q_difference")
    optimized = bool(getattr(args, "optimized_spatial", False)) or bool(
        getattr(args, "optimized_temporal", False)
    )
    if not optimized:
        method = f"fixed_{getattr(args, 'mode', 'unknown')}"
    elif use_critic:
        method = "optimized_graph_critic"
    elif edge_rewards:
        method = "optimized_graph_edge_ig"
    else:
        method = "optimized_graph"
    return {
        "method": method,
        "seed": int(getattr(args, "seed", 42)),
        "reward": {
            "graph": (
                "standardized_correctness"
                if use_group_advantage
                else "binary_correctness"
            ),
            "edge": edge_rewards or ["none"],
            "lambda": edge_lambda,
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
    if not reward_scores:
        return 0.0, [], 0.0, 0.0
    scores = [float(score) for score in reward_scores]
    baseline = sum(scores) / len(scores)
    advantages = [score - baseline for score in scores]
    variance = sum(value * value for value in advantages) / len(advantages)
    std = math.sqrt(variance)
    if std > float(advantage_epsilon):
        advantages = [value / std for value in advantages]
    return baseline, advantages, variance, std


def edge_information_gain_loss(
    graph,
    edge_details: Dict[str, Dict[str, Any]],
    reference_loss: torch.Tensor,
    *,
    edge_tanh_temperature: float = 1.0,
    edge_ig_reward_lambda: float = 0.0,
    edge_ig_discount_factor: float = 0.0,
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
    edge_ig_discount_factor: float = 0.0,
    advantage_epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Combine graph correctness advantages and selected-edge teacher-forcing IG."""
    if not (
        len(graph_groups)
        == len(graph_log_prob_groups)
        == len(correctness_groups)
        == len(edge_detail_groups)
    ):
        raise ValueError("Graph reward batches must contain equal group counts.")

    zero = reference_loss.new_tensor(0.0)
    group_losses: List[torch.Tensor] = []
    group_correctness_losses: List[torch.Tensor] = []
    group_edge_ig_losses: List[torch.Tensor] = []
    summaries: List[Dict[str, Any]] = []
    for graphs, graph_log_probs, scores, edge_details_list in zip(
        graph_groups,
        graph_log_prob_groups,
        correctness_groups,
        edge_detail_groups,
    ):
        if not (
            len(graphs)
            == len(graph_log_probs)
            == len(scores)
            == len(edge_details_list)
        ):
            raise ValueError("Each graph reward group must have aligned samples.")

        baseline, advantages, variance, std = _standardized_advantages(
            scores, advantage_epsilon=advantage_epsilon
        )
        per_graph_losses = []
        correctness_losses = []
        edge_ig_losses = []
        edge_ig_records = []
        edge_ig_coefficients = []
        sampled_edges = 0
        missing_log_prob = 0
        missing_detail = 0
        missing_ig_gain = 0
        for advantage, graph, graph_log_prob, edge_details in zip(
            advantages, graphs, graph_log_probs, edge_details_list
        ):
            if not torch.is_tensor(graph_log_prob):
                raise TypeError("Full graph log-prob must be a torch.Tensor.")
            correctness_loss = -(
                graph_log_prob.new_tensor(float(advantage)) * graph_log_prob
            )
            ig_loss, ig_summary = edge_information_gain_loss(
                graph,
                edge_details,
                graph_log_prob,
                edge_tanh_temperature=edge_tanh_temperature,
                edge_ig_reward_lambda=edge_ig_reward_lambda,
                edge_ig_discount_factor=edge_ig_discount_factor,
            )
            per_graph_losses.append(correctness_loss + ig_loss)
            correctness_losses.append(correctness_loss)
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
        group_edge_ig_losses.append(
            torch.mean(torch.stack(edge_ig_losses)) if edge_ig_losses else zero
        )
        summaries.append({
            "correctness_scores": [float(score) for score in scores],
            "correctness_baseline": float(baseline),
            "correctness_variance": float(variance),
            "correctness_std": float(std),
            "correctness_advantages": [float(value) for value in advantages],
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
    edge_ig_norm = _loss_gradient_l2_norm(
        torch.mean(torch.stack(group_edge_ig_losses)), parameters
    )
    for summary in summaries:
        summary["batch_gradient_norms"] = {
            "correctness": correctness_norm,
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
        f"edge_ig={edge_ig_norm:.6f}, "
        f"avg_edges={sum(edge_counts) / len(edge_counts) if edge_counts else 0.0:.2f}"
    )
    return torch.mean(torch.stack(group_losses)), summaries
