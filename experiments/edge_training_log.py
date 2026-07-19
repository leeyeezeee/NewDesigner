from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import torch


def resolve_edge_training_log_file(dataset: str) -> Path:
    return Path("result") / f"{dataset}_log.jsonl"


def reset_edge_training_log(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("", encoding="utf-8")


def resolve_question_id(record: Any, fallback_index: int) -> Any:
    for field in ("question_id", "task_id", "problem_id", "id"):
        try:
            value = record.get(field)
        except (AttributeError, TypeError):
            value = None
        if value is not None and str(value).strip():
            return str(value)
    record_name = getattr(record, "name", None)
    if record_name is not None:
        return str(record_name)
    return str(int(fallback_index))


def append_edge_training_details(
    log_file: Path,
    *,
    question_id: Any,
    edge_details: Dict[str, Dict[str, Any]],
) -> None:
    if not edge_details:
        return
    edges = []
    for detail in edge_details.values():
        if detail.get("type") != "spatial" or "ig_gain" not in detail:
            continue
        edge_id = f"{detail.get('source')}->{detail.get('target')}"
        edges.append({
            "edge": edge_id,
            "round_id": int(detail.get("round", 0)),
            "ig_gain": float(detail["ig_gain"]),
            "edge_token_cost": float(detail.get("edge_token_cost", 0.0)),
        })
    if not edges:
        return
    with log_file.open("a", encoding="utf-8") as file:
        file.write(json.dumps({
            "question_id": question_id,
            "edges": edges,
        }, ensure_ascii=False) + "\n")


def _mean(values) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0


def _range(values) -> float:
    values = [float(value) for value in values]
    return max(values) - min(values) if values else 0.0


def _mean_abs(values) -> float:
    values = [abs(float(value)) for value in values]
    return sum(values) / len(values) if values else 0.0


def _pearson(left_values, right_values):
    pairs = [
        (float(left), float(right))
        for left, right in zip(left_values, right_values)
    ]
    if len(pairs) < 2:
        return None
    left_mean = sum(left for left, _right in pairs) / len(pairs)
    right_mean = sum(right for _left, right in pairs) / len(pairs)
    left_centered = [left - left_mean for left, _right in pairs]
    right_centered = [right - right_mean for _left, right in pairs]
    left_variance = sum(value * value for value in left_centered)
    right_variance = sum(value * value for value in right_centered)
    if left_variance <= 1e-12 or right_variance <= 1e-12:
        return None
    covariance = sum(
        left * right
        for left, right in zip(left_centered, right_centered)
    )
    return covariance / ((left_variance * right_variance) ** 0.5)


def _reward_sample_diagnostics(reward_summaries) -> Dict[str, Any]:
    """Summarize whether token cost is aligned with sampled edge count."""
    token_edge_correlations = []
    global_edges = []
    global_tokens = []
    token_advantages = []
    correctness_advantages = []
    graph_advantages = []
    prompt_token_ranges = []
    normalized_token_cost_ranges = []
    edge_ranges = []
    sample_groups = []

    for group_idx, summary in enumerate(reward_summaries or []):
        edges = [
            float(value)
            for value in summary.get("mean_spatial_edges_per_round", [])
        ]
        tokens = [
            float(value)
            for value in summary.get("graph_token_counts", [])
        ]
        normalized_costs = [
            float(value)
            for value in summary.get("normalized_graph_token_costs", [])
        ]
        group_token_advantages = [
            float(value)
            for value in summary.get("token_advantages", [])
        ]
        group_correctness_advantages = [
            float(value)
            for value in summary.get("correctness_advantages", [])
        ]
        group_graph_advantages = [
            float(value)
            for value in summary.get("graph_advantages", [])
        ]
        correctness_scores = [
            float(value)
            for value in summary.get("correctness_scores", [])
        ]

        group_correlation = _pearson(edges, tokens)
        if group_correlation is not None:
            token_edge_correlations.append(group_correlation)
        if edges and tokens:
            prompt_token_ranges.append(_range(tokens))
            edge_ranges.append(_range(edges))
            global_edges.extend(edges)
            global_tokens.extend(tokens)
        if normalized_costs:
            normalized_token_cost_ranges.append(_range(normalized_costs))

        token_advantages.extend(group_token_advantages)
        correctness_advantages.extend(group_correctness_advantages)
        graph_advantages.extend(group_graph_advantages)

        samples = []
        sample_count = max(
            len(edges),
            len(tokens),
            len(normalized_costs),
            len(group_token_advantages),
            len(group_correctness_advantages),
            len(group_graph_advantages),
            len(correctness_scores),
        )
        for sample_idx in range(sample_count):
            samples.append({
                "sample": sample_idx,
                "sampled_edges": (
                    edges[sample_idx] if sample_idx < len(edges) else None
                ),
                "prompt_tokens": (
                    tokens[sample_idx] if sample_idx < len(tokens) else None
                ),
                "normalized_token_cost": (
                    normalized_costs[sample_idx]
                    if sample_idx < len(normalized_costs)
                    else None
                ),
                "correctness": (
                    correctness_scores[sample_idx]
                    if sample_idx < len(correctness_scores)
                    else None
                ),
                "token_advantage": (
                    group_token_advantages[sample_idx]
                    if sample_idx < len(group_token_advantages)
                    else None
                ),
                "correctness_advantage": (
                    group_correctness_advantages[sample_idx]
                    if sample_idx < len(group_correctness_advantages)
                    else None
                ),
                "graph_advantage": (
                    group_graph_advantages[sample_idx]
                    if sample_idx < len(group_graph_advantages)
                    else None
                ),
            })
        sample_groups.append({
            "group": group_idx,
            "token_edge_correlation": group_correlation,
            "prompt_token_range": _range(tokens),
            "sampled_edge_range": _range(edges),
            "samples": samples,
        })

    global_correlation = _pearson(global_edges, global_tokens)
    return {
        "token_edge_correlation": (
            _mean(token_edge_correlations)
            if token_edge_correlations
            else None
        ),
        "token_edge_correlation_valid_groups": len(token_edge_correlations),
        "global_token_edge_correlation": global_correlation,
        "avg_abs_token_advantage": _mean_abs(token_advantages),
        "avg_abs_correctness_advantage": _mean_abs(correctness_advantages),
        "avg_abs_graph_advantage": _mean_abs(graph_advantages),
        "avg_prompt_token_range_per_group": _mean(prompt_token_ranges),
        "avg_normalized_token_cost_range_per_group": _mean(
            normalized_token_cost_ranges
        ),
        "avg_edge_range_per_group": _mean(edge_ranges),
        "sample_groups": sample_groups,
    }


def _aggregate_named_statistics(graphs, section: str) -> Dict[str, Any]:
    names = set()
    for graph in graphs:
        names.update(
            getattr(graph, "topology_diagnostics", {})
            .get(section, {})
            .keys()
        )
    aggregated = {}
    for name in sorted(names):
        entries = [
            getattr(graph, "topology_diagnostics", {})
            .get(section, {})
            .get(name, {})
            for graph in graphs
        ]
        keys = set().union(*(entry.keys() for entry in entries if entry))
        aggregated[name] = {
            key: _mean(
                entry[key]
                for entry in entries
                if key in entry
            )
            for key in sorted(keys)
        }
    return aggregated


def _task_adaptation_proxies(graph_groups) -> Dict[str, Any]:
    group_embeddings = []
    group_probability_means = []
    for group in graph_groups:
        embeddings = [
            getattr(graph, "topology_diagnostics", {}).get(
                "final_node_embeddings"
            )
            for graph in group
        ]
        embeddings = [value.float() for value in embeddings if torch.is_tensor(value)]
        if embeddings:
            group_embeddings.append(torch.stack(embeddings).mean(dim=0))
        probability_means = [
            getattr(graph, "topology_diagnostics", {})
            .get("edge_distribution", {})
            .get("probability_mean")
            for graph in group
        ]
        probability_means = [
            float(value) for value in probability_means if value is not None
        ]
        if probability_means:
            group_probability_means.append(_mean(probability_means))

    same_role_cross_task_cosines = []
    for left_idx, left in enumerate(group_embeddings):
        for right in group_embeddings[left_idx + 1:]:
            node_count = min(int(left.size(0)), int(right.size(0)))
            if node_count == 0:
                continue
            same_role_cross_task_cosines.extend(
                torch.nn.functional.cosine_similarity(
                    left[:node_count],
                    right[:node_count],
                    dim=-1,
                    eps=1e-6,
                ).tolist()
            )

    task_probability_std = 0.0
    if len(group_probability_means) > 1:
        probability_tensor = torch.tensor(group_probability_means)
        task_probability_std = float(
            probability_tensor.std(unbiased=False).item()
        )
    cross_task_cosine = (
        _mean(same_role_cross_task_cosines)
        if same_role_cross_task_cosines
        else None
    )
    return {
        "same_role_cross_task_cosine": cross_task_cosine,
        "task_sensitivity_distance": (
            1.0 - cross_task_cosine if cross_task_cosine is not None else None
        ),
        "mean_probability_across_tasks_std": task_probability_std,
        "task_groups": len(group_embeddings),
    }


def append_topology_diagnostics(
    log_file: Path,
    *,
    iteration: int,
    graph_groups: Sequence[Sequence[Any]],
    reward_summaries: Sequence[Dict[str, Any]],
) -> None:
    """Append one compact topology-policy diagnostic record per iteration."""
    graphs = [graph for group in graph_groups for graph in group]
    if not graphs:
        return
    edge_distributions = [
        getattr(graph, "topology_diagnostics", {}).get("edge_distribution", {})
        for graph in graphs
    ]
    edge_keys = set().union(*(
        distribution.keys()
        for distribution in edge_distributions
        if distribution
    ))
    edge_distribution = {
        key: _mean(
            distribution[key]
            for distribution in edge_distributions
            if key in distribution
        )
        for key in sorted(edge_keys)
    }
    sampled_edges = [
        float(getattr(graph, "mean_spatial_edges_per_round", 0.0))
        for graph in graphs
    ]
    expected_edges = float(edge_distribution.get("expected_edges", 0.0))
    avg_sampled_edges = _mean(sampled_edges)
    gradient_norms = (
        dict(reward_summaries[0].get("batch_gradient_norms", {}))
        if reward_summaries
        else {}
    )
    correctness_scores = [
        float(score)
        for summary in reward_summaries
        for score in summary.get("correctness_scores", [])
    ]
    node_embeddings = _aggregate_named_statistics(graphs, "node_embeddings")
    final_cosine = (
        node_embeddings.get("gatv2_layer2", {}).get("mean_pairwise_cosine")
    )
    input_cosine = (
        node_embeddings.get("joint_input", {}).get("mean_pairwise_cosine")
    )
    task_adaptation_proxy = _task_adaptation_proxies(graph_groups)
    within_task_role_separation = (
        1.0 - float(final_cosine) if final_cosine is not None else None
    )
    task_adaptation_proxy["within_task_role_separation"] = (
        within_task_role_separation
    )
    task_sensitivity = task_adaptation_proxy["task_sensitivity_distance"]
    task_adaptation_proxy["task_to_role_separation_ratio"] = (
        float(task_sensitivity / within_task_role_separation)
        if (
            task_sensitivity is not None
            and within_task_role_separation is not None
            and within_task_role_separation > 1e-12
        )
        else None
    )
    record = {
        "record_type": "topology_diagnostics",
        "iteration": int(iteration),
        "architecture": getattr(
            graphs[0], "spatial_policy_architecture", "unknown"
        ),
        "num_graphs": len(graphs),
        "accuracy": _mean(correctness_scores),
        "initial_residual_weight": float(
            getattr(graphs[0], "topology_diagnostics", {}).get(
                "initial_residual_weight", 0.0
            )
        ),
        "sampling_temperature": float(
            getattr(graphs[0], "topology_diagnostics", {}).get(
                "sampling_temperature", 0.0
            )
        ),
        "node_embeddings": node_embeddings,
        "oversmoothing_cosine_delta": (
            float(final_cosine - input_cosine)
            if final_cosine is not None and input_cosine is not None
            else None
        ),
        "attention": _aggregate_named_statistics(graphs, "attention"),
        "edge_distribution": edge_distribution,
        "sampling": {
            "avg_sampled_edges": avg_sampled_edges,
            "avg_expected_edges": expected_edges,
            "sample_minus_expected_edges": avg_sampled_edges - expected_edges,
        },
        "reward_sample_diagnostics": _reward_sample_diagnostics(
            reward_summaries
        ),
        "task_adaptation_proxy": task_adaptation_proxy,
        "reward_gradient_norms": gradient_norms,
    }
    with log_file.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
