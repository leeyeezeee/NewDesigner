from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _args_to_dict(args: Any) -> Dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, dict):
        return dict(args)
    return {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool, type(None), list, tuple, dict))
    }


def save_graph_checkpoint(
    graph,
    checkpoint_file: Optional[str],
    *,
    dataset: str,
    args: Any = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    edge_selector: Optional[torch.nn.Module] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    if not checkpoint_file:
        return

    output_path = Path(checkpoint_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    checkpoint = {
        "dataset": dataset,
        "graph": {
            "domain": graph.domain,
            "llm_name": graph.llm_name,
            "agent_names": list(graph.agent_names),
            "optimized_spatial": bool(graph.optimized_spatial),
            "optimized_temporal": bool(graph.optimized_temporal),
            "spatial_policy_architecture": "residual_gat_2x64_4head_v1",
            "gat_state_dict": graph.gat.state_dict(),
            "spatial_affinity_state_dict": graph.spatial_affinity.state_dict(),
            "spatial_masks": graph.spatial_masks.detach().cpu(),
            "temporal_logits": graph.temporal_logits.detach().cpu(),
            "temporal_masks": graph.temporal_masks.detach().cpu(),
        },
        "args": _args_to_dict(args),
        "metrics": dict(metrics or {}),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if edge_selector is not None:
        checkpoint["edge_selector_state_dict"] = edge_selector.state_dict()

    torch.save(checkpoint, tmp_path)
    tmp_path.replace(output_path)
    print(f"Saved checkpoint: {output_path}")


def _copy_parameter(target: torch.nn.Parameter, value: torch.Tensor, name: str) -> None:
    if tuple(target.shape) != tuple(value.shape):
        raise ValueError(
            f"Checkpoint tensor {name!r} has shape {tuple(value.shape)}, "
            f"but graph expects {tuple(target.shape)}."
        )
    with torch.no_grad():
        target.copy_(value.to(device=target.device, dtype=target.dtype))


def _copy_graph_mask(graph, target, value, name: str, *, spatial: bool) -> None:
    value = graph._prepare_fixed_mask(
        value,
        graph.num_nodes - 1,
        graph.num_nodes,
        spatial=spatial,
    )
    _copy_parameter(target, value, name)


def _copy_temporal_logits(graph, value: torch.Tensor) -> None:
    target = graph.temporal_logits
    if tuple(target.shape) != tuple(value.shape):
        regular_node_count = graph.num_nodes - 1
        if value.numel() == regular_node_count * regular_node_count:
            expanded = target.detach().clone().view(graph.num_nodes, graph.num_nodes)
            expanded[:regular_node_count, :regular_node_count] = value.view(
                regular_node_count,
                regular_node_count,
            ).to(device=expanded.device, dtype=expanded.dtype)
            value = expanded.reshape(-1)
    _copy_parameter(target, value, "temporal_logits")


def load_graph_checkpoint(
    graph,
    checkpoint_file: str,
    *,
    load_optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_file)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    graph_state = checkpoint.get("graph", {})

    if "gat_state_dict" in graph_state:
        graph.gat.load_state_dict(graph_state["gat_state_dict"])
    elif any(
        key in graph_state
        for key in (
            "gcn_state_dict",
            "node_self_projection_state_dict",
            "node_feature_norm_state_dict",
            "mlp_state_dict",
        )
    ):
        raise ValueError(
            "This checkpoint contains the retired GCN/MLP spatial policy and "
            "cannot be loaded into the residual GAT policy. Retrain the spatial "
            "policy with the current architecture."
        )
    else:
        raise ValueError(
            "Checkpoint does not contain a 'gat_state_dict' for the current "
            "residual GAT spatial policy."
        )
    if "spatial_affinity_state_dict" in graph_state:
        graph.spatial_affinity.load_state_dict(
            graph_state["spatial_affinity_state_dict"]
        )
    elif "spatial_affinity_weight" in graph_state:
        # Backward compatibility for checkpoints created before the affinity
        # decoder became a spectrally normalized Linear module.
        _copy_parameter(
            graph.spatial_affinity.parametrizations.weight.original,
            graph_state["spatial_affinity_weight"],
            "spatial_affinity_weight",
        )
    if "temporal_logits" in graph_state:
        _copy_temporal_logits(graph, graph_state["temporal_logits"])
    if "spatial_masks" in graph_state:
        _copy_graph_mask(
            graph,
            graph.spatial_masks,
            graph_state["spatial_masks"],
            "spatial_masks",
            spatial=True,
        )
    if "temporal_masks" in graph_state:
        _copy_graph_mask(
            graph,
            graph.temporal_masks,
            graph_state["temporal_masks"],
            "temporal_masks",
            spatial=False,
        )

    if load_optimizer is not None and "optimizer_state_dict" in checkpoint:
        load_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loaded checkpoint: {checkpoint_path}")
    return checkpoint
