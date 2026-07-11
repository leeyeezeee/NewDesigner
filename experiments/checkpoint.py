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
            "refine_rank": int(graph.refine_rank),
            "edge_bias_scale": float(getattr(graph, "edge_bias_scale", 0.0)),
            "gcn_state_dict": graph.gcn.state_dict(),
            "mlp_state_dict": graph.mlp.state_dict(),
            "spatial_affinity_weight": graph.spatial_affinity_weight.detach().cpu(),
            "refinement_weight": graph.refinement_weight.detach().cpu(),
            "spatial_edge_bias": graph.spatial_edge_bias.detach().cpu(),
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

    if "edge_bias_scale" in graph_state:
        graph.edge_bias_scale = float(graph_state["edge_bias_scale"])
    if "gcn_state_dict" in graph_state:
        graph.gcn.load_state_dict(graph_state["gcn_state_dict"])
    if "mlp_state_dict" in graph_state:
        graph.mlp.load_state_dict(graph_state["mlp_state_dict"])
    if "spatial_affinity_weight" in graph_state:
        _copy_parameter(
            graph.spatial_affinity_weight,
            graph_state["spatial_affinity_weight"],
            "spatial_affinity_weight",
        )
    if "refinement_weight" in graph_state:
        _copy_parameter(graph.refinement_weight, graph_state["refinement_weight"], "refinement_weight")
    if "spatial_edge_bias" in graph_state:
        _copy_parameter(graph.spatial_edge_bias, graph_state["spatial_edge_bias"], "spatial_edge_bias")
    if "temporal_logits" in graph_state:
        _copy_parameter(graph.temporal_logits, graph_state["temporal_logits"], "temporal_logits")
    if "spatial_masks" in graph_state:
        _copy_parameter(graph.spatial_masks, graph_state["spatial_masks"], "spatial_masks")
    if "temporal_masks" in graph_state:
        _copy_parameter(graph.temporal_masks, graph_state["temporal_masks"], "temporal_masks")

    if load_optimizer is not None and "optimizer_state_dict" in checkpoint:
        load_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loaded checkpoint: {checkpoint_path}")
    return checkpoint
