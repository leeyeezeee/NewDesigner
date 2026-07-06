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
