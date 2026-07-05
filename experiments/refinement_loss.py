from typing import Iterable, Tuple

import torch


def refinement_regularization_loss(
    realized_graphs: Iterable,
    reference_loss: torch.Tensor,
    anchor_reg_weight: float = 1.0,
    sparsity_reg_weight: float = 1.0,
    edge_bias_l2_weight: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    anchor_losses = []
    sparse_losses = []
    edge_bias_losses = []
    for graph in realized_graphs:
        anchor_loss = getattr(graph, "refinement_anchor_loss", None)
        sparse_loss = getattr(graph, "refinement_sparse_loss", None)
        edge_bias_loss = getattr(graph, "edge_bias_l2_loss", None)
        if anchor_loss is not None:
            anchor_losses.append(anchor_loss)
        if sparse_loss is not None:
            sparse_losses.append(sparse_loss)
        if edge_bias_loss is not None:
            edge_bias_losses.append(edge_bias_loss)

    zero = reference_loss.new_tensor(0.0)
    anchor_loss = torch.mean(torch.stack(anchor_losses)) if anchor_losses else zero
    sparse_loss = torch.mean(torch.stack(sparse_losses)) if sparse_losses else zero
    edge_bias_loss = torch.mean(torch.stack(edge_bias_losses)) if edge_bias_losses else zero
    reg_loss = (
        anchor_reg_weight * anchor_loss
        + sparsity_reg_weight * sparse_loss
        + edge_bias_l2_weight * edge_bias_loss
    )
    return reg_loss, anchor_loss, sparse_loss
