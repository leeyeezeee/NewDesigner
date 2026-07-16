from typing import Iterable, Tuple

import torch


def refinement_regularization_loss(
    realized_graphs: Iterable,
    reference_loss: torch.Tensor,
    anchor_reg_weight: float = 1.0,
    sparsity_reg_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the original GDesigner anchor and nuclear-norm penalties."""
    anchor_losses = []
    sparse_losses = []
    for graph in realized_graphs:
        anchor_loss = getattr(graph, "refinement_anchor_loss", None)
        sparse_loss = getattr(graph, "refinement_sparse_loss", None)
        if anchor_loss is not None:
            anchor_losses.append(anchor_loss)
        if sparse_loss is not None:
            sparse_losses.append(sparse_loss)

    zero = reference_loss.new_tensor(0.0)
    anchor_loss = (
        torch.mean(torch.stack(anchor_losses)) if anchor_losses else zero
    )
    sparse_loss = (
        torch.mean(torch.stack(sparse_losses)) if sparse_losses else zero
    )
    reg_loss = (
        float(anchor_reg_weight) * anchor_loss
        + float(sparsity_reg_weight) * sparse_loss
    )
    return reg_loss, anchor_loss, sparse_loss
