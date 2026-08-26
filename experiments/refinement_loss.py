import math
from typing import Iterable, Tuple

import torch


def refinement_regularization_loss(
    realized_graphs: Iterable,
    reference_loss: torch.Tensor,
    *,
    anchor_reg_weight: float = 0.0,
    sparsity_reg_weight: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return G-Designer's anchor and nuclear-norm refinement penalties."""
    anchor_reg_weight = float(anchor_reg_weight)
    sparsity_reg_weight = float(sparsity_reg_weight)
    if (
        not math.isfinite(anchor_reg_weight)
        or not math.isfinite(sparsity_reg_weight)
        or anchor_reg_weight < 0.0
        or sparsity_reg_weight < 0.0
    ):
        raise ValueError(
            "Refinement regularization weights must be non-negative; "
            f"received anchor={anchor_reg_weight}, sparsity={sparsity_reg_weight}."
        )

    anchor_losses = []
    sparse_losses = []
    for graph in realized_graphs:
        anchor_loss = getattr(graph, "refinement_anchor_loss", None)
        sparse_loss = getattr(graph, "refinement_sparse_loss", None)
        if torch.is_tensor(anchor_loss):
            anchor_losses.append(anchor_loss)
        if torch.is_tensor(sparse_loss):
            sparse_losses.append(sparse_loss)

    zero = reference_loss.new_tensor(0.0)
    anchor_loss = (
        torch.mean(torch.stack(anchor_losses)) if anchor_losses else zero
    )
    sparse_loss = (
        torch.mean(torch.stack(sparse_losses)) if sparse_losses else zero
    )
    regularization_loss = (
        anchor_loss * anchor_reg_weight
        + sparse_loss * sparsity_reg_weight
    )
    return regularization_loss, anchor_loss, sparse_loss
