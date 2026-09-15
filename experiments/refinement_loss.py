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
    """Keep legacy zero-valued diagnostics without a retired refinement loss."""
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

    if anchor_reg_weight != 0.0 or sparsity_reg_weight != 0.0:
        raise ValueError(
            "The direct-affinity decoder has no low-rank refinement penalties; "
            "anchor_reg_weight and sparsity_reg_weight must both be 0."
        )
    zero = reference_loss.new_tensor(0.0)
    return zero, zero, zero
