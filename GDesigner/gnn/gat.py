import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class InitialResidualGATv2Encoder(torch.nn.Module):
    """Two-layer GATv2 encoder with a fixed GCNII-style initial residual."""

    INITIAL_RESIDUAL_WEIGHT = 0.2

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_channels % heads != 0:
            raise ValueError(
                "hidden_channels must be divisible by heads; "
                f"received {hidden_channels=} and {heads=}."
            )

        head_channels = hidden_channels // heads
        self.dropout = float(dropout)
        self.input_projection = torch.nn.Linear(
            in_channels,
            hidden_channels,
            bias=False,
        )
        self.input_norm = torch.nn.LayerNorm(hidden_channels)
        self.conv1 = GATv2Conv(
            hidden_channels,
            head_channels,
            heads=heads,
            concat=True,
            dropout=self.dropout,
            add_self_loops=True,
        )
        self.norm1 = torch.nn.LayerNorm(hidden_channels)
        self.conv2 = GATv2Conv(
            hidden_channels,
            head_channels,
            heads=heads,
            concat=True,
            dropout=self.dropout,
            add_self_loops=True,
        )
        self.norm2 = torch.nn.LayerNorm(hidden_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.input_projection.weight)
        self.input_norm.reset_parameters()
        self.conv1.reset_parameters()
        self.norm1.reset_parameters()
        self.conv2.reset_parameters()
        self.norm2.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ):
        initial = self.input_norm(self.input_projection(x))
        alpha = self.INITIAL_RESIDUAL_WEIGHT

        update, attention1 = self.conv1(
            initial,
            edge_index,
            return_attention_weights=True,
        )
        update = F.elu(update)
        update = F.dropout(update, p=self.dropout, training=self.training)
        layer1 = self.norm1((1.0 - alpha) * update + alpha * initial)

        update, attention2 = self.conv2(
            layer1,
            edge_index,
            return_attention_weights=True,
        )
        update = F.elu(update)
        update = F.dropout(update, p=self.dropout, training=self.training)
        layer2 = self.norm2((1.0 - alpha) * update + alpha * initial)
        if not return_diagnostics:
            return layer2
        return layer2, {
            "initial": initial.detach(),
            "layer1": layer1.detach(),
            "layer2": layer2.detach(),
            "attention1_edge_index": attention1[0].detach(),
            "attention1_weights": attention1[1].detach(),
            "attention2_edge_index": attention2[0].detach(),
            "attention2_weights": attention2[1].detach(),
            "initial_residual_weight": alpha,
        }
