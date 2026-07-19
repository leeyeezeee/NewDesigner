import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class ResidualGATEncoder(torch.nn.Module):
    """Two-layer residual GAT encoder for task-conditioned role features."""

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
        self.conv1 = GATConv(
            hidden_channels,
            head_channels,
            heads=heads,
            concat=True,
            dropout=self.dropout,
            add_self_loops=True,
        )
        self.norm1 = torch.nn.LayerNorm(hidden_channels)
        self.conv2 = GATConv(
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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(self.input_projection(x))

        update = self.conv1(x, edge_index)
        update = F.elu(update)
        update = F.dropout(update, p=self.dropout, training=self.training)
        x = self.norm1(x + update)

        update = self.conv2(x, edge_index)
        update = F.elu(update)
        update = F.dropout(update, p=self.dropout, training=self.training)
        return self.norm2(x + update)
