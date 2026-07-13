import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    """Three-layer graph-attention encoder for task-conditioned agent features."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.conv1 = GATConv(in_channels, hidden_channels, heads=1, concat=False)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=1, concat=False)
        self.conv3 = GATConv(hidden_channels, out_channels, heads=1, concat=False)

    def reset_parameters(self) -> None:
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv3(x, edge_index)
