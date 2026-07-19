import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class InitialResidualGATv2Encoder(torch.nn.Module):
    """Two-layer bottleneck GATv2 encoder matching G-Designer's GCN shape."""

    INITIAL_RESIDUAL_WEIGHT = 0.5

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int = 16,
        out_channels: int = 384,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if bottleneck_channels % heads != 0:
            raise ValueError(
                "bottleneck_channels must be divisible by heads; "
                f"received {bottleneck_channels=} and {heads=}."
            )
        if out_channels % heads != 0:
            raise ValueError(
                "out_channels must be divisible by heads; "
                f"received {out_channels=} and {heads=}."
            )

        bottleneck_head_channels = bottleneck_channels // heads
        out_head_channels = out_channels // heads
        self.dropout = float(dropout)
        self.input_norm = torch.nn.LayerNorm(in_channels)
        self.initial_projection1 = torch.nn.Linear(
            in_channels,
            bottleneck_channels,
            bias=False,
        )
        self.initial_projection2 = torch.nn.Linear(
            in_channels,
            out_channels,
            bias=False,
        )
        self.conv1 = GATv2Conv(
            in_channels,
            bottleneck_head_channels,
            heads=heads,
            concat=True,
            dropout=self.dropout,
            add_self_loops=True,
        )
        self.norm1 = torch.nn.LayerNorm(bottleneck_channels)
        self.conv2 = GATv2Conv(
            bottleneck_channels,
            out_head_channels,
            heads=heads,
            concat=True,
            dropout=self.dropout,
            add_self_loops=True,
        )
        self.norm2 = torch.nn.LayerNorm(out_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.input_norm.reset_parameters()
        torch.nn.init.xavier_uniform_(self.initial_projection1.weight)
        torch.nn.init.xavier_uniform_(self.initial_projection2.weight)
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
        normalized_input = self.input_norm(x)
        initial1 = self.initial_projection1(normalized_input)
        initial2 = self.initial_projection2(normalized_input)
        alpha = self.INITIAL_RESIDUAL_WEIGHT

        update, attention1 = self.conv1(
            normalized_input,
            edge_index,
            return_attention_weights=True,
        )
        update = F.elu(update)
        update = F.dropout(update, p=self.dropout, training=self.training)
        layer1 = self.norm1((1.0 - alpha) * update + alpha * initial1)

        update, attention2 = self.conv2(
            layer1,
            edge_index,
            return_attention_weights=True,
        )
        update = F.elu(update)
        update = F.dropout(update, p=self.dropout, training=self.training)
        layer2 = self.norm2((1.0 - alpha) * update + alpha * initial2)
        if not return_diagnostics:
            return layer2
        return layer2, {
            "initial": initial2.detach(),
            "initial_bottleneck": initial1.detach(),
            "layer1": layer1.detach(),
            "layer2": layer2.detach(),
            "attention1_edge_index": attention1[0].detach(),
            "attention1_weights": attention1[1].detach(),
            "attention2_edge_index": attention2[0].detach(),
            "attention2_weights": attention2[1].detach(),
            "initial_residual_weight": alpha,
        }
