"""
Configurable MLP with options for:
 - Dropout
 - Normalization layers (BatchNorm / LayerNorm)
 - Skip/residual connections
 - Bottleneck layers (narrow intermediate layers)
Designed for experiments: easy to instantiate from a config dict / args.

Requires: torch
"""
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Simple residual block for fully-connected layers.
    It contains two Linear -> Norm -> Activation -> Dropout sequences and adds
    the block input to the output (if shapes match).
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        norm: Optional[nn.Module] = None,
        activation: nn.Module = nn.ReLU,
        dropout: float = 0.0,
    ):
        super().__init__()
        h = hidden_dim if hidden_dim is not None else dim
        self.fc1 = nn.Linear(dim, h)
        self.norm1 = norm(h) if norm is not None else None
        self.act1 = activation()
        self.drop1 = nn.Dropout(dropout) if dropout and dropout > 0 else None

        self.fc2 = nn.Linear(h, dim)
        self.norm2 = norm(dim) if norm is not None else None
        self.act2 = activation()
        self.drop2 = nn.Dropout(dropout) if dropout and dropout > 0 else None

    def forward(self, x):
        out = self.fc1(x)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.act1(out)
        if self.drop1 is not None:
            out = self.drop1(out)

        out = self.fc2(out)
        if self.norm2 is not None:
            out = self.norm2(out)
        # Note: typically residual connection is added before final non-linearity,
        # but here we follow common practice: add then apply activation.
        out = out + x
        out = self.act2(out)
        if self.drop2 is not None:
            out = self.drop2(out)
        return out


class ConfigurableMLP(nn.Module):
    """
    Build an MLP from a simple experiment config.

    Args:
      input_dim: int, number of input features
      output_dim: int, number of outputs (e.g., classes for classification)
      hidden_dims: list[int], sizes of hidden layers (e.g. [512, 256, 128])
      activation: torch.nn.Module class or callable (default: nn.ReLU)
      dropout: float or list[float] specifying dropout rate(s)
      norm: None | 'batch' | 'layer'
      use_skip: bool - enable residual connections where possible
      bottleneck: dict | None - if provided, describes bottleneck block(s),
                  e.g. {'pos': 'middle', 'factor': 4} will make a bottleneck
                  in the middle layer sized hidden_dim // factor
      final_activation: None | torch.nn.Module - activation after output layer
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: type = nn.ReLU,
        dropout: float = 0.0,
        norm: Optional[str] = None,
        use_skip: bool = False,
        bottleneck: Optional[dict] = None,
        final_activation: Optional[type] = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        # Normalize dropout to a list of same length as hidden layers
        if isinstance(dropout, (int, float)):
            dropout = [float(dropout)] * len(hidden_dims)
        elif isinstance(dropout, list):
            if len(dropout) != len(hidden_dims):
                raise ValueError("dropout list must match hidden_dims length")

        # Norm factory
        def norm_factory(size):
            if norm == 'batch':
                return nn.BatchNorm1d
            elif norm == 'layer':
                return lambda dim: nn.LayerNorm(dim)
            else:
                return None

        Norm = norm_factory(None)

        layers: List[nn.Module] = []
        in_dim = input_dim

        # Optionally insert a bottleneck: the config can specify
        # {'position': 'middle'|'first'|'last'|'indices':[i,j], 'factor': int}
        bottleneck_positions = []
        bottleneck_map = {}
        if bottleneck:
            factor = int(bottleneck.get('factor', 4))
            pos = bottleneck.get('position', 'middle')
            # compute index to insert bottleneck layer
            if pos == 'middle':
                idx = len(hidden_dims) // 2
                bottleneck_positions = [idx]
            elif pos == 'first':
                bottleneck_positions = [0]
            elif pos == 'last':
                bottleneck_positions = [len(hidden_dims)-1]
            elif isinstance(pos, list):
                bottleneck_positions = pos
            else:
                raise ValueError("Unsupported bottleneck position: %r" % (pos,))
            # map indexes -> bottleneck size
            for i in bottleneck_positions:
                bottleneck_map[i] = max(1, hidden_dims[i] // factor)

        # If using skip connections, we will consider grouping two sequential
        # layers into a ResidualBlock when dims allow (input dim -> output dim equal)
        i = 0
        layer_idx = 0
        while i < len(hidden_dims):
            h = hidden_dims[i]
            dr = dropout[i]
            # Check for bottleneck at this index
            if i in bottleneck_map:
                bottleneck_dim = bottleneck_map[i]
                # build 3-layer bottleneck: in_dim -> bottleneck_dim -> h
                # linear -> norm -> act -> dropout -> linear -> norm -> act
                layers.append(nn.Linear(in_dim, bottleneck_dim))
                if Norm is not None:
                    layers.append(Norm(bottleneck_dim))
                layers.append(activation())
                if dr > 0:
                    layers.append(nn.Dropout(dr))

                layers.append(nn.Linear(bottleneck_dim, h))
                if Norm is not None:
                    layers.append(Norm(h))
                layers.append(activation())
                if dr > 0:
                    layers.append(nn.Dropout(dr))

                in_dim = h
                i += 1
                layer_idx += 1
                continue

            # If use_skip and next layer has same size as in_dim, create ResidualBlock
            if use_skip and (i + 1 < len(hidden_dims)) and (hidden_dims[i+1] == in_dim):
                # build residual block with dimension in_dim
                # we will consume one hidden size (hidden_dims[i]) but ensure shapes
                # are maintained. To be conservative, only build ResidualBlock when
                # in_dim == hidden_dims[i+1] so identity add is defined.
                rb = ResidualBlock(
                    dim=in_dim,
                    hidden_dim=hidden_dims[i],
                    norm=(lambda d: Norm(d)) if Norm is not None else None,
                    activation=activation,
                    dropout=dr,
                )
                layers.append(rb)
                # After residual block, next index consumed (i+1) as it's same shape
                in_dim = in_dim  # no change
                i += 2
                layer_idx += 1
                continue

            # Standard linear layer
            layers.append(nn.Linear(in_dim, h))
            if Norm is not None:
                layers.append(Norm(h))
            layers.append(activation())
            if dr > 0:
                layers.append(nn.Dropout(dr))

            in_dim = h
            i += 1
            layer_idx += 1

        # Final head
        self.feature_extractor = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, output_dim)
        self.final_activation = final_activation() if final_activation is not None else None

        # Good initialization
        self._init_weights()

    def _init_weights(self):
        # Kaiming init for linear layers, small bias
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                try:
                    if hasattr(m, 'weight') and m.weight is not None:
                        nn.init.constant_(m.weight, 1.0)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                except Exception:
                    pass

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.head(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x