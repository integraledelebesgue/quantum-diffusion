from __future__ import annotations

import einops
import torch
from typing_extensions import override

from .utils import get_label_embedding


def build_conv_layers(channels: list[int]) -> torch.nn.Sequential:
    """Build convolutional layers for undirected models."""
    layers: list[torch.nn.Module] = []
    
    for i in range(len(channels) - 1):
        layers.append(
            torch.nn.Conv2d(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=3,
                padding=1,
            )
        )
        layers.append(torch.nn.ReLU())

    layers.append(torch.nn.Sigmoid())
    
    return torch.nn.Sequential(*layers)


class DeepConvUndirected(torch.nn.Module):
    """Deep Convolutional Neural Network. Undirected"""

    channels: list[int]
    shape: tuple[int, int]
    net: torch.nn.Sequential

    def __init__(self, channels: list[int], shape: tuple[int, int]) -> None:
        super().__init__()
        assert channels[0] == channels[-1], "Input and output channels must be equal"

        self.channels = channels
        self.net = build_conv_layers(channels)
        self.shape = shape

    @override
    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        assert len(x.shape) == 4, "Input must be 4D tensor"
        
        return self.net.forward(x)

    @override
    def __repr__(self) -> str:
        return f"DeepConvUndirected({self.net})"

    def save_name(self) -> str:
        return f"deep_conv_undirected_{'_'.join(map(str, self.channels))}"


class DeepConvDirectedMulti(torch.nn.Module):
    """Deep Convolutional Neural Network. Directed"""

    channels: list[int]
    layers: torch.nn.ModuleList

    def __init__(self, channels: list[int]):
        super().__init__()
        assert channels[0] == channels[-1], "Input and output channels must be equal"

        self.channels = channels

        layers: list[torch.nn.Module] = []
        
        for i in range(len(channels) - 1):
            layers.append(
                torch.nn.Conv2d(
                    in_channels=channels[i] + 1,
                    out_channels=channels[i + 1],
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(torch.nn.ReLU())

        layers[-1] = torch.nn.Sigmoid()
        self.layers = torch.nn.ModuleList(layers)

    @override
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, "Input must be 4D tensor"

        y = einops.repeat(y, "b -> b 1 h w", h=x.shape[2], w=x.shape[3])

        for layer in self.layers:
            if isinstance(layer, torch.nn.Conv2d):
                x = torch.cat((x, y), dim=1)  # Concatenate label channel
            
            x = layer.forward(x)

        return x

    @override
    def __repr__(self) -> str:
        return f"DeepConvDirectedMulti({self.layers})"

    def save_name(self) -> str:
        return f"deep_conv_directed_multi_{'_'.join(map(str, self.channels))}"


class DeepConvDirectedSingle(torch.nn.Module):
    """Deep Convolutional Neural Network. Directed (single label embedding)"""

    channels: list[int]
    shape: tuple[int, int]
    net: torch.nn.Sequential

    def __init__(self, channels: list[int], shape: tuple[int, int]) -> None:
        super().__init__()
        assert channels[0] == channels[-1], "Input and output channels must be equal"

        self.channels = channels
        self.net = build_conv_layers(channels)
        self.shape = shape

    @override
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, "Input must be 4D tensor"

        y = y.unsqueeze(-1)
        mask = get_label_embedding(y, self.shape[0], self.shape[1])
        masked_x = x + mask

        return self.net.forward(masked_x)

    @override
    def __repr__(self) -> str:
        return f"DeepConvDirectedSingle({self.net})"

    def save_name(self) -> str:
        return f"deep_conv_directed_single_{'_'.join(map(str, self.channels))}"
