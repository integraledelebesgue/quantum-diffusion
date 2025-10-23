import einops
import torch
from typing_extensions import override


class DenseUndirected(torch.nn.Module):
    """Dense neural network. Undirected"""

    shapes: list[int]
    net: torch.nn.Sequential

    def __init__(self, shapes: list[int]) -> None:
        super().__init__()
        assert shapes[0] == shapes[-1], "Input and output shapes must be equal"

        self.shapes = shapes

        layers: list[torch.nn.Module] = []
        for i in range(len(shapes) - 1):
            layers.append(
                torch.nn.Linear(in_features=shapes[i], out_features=shapes[i + 1])
            )
            layers.append(torch.nn.ReLU())

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        _, _, w, h = x.shape
        x = einops.rearrange(x, "b 1 w h -> b (w h)")
        x = self.net.forward(x)
        x = einops.rearrange(x, "b (w h) -> b 1 w h", w=w, h=h)
        return x

    @override
    def __repr__(self) -> str:
        return f"DenseUndirected({self.net})"

    def save_name(self) -> str:
        return f"dense_undirected_{'_'.join(map(str, self.shapes))}"


class DenseDirected(torch.nn.Module):
    """Dense neural network with label"""

    shapes: list[int]
    net: torch.nn.Sequential

    def __init__(self, shapes: list[int]) -> None:
        super().__init__()
        assert shapes[0] == shapes[-1], "Input and output shapes must be equal"

        self.shapes = shapes

        shapes = list(shapes)
        shapes[0] += 1  # Add label

        layers: list[torch.nn.Module] = []
        for i in range(len(shapes) - 1):
            layers.append(
                torch.nn.Linear(in_features=shapes[i], out_features=shapes[i + 1])
            )
            layers.append(torch.nn.ReLU())

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _, _, w, h = x.shape
        y = y.unsqueeze(-1)
        x = einops.rearrange(x, "b 1 w h -> b (w h)")
        x = torch.cat((x, y), dim=1)
        x = self.net(x)
        x = einops.rearrange(x, "b (w h) -> b 1 w h", w=w, h=h)
        return x

    @override
    def __repr__(self) -> str:
        return f"DenseDirected({self.net})"

    def save_name(self) -> str:
        return f"dense_directed_{'_'.join(map(str, self.shapes))}"
