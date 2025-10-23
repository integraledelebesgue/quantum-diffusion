from typing import Protocol

import torch


class GuidedDenoisingNetwork(Protocol):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...
    def save_name(self) -> str: ...


class UnguidedDenoisingNetwork(Protocol):
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def save_name(self) -> str: ...


DenoisingNetwork = GuidedDenoisingNetwork | UnguidedDenoisingNetwork
