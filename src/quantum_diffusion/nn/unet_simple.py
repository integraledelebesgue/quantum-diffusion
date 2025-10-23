import torch
from typing_extensions import override

from .qconv import QConv2d
from .unet import DownBlock, UpBlock, get_label_embedding


class DownBlockS(DownBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pooling: bool,
        kernel_size: int = 3,
        qdepth: int = 3,
    ) -> None:
        super().__init__(in_channels, out_channels, pooling, kernel_size, qdepth)
        self.net = torch.nn.Sequential(
            QConv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                qdepth=qdepth,
                padding=1,
            ),
            torch.nn.BatchNorm2d(self.out_channels),
        )


class UpBlockS(UpBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        qdepth: int = 3,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, qdepth=0)
        self.net = torch.nn.Sequential(
            QConv2d(
                in_channels=2 * out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=1,
                qdepth=qdepth,
            ),
            torch.nn.BatchNorm2d(out_channels),
        )
        self.up_conv = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="bilinear"),
            QConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                qdepth=qdepth,
            ),
        )


class UNetUndirectedS(torch.nn.Module):
    """
    Simplified U-shaped Network for image segmentation.
    Undirected (no labels).
    """

    depth: int
    start_channels: int
    qdepth: int
    down_blocks: torch.nn.ModuleList
    up_blocks: torch.nn.ModuleList
    final_conv: torch.nn.Module

    def __init__(
        self,
        depth: int = 3,
        start_channels: int = 8,
        qdepth: int = 3,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.start_channels = start_channels
        self.qdepth = qdepth
        assert self.depth > 0, "Depth must be greater than 0"
        out_channel = -1  # to suppress warnings about uninitialized variables
        down_blocks = []
        for i in range(self.depth):
            in_channel = 1 if i == 0 else out_channel  # 1 for the first layer
            out_channel = self.start_channels * 2**i
            pooling = i < depth - 1  # no pooling in the last layer
            down_blocks.append(
                DownBlockS(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    pooling=pooling,
                    kernel_size=3,
                    qdepth=self.qdepth,
                )
            )

        up_blocks = []
        for _ in range(self.depth - 1):
            in_channel = out_channel  # set the input channel to the output channel of the previous layer
            out_channel = out_channel // 2
            up_blocks.append(
                UpBlockS(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=3,
                    qdepth=self.qdepth,
                )
            )

        self.down_blocks = torch.nn.ModuleList(down_blocks)
        self.up_blocks = torch.nn.ModuleList(up_blocks)
        self.final_conv = QConv2d(
            in_channels=out_channel,
            out_channels=1,
            kernel_size=1,
            padding=0,
            qdepth=self.qdepth,
        )

    @override
    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        encoder_outputs = []  # list of skip connections
        for _, block in enumerate(self.down_blocks):
            x, before_pool = block.forward(x)
            encoder_outputs.append(before_pool)

        for i, block in enumerate(self.up_blocks):
            skip = encoder_outputs[-(i + 2)]
            x = block(skip, x)

        x = self.final_conv(x)

        return x

    def extra_repr(self) -> str:
        return f"depth={self.depth}"

    def save_name(self) -> str:
        return f"unet_s_undirected_d{self.depth}_s{self.start_channels}_d{self.qdepth}"


class UnetDirectedS(torch.nn.Module):
    """
    Simplified U-shaped Network for image segmentation.
    Directed (with labels).
    """

    depth: int
    start_channels: int
    qdepth: int
    down_blocks: torch.nn.ModuleList
    up_blocks: torch.nn.ModuleList
    final_conv: torch.nn.Module

    def __init__(
        self,
        depth: int = 3,
        start_channels: int = 8,
        qdepth: int = 3,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.start_channels = start_channels
        self.qdepth = qdepth
        assert self.depth > 0, "Depth must be greater than 0"
        out_channel = -1  # to suppress warnings about uninitialized variables
        down_blocks = []
        for i in range(self.depth):
            in_channel = 1 if i == 0 else out_channel  # 1 for the first layer
            out_channel = self.start_channels * 2**i
            pooling = i < depth - 1  # no pooling in the last layer
            down_blocks.append(
                DownBlockS(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    pooling=pooling,
                    kernel_size=3,
                    qdepth=self.qdepth,
                )
            )

        up_blocks = []
        for _ in range(self.depth - 1):
            in_channel = out_channel  # set the input channel to the output channel of the previous layer
            out_channel = out_channel // 2
            up_blocks.append(
                UpBlockS(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=3,
                    qdepth=self.qdepth,
                )
            )

        self.down_blocks = torch.nn.ModuleList(down_blocks)
        self.up_blocks = torch.nn.ModuleList(up_blocks)
        self.final_conv = QConv2d(
            in_channels=out_channel,
            out_channels=1,
            kernel_size=1,
            padding=0,
            qdepth=self.qdepth,
        )

    @override
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mask = get_label_embedding(y, x.shape[2], x.shape[3])
        masked_x = x + mask

        encoder_outputs = []  # list of skip connections
        for _, block in enumerate(self.down_blocks):
            masked_x, before_pool = block(masked_x)
            encoder_outputs.append(before_pool)

        for i, block in enumerate(self.up_blocks):
            skip = encoder_outputs[-(i + 2)]
            masked_x = block(skip, masked_x)

        masked_x = self.final_conv(masked_x)

        return masked_x

    def extra_repr(self) -> str:
        return f"depth={self.depth}"

    def save_name(self) -> str:
        return f"unet_s_directed_d{self.depth}_s{self.start_channels}_d{self.qdepth}"
