import math
import warnings
from typing import Any

import einops
import pennylane as qml
import qw_map
import torch
from typing_extensions import Self, override


class QConv2dFast(torch.nn.Module):
    """Fastest version of QConv2d. Less Memory efficient."""

    in_channels: int
    out_channels: int
    kernel_size: tuple[int, int]
    padding: tuple[int, int]
    wires: int
    qdepth: int

    unfold: torch.nn.Unfold
    weights: torch.nn.Parameter

    qnode: qml.QNode

    sample_qnode: qml.QNode | None
    sample_matrix: torch.Tensor | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] | int = 3,
        padding: tuple[int, int] | int = 1,
        qdepth: int = 2,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.unfold = torch.nn.Unfold(kernel_size=kernel_size, padding=padding)
        self.qdepth = qdepth

        wires_for_inp = math.ceil(
            math.log2(self.kernel_size[0] * self.kernel_size[1] * in_channels)
        )
        wires_for_out = math.ceil(math.log2(out_channels))
        self.wires = max(wires_for_inp, wires_for_out, 1)

        if self.wires > 10:
            warnings.warn(
                f"Too many wires ({self.wires}). This might cause performance issues."
            )

        template_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=qdepth, n_wires=self.wires
        )

        weights = torch.rand(template_shape, requires_grad=True)
        weights = weights * math.pi - math.pi / 2
        self.weights = torch.nn.Parameter(weights)

        self.qnode = qml.QNode(
            func=self.circuit,
            device=qml.device("default.qubit", wires=self.wires),
            cache=True,
            cachesize=int(1e6),
            interface="torch",
            diff_method="backprop",
        )

        self.sample_qnode = None
        self.sample_matrix = None

    def circuit(self, x: torch.Tensor):
        qml.AmplitudeEmbedding(
            features=x,
            wires=range(self.wires),
            pad_with=0.5,
            normalize=True,
        )
        qml.StronglyEntanglingLayers(qw_map.tanh(self.weights), wires=range(self.wires))
        return qml.probs(wires=range(self.wires))

    def apply_circuit(self, input: torch.Tensor) -> torch.Tensor:
        out = self.qnode(input)
        assert isinstance(out, torch.Tensor)

        # Scale to approximatly [0, 1]
        out = out * out.shape[-1] * 0.5

        # Clamp to [0, 1] to avoid numerical errors
        out = torch.clamp(out, 0.0, 1.0)

        out = out[:, ::2]  # Drop all probabilities of |1>

        # Select the first out_channels probabilities
        out = out[:, : self.out_channels]

        return out

    @override
    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        b, c, h_in, w_in = x.shape
        assert c == self.in_channels, f"Expected {self.in_channels} channels, got {c}"

        h_out = h_in + 2 * self.padding[0] - self.kernel_size[0] + 1  # output height
        w_out = w_in + 2 * self.padding[1] - self.kernel_size[1] + 1  # output width

        x = self.unfold(x)
        x = einops.rearrange(x, "batch channel feat -> (batch feat) channel")
        x = x + 0.1
        x = self.apply_circuit(x)
        x = x.float()

        x = einops.rearrange(
            x,
            "(batch h_out w_out) channel -> batch channel h_out w_out",
            batch=b,
            h_out=h_out,
            w_out=w_out,
        )

        return x

    @override
    def __repr__(self) -> str:
        return f"QConv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, padding={self.padding}, wires={self.wires})"

    def save_name(self) -> str:
        return f"qconv2d_fast_i{self.in_channels}_o{self.out_channels}_q{self.qdepth}"

    @override
    def train(self, mode: bool = True) -> Self:
        super().train(mode)

        if not mode and self.sample_qnode is None:
            # print("Creating sample qnode")
            def _sub_circuit():
                qml.StronglyEntanglingLayers(
                    qw_map.tanh(self.weights), wires=range(self.wires)
                )

            matrix: torch.Tensor = qml.matrix(_sub_circuit, range(self.wires))()
            assert isinstance(matrix, torch.Tensor)
            self.sample_matrix = matrix

            def _sample_circuit(features):
                qml.AmplitudeEmbedding(
                    features=features,
                    wires=range(self.wires),
                    pad_with=0.5,
                    normalize=True,
                )
                qml.QubitUnitary(matrix, wires=range(self.wires))
                return qml.probs(wires=range(self.wires))

            self.sample_qnode = qml.QNode(
                func=_sample_circuit,
                device=qml.device("default.qubit", wires=self.wires),
                cache=True,
                cachesize=int(1e6),
                interface="torch",
                diff_method="backprop",
            )

        if mode:
            self.sample_qnode = None
            self.sample_matrix = None

        return self


class QConv2dMedium(torch.nn.Module):
    """Faster version of QConv2d_SLOW, but still slow. Memory efficient."""

    in_channels: int
    out_channels: int
    kernel_size: tuple[int, int]
    padding: tuple[int, int]
    unfold: torch.nn.Unfold
    qdepth: int
    wires: int
    unfolded_padding_size: tuple[int, int]

    weights: torch.nn.ParameterList
    pad: torch.nn.ConstantPad2d

    _qnode: qml.QNode
    qnode: Any

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] | int = 3,
        padding: tuple[int, int] | int = 1,
        qdepth: int = 2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.unfold = torch.nn.Unfold(
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        self.qdepth = qdepth

        min_wires_inp = math.ceil(math.log2(self.kernel_size[0] * self.kernel_size[1]))
        min_wires_outp = math.ceil(math.log2(out_channels))
        self.wires = max(min_wires_inp, min_wires_outp, 1)

        template_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=qdepth,
            n_wires=self.wires,
        )

        self.weights = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.rand(template_shape, requires_grad=True))
                for _ in range(in_channels)
            ]
        )

        self.pad = torch.nn.ConstantPad2d(
            (self.padding[0], self.padding[0], self.padding[1], self.padding[1]),
            0.01,
        )

        self.unfolded_padding_size = (
            0,
            2**self.wires - self.kernel_size[0] * self.kernel_size[1],
        )

        self._qnode = qml.QNode(
            func=self.circuit,
            device=qml.device("default.qubit", wires=self.wires),
            cache=True,
            cachesize=int(1e6),
            interface="torch",
            diff_method="backprop",
        )

        self.qnode = qml.batch_input(
            self._qnode,
            [i * 2 for i in range(self.in_channels)],
        )

    def circuit(self, x: torch.Tensor):
        for ic in range(self.in_channels):
            qml.MottonenStatePreparation(x[:, ic], wires=range(self.wires))
            qml.StronglyEntanglingLayers(self.weights[ic], wires=range(self.wires))

        return qml.probs(wires=range(self.wires))

    def apply_circuit(self, input: torch.Tensor) -> torch.Tensor:
        out = self.qnode(input)
        assert isinstance(out, torch.Tensor)

        # Scale to approximatly [0, 1]
        out = out * out.shape[-1] * 0.5

        # Clamp to [0, 1] to avoid numerical errors
        out = torch.clamp(out, 0.0, 1.0)

        out = out[..., : self.out_channels]

        return out

    @override
    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        b, c, h_in, w_in = x.shape
        assert c == self.in_channels, f"Expected {self.in_channels} channels, got {c}"

        h_out = h_in + 2 * self.padding[0] - self.kernel_size[0] + 1
        w_out = w_in + 2 * self.padding[1] - self.kernel_size[1] + 1

        x = self.pad(x)
        x = x.unfold(dimension=2, size=self.kernel_size[0], step=1)
        x = x.unfold(dimension=3, size=self.kernel_size[1], step=1)
        x = x.contiguous()
        x = einops.rearrange(x, "b c bh bw k0 k1 -> (b bh bw) c (k0 k1)")
        x = torch.nn.functional.pad(x, self.unfolded_padding_size)
        x = torch.nn.functional.normalize(x, dim=-1, p=2)
        x = self.apply_circuit(x)

        x = einops.rearrange(x, "(b h2 w2) c -> b c h2 w2", b=b, h2=h_out, w2=w_out)

        return x

    @override
    def __repr__(self) -> str:
        return f"QConv2d_MEDIUM({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, padding={self.padding}, wires={self.wires})"

    def save_name(self) -> str:
        return f"qconv2d_medium_i{self.in_channels}_o{self.out_channels}_q{self.qdepth}"


class QConv2dSlow(torch.nn.Module):
    """Very slow convolutional layer. Very memory efficient"""

    in_channels: int
    out_channels: int
    kernel_size: tuple[int, int]
    padding: tuple[int, int]
    unfold: torch.nn.Unfold
    qdepth: int
    wires: int

    weights: torch.nn.ParameterList
    unfolded_pad: torch.nn.ConstantPad1d

    qnode: qml.QNode

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] | int = 3,
        padding: tuple[int, int] | int = 1,
        qdepth: int = 2,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.unfold = torch.nn.Unfold(
            kernel_size=self.kernel_size, padding=self.padding
        )
        self.qdepth = qdepth

        min_wires_inp = math.ceil(math.log2(self.kernel_size[0] * self.kernel_size[1]))
        min_wires_outp = math.ceil(math.log2(out_channels))
        self.wires = max(min_wires_inp, min_wires_outp, 1)

        template_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=qdepth, n_wires=self.wires
        )

        self.weights = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.rand(template_shape, requires_grad=True))
                for _ in range(in_channels)
            ]
        )

        pad_size = 2**self.wires - self.kernel_size[0] * self.kernel_size[1]
        self.unfolded_pad = torch.nn.ConstantPad1d((0, pad_size), 0.01)

        self.qnode = qml.QNode(
            func=self._circuit,
            device=qml.device("default.qubit", wires=self.wires),
            cache=True,
            cachesize=int(1e6),  # Deactivate caching to avoid memory issues
            interface="torch",
            diff_method="backprop",
        )

    def _circuit(self, *args):
        for ic in range(self.in_channels):
            qml.MottonenStatePreparation(args[ic], wires=range(self.wires))
            qml.StronglyEntanglingLayers(self.weights[ic], wires=range(self.wires))

        return qml.probs(wires=range(self.wires))

    def post_process(self, quantum_probs: torch.Tensor) -> torch.Tensor:
        assert quantum_probs.dim() == 1, "Only for unbatched inputs"

        # Scale to approximatly [0, 1]
        quantum_probs = quantum_probs * quantum_probs.shape[-1] * 0.5

        # Clamp to [0, 1] to avoid numerical errors
        quantum_probs = torch.clamp(quantum_probs, 0.0, 1.0)

        quantum_probs = quantum_probs[: self.out_channels]

        return quantum_probs

    @override
    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        b, c, h_in, w_in = x.shape
        assert c == self.in_channels, f"Expected {self.in_channels} channels, got {c}"

        h_out = h_in + 2 * self.padding[0] - self.kernel_size[0] + 1  # output height
        w_out = w_in + 2 * self.padding[1] - self.kernel_size[1] + 1  # output width

        x = self.unfold(x)  # unfold the input
        x = einops.rearrange(
            x, "ub (cin kernel) feat -> (ub feat) cin kernel", cin=self.in_channels
        )
        x = x + 0.01  # TODO: find a better way to avoid zero inputs
        x = self.unfolded_pad(x)
        x = torch.nn.functional.normalize(x, dim=-1, p=2)

        x_out = torch.empty((x.shape[0], self.out_channels), requires_grad=False)

        for ix in range(x.shape[0]):
            circuit_out = self.qnode(*x[ix])
            pp = self.post_process(circuit_out)
            x_out[ix] = pp

        x_out = einops.rearrange(x_out, "(b h w) c -> b c h w", b=b, h=h_out, w=w_out)

        return x_out

    @override
    def __repr__(self) -> str:
        return f"QConv2d_SLOW({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, padding={self.padding}, wires={self.wires})"

    def save_name(self) -> str:
        return f"qconv2d_slow_i{self.in_channels}_o{self.out_channels}_q{self.qdepth}"


QConv2d = QConv2dFast
