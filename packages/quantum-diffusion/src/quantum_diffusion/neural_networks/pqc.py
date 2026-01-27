from collections.abc import Callable

import einops
import jax
import jax.numpy as jnp
import pennylane as qml
import quantum_image_encoding
from flax import nnx

__all__ = ['PQCGuided']


class PQCGuided(nnx.Module):
    num_wires: int
    num_layers: int
    num_pixels: int
    input_shape: tuple[int, int, int]

    encoding: quantum_image_encoding.Encoding

    weights: nnx.Param[jax.Array]
    circuit: Callable[[jax.Array, jax.Array, jax.Array], jax.Array]

    def __init__(
        self,
        num_layers: int,
        input_shape: tuple[int, int, int],
        *,
        device: str = 'default.qubit',
        qjit: bool = False,
        encoding: quantum_image_encoding.Encoding | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        num_channels, height, width = input_shape
        assert num_channels == 1

        if encoding is None:
            encoding = quantum_image_encoding.Amplitude(input_shape)

        self.encoding = encoding
        self.num_wires = encoding.num_wires() + 1
        self.num_layers = num_layers
        self.num_pixels = height * width
        self.input_shape = input_shape

        weights_shape = qml.StronglyEntanglingLayers.shape(num_layers, self.num_wires)
        weights_initializer = nnx.initializers.normal()
        self.weights = nnx.Param(weights_initializer(rngs.params(), weights_shape) * 0.4)

        device = qml.device(device, wires=self.num_wires)

        @qml.qnode(device, interface='jax-jit', diff_method='best')
        def circuit(x, label, weights):
            self.encoding.encode(x, list(range(self.num_wires - 1)))
            qml.RX(phi=2 * jnp.pi * label, wires=self.num_wires - 1)
            qml.StronglyEntanglingLayers(
                weights=weights,
                wires=range(self.num_wires),
            )
            return qml.probs(wires=range(self.num_wires))

        if qjit:
            circuit = qml.qjit(circuit)

        circuit = jax.vmap(circuit, in_axes=(0, 0, None), out_axes=0)

        self.circuit = circuit

    def __call__(self, image: jax.Array, label: jax.Array) -> jax.Array:
        *_, num_channels, height, width = image.shape
        assert (num_channels, height, width) == self.input_shape

        image, packed_shape = einops.pack([image], '* c h w')
        image = einops.rearrange(image, 'b c h w -> b (c h w)')

        weights_remapped = jnp.pi * jnp.tanh(self.weights.value)

        probabilities = self.circuit(image, label, weights_remapped)
        probabilities = probabilities[..., ::2]  # Take probabilities of |...0> (label qubit = 0)

        out = self.encoding.decode(probabilities)
        out = einops.rearrange(out, 'b (c h w) -> b c h w', c=num_channels, h=height, w=width)
        [out] = einops.unpack(out, packed_shape, '* c h w')

        return out
