from collections.abc import Callable
from functools import partial
from math import ceil, log2

import einops
import jax
import jax.numpy as jnp
import pennylane as qml
from flax import nnx

__all__ = ['PQCGuided']


class PQCGuided(nnx.Module):
    num_wires: int
    num_layers: int
    num_pixels: int
    input_shape: tuple[int, int, int]

    encode: Callable[[jax.Array, list[int]], jax.Array]
    decode: Callable[[jax.Array], jax.Array]

    weights: nnx.Param[jax.Array]
    circuit: Callable[[jax.Array, jax.Array, jax.Array], jax.Array]

    def __init__(
        self,
        num_layers: int,
        input_shape: tuple[int, int, int],
        encode: Callable[[jax.Array, list[int]], jax.Array],
        decode: Callable[[jax.Array], jax.Array],
        rngs: nnx.Rngs,
    ) -> None:
        num_channels, height, width = input_shape
        assert num_channels == 1

        self.num_wires = int(ceil(log2(height * width))) + 1
        self.num_layers = num_layers
        self.num_pixels = height * width
        self.input_shape = input_shape

        weights_shape = qml.StronglyEntanglingLayers.shape(num_layers, self.num_wires)
        weights_initializer = nnx.initializers.normal()
        self.weights = nnx.Param(weights_initializer(rngs.params(), weights_shape) * 0.4)

        self.encode = encode
        self.decode = decode

        device = qml.device('default.qubit', wires=self.num_wires)

        @partial(jax.vmap, in_axes=(0, 0, None), out_axes=0)
        # @qml.qjit
        # @qml.transforms.broadcast_expand
        # @qml.set_shots(shots=None)
        @qml.qnode(device, interface='jax', diff_method='best')
        def circuit(x, label, weights):
            encode(x, list(range(self.num_wires - 1)))
            qml.RX(phi=label, wires=self.num_wires - 1)
            qml.StronglyEntanglingLayers(
                weights=weights,
                wires=range(self.num_wires),
            )
            return qml.probs(wires=range(self.num_wires))

        self.circuit = circuit

    def __call__(self, image: jax.Array, label: jax.Array) -> jax.Array:
        *_, num_channels, height, width = image.shape
        assert (num_channels, height, width) == self.input_shape

        image, packed_shape = einops.pack([image], '* c h w')
        image = einops.rearrange(image, 'b c h w -> b (c h w)')

        weights_remapped = jnp.pi * jnp.tanh(self.weights.value)

        probabilities = self.circuit(image, label, weights_remapped)
        probabilities = probabilities[..., ::2]  # Take probabilities of |...0> (label qubit = 0)

        out = self.decode(probabilities)
        out = einops.rearrange(out, 'b (c h w) -> b c h w', c=num_channels, h=height, w=width)
        [out] = einops.unpack(out, packed_shape, '* c h w')

        return out
