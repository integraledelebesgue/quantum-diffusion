from math import ceil, log2

import jax
import jax.numpy as jnp
import pennylane as qml

from .interface import Encoding


class Amplitude(Encoding):
    image_shape: tuple[int, int, int]
    normalize: bool
    pad_with: float

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        normalize: bool = True,
        pad_with: float = 0.0,
    ) -> None:
        self.image_shape = image_shape
        self.normalize = normalize
        self.pad_with = pad_with

    def num_wires(self) -> int:
        _, height, width = self.image_shape
        return int(ceil(log2(height * width)))

    def encode(self, image: jax.Array, wires: list[int]) -> None:
        qml.AmplitudeEmbedding(
            features=image,
            wires=wires,
            pad_with=self.pad_with,
            normalize=self.normalize,
        )

    def decode(self, probabilities: jax.Array) -> jax.Array:
        *_, num_pixels = probabilities.shape
        out = probabilities * num_pixels
        out = jnp.clip(out, 0.0, 1.0)
        return out
