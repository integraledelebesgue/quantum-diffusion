from collections.abc import Iterable

import jax
import jax.numpy as jnp
import pennylane as qml


def encode(x: jax.Array, wires: Iterable[int]) -> None:
    qml.AmplitudeEmbedding(
        features=x,
        wires=wires,
        pad_with=0.1,
        normalize=True,
    )


def decode(probabilities: jax.Array) -> jax.Array:
    *_, num_pixels = probabilities.shape
    out = probabilities * num_pixels
    out = jnp.clip(out, 0.0, 1.0)
    return out
