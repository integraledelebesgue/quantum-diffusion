from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import kornia
import pennylane as qml
import pytest
from quantum_image_encoding import FRQI

CAT = Path(__file__).parent / 'data' / 'cat.jpg'


@pytest.fixture
def image() -> jax.Array:
    image = kornia.io.load_image(CAT)
    image = kornia.color.rgb_to_grayscale(image)
    image = kornia.geometry.resize(image, (8, 8))
    image = jax.dlpack.from_dlpack(image, copy=False)
    image = image.flatten()
    return image


@pytest.fixture
def image_batch(image: jax.Array) -> jax.Array:
    return jnp.stack([image for _ in range(16)])


def test_frqi_default_qubit_8x8_single(image: jax.Array) -> None:
    frqi = FRQI((1, 8, 8))
    num_wires = frqi.num_wires()

    device = qml.device('default.qubit', wires=range(num_wires))

    @qml.qnode(device, interface='jax')
    def encode(x):
        frqi.encode(x, list(range(num_wires)))
        return qml.probs(wires=range(num_wires))

    image_encoded = encode(image)
    image_decoded = frqi.decode(image_encoded)

    assert jnp.isclose(image, image_decoded).all().item()


def test_frqi_default_qubit_8x8_batch(image_batch: jax.Array) -> None:
    frqi = FRQI((1, 8, 8))
    num_wires = frqi.num_wires()

    device = qml.device('default.qubit', wires=range(num_wires))

    @partial(jax.vmap, in_axes=0, out_axes=0)
    @qml.qnode(device, interface='jax')
    def encode(x):
        frqi.encode(x, list(range(num_wires)))
        return qml.probs(wires=range(num_wires))

    image_batch_encoded = encode(image_batch)
    image_batch_decoded = frqi.decode(image_batch_encoded)

    assert jnp.isclose(image_batch, image_batch_decoded).all().item()


def test_frqi_lightning_qubit_compiled_8x8(image: jax.Array) -> None:
    frqi = FRQI((1, 8, 8))
    num_wires = frqi.num_wires()

    device = qml.device('lightning.qubit', wires=range(num_wires))

    @qml.qjit
    @qml.qnode(device, interface='jax')
    def encode(x):
        frqi.encode(x, list(range(num_wires)))
        return qml.probs(wires=range(num_wires))

    image_encoded = encode(image)
    image_decoded = frqi.decode(image_encoded)

    assert jnp.isclose(image, image_decoded).all().item()


def test_frqi_lightning_qubit_compiled_8x8_batch(image_batch: jax.Array) -> None:
    frqi = FRQI((1, 8, 8))
    num_wires = frqi.num_wires()

    device = qml.device('lightning.qubit', wires=range(num_wires))

    @partial(jax.vmap, in_axes=0, out_axes=0)
    @qml.qjit
    @qml.qnode(device, interface='jax-jit')
    def encode(x):
        frqi.encode(x, list(range(num_wires)))
        return qml.probs(wires=range(num_wires))

    image_batch_encoded = encode(image_batch)
    image_batch_decoded = frqi.decode(image_batch_encoded)

    assert jnp.isclose(image_batch, image_batch_decoded).all().item()
