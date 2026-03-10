from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import kornia
import pennylane as qml
import pytest
from quantum_image_encoding.frqi import FRQI, MinimizationMethod

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


@pytest.mark.parametrize('gray_code', [False, True], ids=['linear', 'gray_code'])
def test_frqi_default_qubit_8x8_single(image: jax.Array, gray_code: bool) -> None:
    frqi = FRQI((1, 8, 8), gray_code=gray_code)
    num_wires = frqi.num_wires()

    device = qml.device('default.qubit', wires=range(num_wires))

    @qml.qnode(device, interface='jax')
    def encode(x):
        frqi.encode(x, list(range(num_wires)))
        return qml.probs(wires=range(num_wires))

    image_encoded = encode(image)
    image_decoded = frqi.decode(image_encoded)

    assert jnp.allclose(image, image_decoded, atol=1e-6)


@pytest.mark.parametrize('gray_code', [False, True], ids=['linear', 'gray_code'])
def test_frqi_default_qubit_8x8_batch(image_batch: jax.Array, gray_code: bool) -> None:
    frqi = FRQI((1, 8, 8), gray_code=gray_code)
    num_wires = frqi.num_wires()

    device = qml.device('default.qubit', wires=range(num_wires))

    @partial(jax.vmap, in_axes=0, out_axes=0)
    @qml.qnode(device, interface='jax')
    def encode(x):
        frqi.encode(x, list(range(num_wires)))
        return qml.probs(wires=range(num_wires))

    image_batch_encoded = encode(image_batch)
    image_batch_decoded = frqi.decode(image_batch_encoded)

    assert jnp.allclose(image_batch, image_batch_decoded, atol=1e-6)


@pytest.mark.parametrize('gray_code', [False, True], ids=['linear', 'gray_code'])
def test_frqi_lightning_qubit_compiled_8x8(image: jax.Array, gray_code: bool) -> None:
    frqi = FRQI((1, 8, 8), gray_code=gray_code)
    num_wires = frqi.num_wires()

    device = qml.device('lightning.qubit', wires=range(num_wires))

    @qml.qjit
    @qml.qnode(device, interface='jax')
    def encode(x):
        frqi.encode(x, list(range(num_wires)))
        return qml.probs(wires=range(num_wires))

    image_encoded = encode(image)
    image_decoded = frqi.decode(image_encoded)

    assert jnp.allclose(image, image_decoded, atol=1e-6)


@pytest.mark.parametrize('gray_code', [False, True], ids=['linear', 'gray_code'])
def test_frqi_lightning_qubit_compiled_8x8_batch(image_batch: jax.Array, gray_code: bool) -> None:
    frqi = FRQI((1, 8, 8), gray_code=gray_code)
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

    assert jnp.allclose(image_batch, image_batch_decoded, atol=1e-6)


def test_frqi_cheat_state_preparation_default_qubit_8x8_single(image: jax.Array) -> None:
    frqi = FRQI((1, 8, 8))
    num_wires = frqi.num_wires()

    device = qml.device('default.qubit', wires=range(num_wires))

    @qml.qnode(device, interface='jax')
    def encode(x):
        frqi.cheat_state_preparation(x, list(range(num_wires)))
        return qml.probs(wires=range(num_wires))

    image_encoded = encode(image)
    image_decoded = frqi.decode(image_encoded)

    assert jnp.allclose(image, image_decoded, atol=1e-6)


@pytest.mark.parametrize(
    'kwargs',
    [
        {},
        {'gray_code': True},
        {'minimization_method': 'quine-mccluskey'},
        {'minimization_method': 'quine-mccluskey', 'gray_code': True},
        {'minimization_method': 'espresso'},
        {'minimization_method': 'espresso', 'gray_code': True},
    ],
    ids=['linear', 'gray_code', 'qm', 'qm_sorted', 'espresso', 'espresso_sorted'],
)
def test_frqi_state_matches_cheat(image: jax.Array, kwargs: dict) -> None:
    frqi = FRQI((1, 8, 8), **kwargs)
    num_wires = frqi.num_wires()

    device = qml.device('default.qubit', wires=range(num_wires))

    @qml.qnode(device, interface='jax')
    def encode_circuit(x):
        frqi.encode(x, list(range(num_wires)))
        return qml.state()

    @qml.qnode(device, interface='jax')
    def cheat_circuit(x):
        frqi.cheat_state_preparation(x, list(range(num_wires)))
        return qml.state()

    state_encode = encode_circuit(image)
    state_cheat = cheat_circuit(image)

    assert jnp.allclose(state_encode, state_cheat, atol=1e-6)


@pytest.mark.parametrize('gray_code', [False, True], ids=['linear', 'sorted'])
@pytest.mark.parametrize('minimization_method', ['quine-mccluskey', 'espresso'])
def test_frqi_minimization_methodd_default_qubit_8x8_single(
    image: jax.Array,
    gray_code: bool,
    minimization_method: MinimizationMethod,
) -> None:
    frqi = FRQI((1, 8, 8), minimization_method=minimization_method, gray_code=gray_code)
    num_wires = frqi.num_wires()

    device = qml.device('default.qubit', wires=range(num_wires))

    @qml.qnode(device, interface='jax')
    def encode(x):
        frqi.encode(x, list(range(num_wires)))
        return qml.probs(wires=range(num_wires))

    image_encoded = encode(image)
    image_decoded = frqi.decode(image_encoded)

    assert jnp.allclose(image, image_decoded, atol=1e-6)
