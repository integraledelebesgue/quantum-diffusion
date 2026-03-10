from pathlib import Path

import jax
import jax.numpy as jnp
import kornia
import pennylane as qml
import pytest
from quantum_image_encoding import Amplitude, RealKet

CAT = Path(__file__).parent / 'data' / 'cat.jpg'


@pytest.fixture
def image() -> jax.Array:
    image = kornia.io.load_image(CAT)
    image = kornia.color.rgb_to_grayscale(image)
    image = kornia.geometry.resize(image, (8, 8))
    image = jax.dlpack.from_dlpack(image, copy=False)
    image = image.flatten()
    return image


def test_real_ket_wire_counts() -> None:
    rk = RealKet((1, 8, 8))
    assert rk.num_state_wires() == 6
    assert rk.num_wires() == 9

    rk4 = RealKet((1, 8, 8), max_bond_dim=4)
    assert rk4.num_state_wires() == 6
    assert rk4.num_wires() == 8

    rk2 = RealKet((1, 8, 8), max_bond_dim=2)
    assert rk2.num_state_wires() == 6
    assert rk2.num_wires() == 7


def test_real_ket_default_qubit_8x8_single(image: jax.Array) -> None:
    real_ket = RealKet((1, 8, 8))
    num_wires = real_ket.num_wires()
    num_state_wires = real_ket.num_state_wires()

    device = qml.device('default.qubit', wires=range(num_wires))

    @qml.qnode(device, interface='jax')
    def encode(x):
        real_ket.encode(x, list(range(num_wires)))
        return qml.probs(wires=range(num_state_wires))

    image_encoded = encode(image)
    image_decoded = real_ket.decode(image_encoded)

    image_normalized = image / jnp.linalg.norm(image)
    assert jnp.allclose(image_normalized, image_decoded, atol=1e-6)


def test_real_ket_state_matches_amplitude(image: jax.Array) -> None:
    """Verify that RealKet and Amplitude produce equivalent states (same amplitudes, different ordering)."""

    real_ket = RealKet((1, 8, 8))
    amplitude = Amplitude((1, 8, 8))

    num_wires = real_ket.num_wires()
    num_state_wires = real_ket.num_state_wires()

    device_rk = qml.device('default.qubit', wires=range(num_wires))
    device_amp = qml.device('default.qubit', wires=range(amplitude.num_wires()))

    @qml.qnode(device_rk, interface='jax')
    def real_ket_circuit(x):
        real_ket.encode(x, list(range(num_wires)))
        return qml.state()

    @qml.qnode(device_amp, interface='jax')
    def amplitude_circuit(x):
        amplitude.encode(x, list(range(amplitude.num_wires())))
        return qml.state()

    state_real_ket = real_ket_circuit(image)
    state_amplitude = amplitude_circuit(image)

    # RealKet state lives in a larger Hilbert space (work wires).
    # Work wires (LSBs in PennyLane convention) are |0>, so take every 2^n_work-th entry.
    n_work = num_wires - num_state_wires
    stride = 2**n_work
    state_real_ket_reduced = state_real_ket[::stride]

    state_amplitude_sorted = jnp.sort(jnp.abs(state_amplitude))
    state_real_ket_reduced_sorted = jnp.sort(jnp.abs(state_real_ket_reduced))

    assert jnp.allclose(state_amplitude_sorted, state_real_ket_reduced_sorted, atol=1e-6)


@pytest.mark.parametrize('max_bond_dim', [2, 4])
def test_real_ket_truncated(image: jax.Array, max_bond_dim: int) -> None:
    real_ket = RealKet((1, 8, 8), max_bond_dim=max_bond_dim)
    num_wires = real_ket.num_wires()
    num_state_wires = real_ket.num_state_wires()

    device = qml.device('default.qubit', wires=range(num_wires))

    @qml.qnode(device, interface='jax')
    def encode(x):
        real_ket.encode(x, list(range(num_wires)))
        return qml.probs(wires=range(num_state_wires))

    probs = encode(image)
    decoded = real_ket.decode(probs)

    image_normalized = image / jnp.linalg.norm(image)
    mse = jnp.mean((image_normalized - decoded) ** 2)
    assert mse < 0.1
