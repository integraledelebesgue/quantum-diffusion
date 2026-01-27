import jax
import jax.numpy as jnp
from flax import nnx
from quantum_diffusion.neural_networks.pqc import PQCGuided


def test_pqc_guided_single_image_forward() -> None:
    network = PQCGuided(
        num_layers=1,
        input_shape=(1, 8, 8),
        rngs=nnx.Rngs(0),
    )

    input = jnp.zeros((1, 8, 8), dtype=jnp.float32)
    label = jnp.array((1,), dtype=jnp.float32)
    output = network(input, label)

    assert output.shape == (1, 8, 8)


def test_pqc_guided_batch_forward() -> None:
    network = PQCGuided(
        num_layers=1,
        input_shape=(1, 8, 8),
        rngs=nnx.Rngs(0),
    )

    input = jnp.zeros((16, 1, 8, 8), dtype=jnp.float32)
    label = jnp.zeros((16,), dtype=jnp.float32)
    output = network(input, label)

    assert output.shape == (16, 1, 8, 8)


def test_pqc_guided_single_image_gradient() -> None:
    network = PQCGuided(
        num_layers=1,
        input_shape=(1, 8, 8),
        rngs=nnx.Rngs(0),
    )

    @nnx.jit
    def mse(network: PQCGuided) -> jax.Array:
        output = network(input, label)
        return jnp.mean((input - output) ** 2)

    input = jnp.zeros((1, 8, 8), dtype=jnp.float32)
    label = jnp.array((1,), dtype=jnp.float32)
    loss, gradient = nnx.value_and_grad(mse)(network)

    assert isinstance(loss, jax.Array)
    assert isinstance(gradient, nnx.State)


def test_pqc_guided_batch_gradient() -> None:
    network = PQCGuided(
        num_layers=1,
        input_shape=(1, 8, 8),
        rngs=nnx.Rngs(0),
    )

    @nnx.jit
    def mse(network: PQCGuided) -> jax.Array:
        output = network(input, label)
        return jnp.sum((input - output) ** 2)

    input = jnp.zeros((16, 1, 8, 8), dtype=jnp.float32)
    label = jnp.zeros((16,), dtype=jnp.float32)
    loss, gradient = nnx.value_and_grad(mse)(network)

    assert isinstance(loss, jax.Array)
    assert isinstance(gradient, nnx.State)
