import jax
import jax.numpy as jnp
import quantum_diffusion
from flax import nnx
from quantum_image_encoding import amplitude


@jax.jit
def mse(image_original: jax.Array, image_reconstructed: jax.Array) -> jax.Array:
    return jnp.sum((image_original - image_reconstructed) ** 2)


def test_pqc_guided_single_image_forward() -> None:
    network = quantum_diffusion.neural_networks.pqc.PQCGuided(
        num_layers=1,
        input_shape=(1, 8, 8),
        encode=amplitude.encode,
        decode=amplitude.decode,
        rngs=nnx.Rngs(0),
    )

    input = jnp.zeros((1, 8, 8), dtype=jnp.float32)
    label = jnp.array((1,), dtype=jnp.float32)
    output = network(input, label)

    assert output.shape == (1, 8, 8)


def test_pqc_guided_batch_forward() -> None:
    network = quantum_diffusion.neural_networks.pqc.PQCGuided(
        num_layers=1,
        input_shape=(1, 8, 8),
        encode=amplitude.encode,
        decode=amplitude.decode,
        rngs=nnx.Rngs(0),
    )

    input = jnp.zeros((16, 1, 8, 8), dtype=jnp.float32)
    label = jnp.zeros((16,), dtype=jnp.float32)
    output = network(input, label)

    assert output.shape == (16, 1, 8, 8)


def test_pqc_guided_single_image_gradient() -> None:
    network = quantum_diffusion.neural_networks.pqc.PQCGuided(
        num_layers=1,
        input_shape=(1, 8, 8),
        encode=amplitude.encode,
        decode=amplitude.decode,
        rngs=nnx.Rngs(0),
    )

    input = jnp.zeros((1, 8, 8), dtype=jnp.float32)
    label = jnp.array((1,), dtype=jnp.float32)
    output = network(input, label)
    gradient = jax.grad(mse)(input, output)

    assert gradient.shape == (1, 8, 8)


def test_pqc_guided_batch_gradient() -> None:
    network = quantum_diffusion.neural_networks.pqc.PQCGuided(
        num_layers=1,
        input_shape=(1, 8, 8),
        encode=amplitude.encode,
        decode=amplitude.decode,
        rngs=nnx.Rngs(0),
    )

    input = jnp.zeros((16, 1, 8, 8), dtype=jnp.float32)
    label = jnp.zeros((16,), dtype=jnp.float32)
    output = network(input, label)
    gradient = jax.grad(mse)(input, output)

    assert gradient.shape == (16, 1, 8, 8)
