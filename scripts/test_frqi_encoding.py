import pathlib

import jax
import jax.numpy as jnp
import kornia
import pennylane as qml
import torch
from quantum_image_encoding import frqi


def tensor_to_array(x: torch.Tensor) -> jax.Array:
    return jax.dlpack.from_dlpack(x, copy=False)


def array_to_tensor(x: jax.Array) -> torch.Tensor:
    return torch.from_dlpack(x.__dlpack__())


SHAPE = (8, 8)


def main() -> None:
    num_wires = frqi.num_wires(SHAPE)
    device = qml.device('lightning.qubit', wires=num_wires)

    @qml.qjit
    @qml.set_shots(shots=None)
    @qml.qnode(device, interface='jax', diff_method='adjoint')
    def encode(x):
        frqi.encode(x, list(range(num_wires)))
        return qml.probs(wires=range(num_wires))

    @jax.jit
    def mse(image_original: jax.Array, image_reconstructed: jax.Array) -> jax.Array:
        return jnp.sum((image_original - image_reconstructed) ** 2)

    data_directory = pathlib.Path('data')

    image = kornia.io.load_image(data_directory / 'koteczek.jpg')
    image = kornia.color.rgb_to_grayscale(image)
    image = kornia.geometry.resize(image, SHAPE)
    image = tensor_to_array(image).reshape(1, -1)

    image_encoded = encode(image)
    image_decoded = frqi.decode(image_encoded)

    # Check if the circuit is properly differentiated:
    _grad = jax.grad(mse)(image, image_decoded)

    image_decoded = image_decoded.reshape(1, *SHAPE) * 255
    image_decoded = image_decoded.astype(jnp.uint8)
    image_decoded = array_to_tensor(image_decoded)

    kornia.io.write_image(data_directory / 'koteczek_decoded.jpg', image_decoded)


if __name__ == '__main__':
    main()
