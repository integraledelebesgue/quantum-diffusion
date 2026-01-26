from math import ceil, log2, sqrt

import jax
import jax.numpy as jnp
import pennylane as qml

from .interface import Encoding

__all__ = ['FRQI']


class FRQI(Encoding):
    """
    Encodes the image in the circuit using `FRQI` method in the circuit, uses `wires`.
    The first wire is used to encode the color, the rest represent the position.
    Requires `ceil(log2(height * width)) + 1` qubits.
    """

    image_shape: tuple[int, int, int]

    def __init__(self, image_shape: tuple[int, int, int]) -> None:
        self.image_shape = image_shape

    def num_wires(self) -> int:
        _, height, width = self.image_shape
        return int(ceil(log2(height * width))) + 1

    def encode(self, image: jax.Array, wires: list[int]) -> None:
        *_, size = image.shape
        num_position_wires = int(ceil(log2(size)))

        assert len(wires) >= num_position_wires + 1, (
            f'FRQI encoding requires at least {num_position_wires + 1} wires but only {len(wires)} were supplied'
        )
        color_wire = wires[0]
        position_wires = wires[1:]

        for wire in position_wires:
            qml.Hadamard(wire)

        for position in range(size):
            position_binary = f'{position:0{num_position_wires}b}'
            position_control_values = [int(bit) for bit in position_binary]

            controlled_rotation = qml.ctrl(
                qml.RY,
                control_values=position_control_values,
                control=position_wires,
            )

            # Encoding 2 * arcsin(alpha) instead of alpha leads to a quantum state
            # in which the probability amplitude of each state |1...>
            # (i.e. some position + the color qubit equal to 1)
            # is equal to the normalized color.
            pixel_value_as_angle = 2 * jnp.arcsin(image[..., position])
            controlled_rotation(pixel_value_as_angle, wires=color_wire)

    def decode(self, probabilities: jax.Array) -> jax.Array:
        """
        Decodes the probabilities obtained by measuring the state that
        has been constructed using FRQI algorithm.
        """

        *_, num_states = probabilities.shape

        # First qubit encodes the color => the rest encode the position
        num_positions = num_states // 2
        state_normalization_factor = sqrt(num_positions)

        return jnp.sqrt(probabilities[..., num_positions:]) * state_normalization_factor
