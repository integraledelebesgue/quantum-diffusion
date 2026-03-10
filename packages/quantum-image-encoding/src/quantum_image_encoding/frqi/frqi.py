from math import ceil, log2, sqrt
from typing import Literal

import jax
import jax.numpy as jnp
import pennylane as qml

from ..interface import Encoding
from .minimize import minimize_minterms, sort_esop_terms
from .product_term import ProductTerm

__all__ = ['FRQI', 'MinimizationMethod']

MinimizationMethod = Literal['quine-mccluskey', 'espresso']


class FRQI(Encoding):
    """
    Encodes the image in the circuit using `FRQI` method in the circuit, uses `wires`.
    The first wire is used to encode the color, the rest represent the position.
    Requires `ceil(log2(height * width)) + 1` qubits.
    """

    image_shape: tuple[int, int, int]
    gray_code: bool
    minimization_method: MinimizationMethod | None

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        gray_code: bool = False,
        minimization_method: MinimizationMethod | None = None,
    ) -> None:
        self.image_shape = image_shape
        self.gray_code = gray_code
        self.minimization_method = minimization_method

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

        # Encoding 2 * arcsin(alpha) instead of alpha leads to a quantum state
        # in which the probability amplitude of each state |1...>
        # (i.e. some position + the color qubit equal to 1)
        # is equal to the normalized color.
        angles = 2 * jnp.arcsin(image)

        if self.minimization_method is None:
            items = _build_angles_with_positions_naive(angles, num_position_wires, self.gray_code)
        else:
            items = _build_angles_with_positions_minimized(angles, num_position_wires, self.minimization_method, self.gray_code)

        self._apply_encoding(items, color_wire, position_wires, num_position_wires)

    def _apply_encoding(
        self,
        items: list[tuple[jax.Array, ProductTerm]],
        color_wire: int,
        position_wires: list[int],
        num_position_wires: int,
    ) -> None:
        current_negations = [False for _ in range(num_position_wires)]

        for angle, term in items:
            control_wires: list[int] = []

            for i in range(num_position_wires):
                bit_index = num_position_wires - 1 - i

                if term.mask & (1 << bit_index):
                    needs_negation = not bool((term.value >> bit_index) & 1)

                    if current_negations[i] != needs_negation:
                        qml.PauliX(position_wires[i])
                        current_negations[i] = needs_negation

                    control_wires.append(position_wires[i])

            if len(control_wires) == 0:  # Apply one color to all positions
                qml.RY(angle, wires=color_wire)  # type: ignore
            else:
                controlled_ry = qml.ctrl(qml.RY, control=control_wires)
                controlled_ry(angle, wires=color_wire)

        # Undo remaining X inversions to restore the initial states of the position wires
        for i, is_inverted in enumerate(current_negations):
            if is_inverted:
                qml.PauliX(position_wires[i])

    def cheat_state_preparation(self, image: jax.Array, wires: list[int]) -> None:
        """Analytically compute the FRQI state vector and prepare it via Möttönen decomposition."""

        assert image.ndim == 1, 'State preparation does not support batched input'
        size = image.size

        num_position_wires = int(ceil(log2(size)))
        num_positions = 1 << num_position_wires

        assert len(wires) >= num_position_wires + 1, (
            f'FRQI encoding requires at least {num_position_wires + 1} wires but only {len(wires)} were supplied'
        )

        angles = jnp.arcsin(image)
        angles_padded = jnp.zeros(num_positions).at[:size].set(angles)

        color_0_components = jnp.cos(angles_padded)
        color_1_components = jnp.sin(angles_padded)

        norm = 1.0 / sqrt(num_positions)
        state = norm * jnp.concat((color_0_components, color_1_components), axis=-1)

        qml.MottonenStatePreparation(state, wires=wires)

    def decode(self, probabilities: jax.Array) -> jax.Array:
        """Decode the probabilities obtained by measuring the state that
        has been constructed using FRQI algorithm."""

        *_, num_states = probabilities.shape

        # First qubit encodes the color => the rest encode the position
        num_positions = num_states // 2
        state_normalization_factor = sqrt(num_positions)

        probability_amplitudes = jnp.sqrt(probabilities[..., num_positions:])

        return jnp.clip(probability_amplitudes * state_normalization_factor, 0.0, 1.0)


def _build_angles_with_positions_naive(
    angles: jax.Array,
    num_position_wires: int,
    gray_code: bool,
) -> list[tuple[jax.Array, ProductTerm]]:
    num_positions = 1 << num_position_wires
    all_bits = (1 << num_position_wires) - 1

    if gray_code:
        order = [g for g in (i ^ (i >> 1) for i in range(num_positions)) if g < len(angles)]
    else:
        order = range(len(angles))

    return [(angles[pos], ProductTerm(all_bits, pos, num_position_wires)) for pos in order]


def _build_angles_with_positions_minimized(
    angles: jax.Array,
    num_position_wires: int,
    minimize_method: MinimizationMethod,
    gray_code: bool,
) -> list[tuple[jax.Array, ProductTerm]]:
    angle_to_positions: dict[float, list[int]] = {}
    for position, angle in enumerate(angles):
        angle = float(angle)

        if angle == 0.0:
            continue

        angle_to_positions.setdefault(angle, []).append(position)

    items: list[tuple[float, ProductTerm]] = []
    for angle, positions in angle_to_positions.items():
        for term in minimize_minterms(positions, num_position_wires, minimize_method):
            items.append((angle, term))

    if gray_code:
        items = _sort_items(items)

    return [(jnp.array(angle), term) for angle, term in items]


def _sort_items(
    items: list[tuple[float, ProductTerm]],
) -> list[tuple[float, ProductTerm]]:
    """Reorder items for minimal X-gate transitions between consecutive terms."""
    sorted_terms = sort_esop_terms([term for _, term in items])

    remaining = list(range(len(items)))
    result: list[tuple[float, ProductTerm]] = []
    for term in sorted_terms:
        for j, idx in enumerate(remaining):
            if items[idx][1] == term:
                result.append(items[idx])
                remaining.pop(j)
                break

    return result
