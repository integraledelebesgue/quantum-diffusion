from math import ceil, log2

import jax
import jax.numpy as jnp
import pennylane as qml

from .interface import Encoding

__all__ = ['RealKet']


class RealKet(Encoding):
    """Encodes the image using the Real Ket method.
    The encoded state is normalized, which implies that the decoded image is normalized, too.
    Uses MPS state preparation, and thus requires auxillary qubits.
    """

    image_shape: tuple[int, int, int]
    max_bond_dim: int | None
    _encode_permutation: list[int]
    _decode_permutation: list[int]

    def __init__(self, image_shape: tuple[int, int, int], max_bond_dim: int | None = None) -> None:
        self.image_shape = image_shape
        self.max_bond_dim = max_bond_dim
        self._encode_permutation, self._decode_permutation = self._compute_permutations()

        if max_bond_dim is not None:
            assert max_bond_dim >= 1, 'max_bond_dim must be >= 1'
            assert (max_bond_dim & (max_bond_dim - 1)) == 0, 'max_bond_dim must be a power of 2'

    def num_wires(self) -> int:
        """Compute the total number of wires required by the encoding.
        It is a sum of state wires (used for representation) and auxillary work wires required by MPS state preparation.
        """

        return self.num_state_wires() + self._num_work_wires()

    def num_state_wires(self) -> int:
        """Compute the number of wires required to represent the image itself,
        excluding the auxillary qubits required by MPS state preparation.
        """

        _, height, width = self.image_shape
        n_rows = int(ceil(log2(height))) if height > 1 else 0
        n_columns = int(ceil(log2(width))) if width > 1 else 0

        return max(1, n_rows + n_columns)

    def _num_work_wires(self) -> int:
        if self.max_bond_dim is not None:
            max_bd = self.max_bond_dim
        else:
            n = self.num_state_wires()
            max_bd = 1 << (n // 2)

        return max(1, int(ceil(log2(max_bd)))) if max_bd > 1 else 1

    def encode(self, image: jax.Array, wires: list[int]) -> None:
        assert image.ndim == 1, 'MPS preparation does not support batched input'

        n_state = self.num_state_wires()
        state_wires = wires[:n_state]
        work_wires = wires[n_state:]

        image_padded = jnp.append(image, 0.0)
        image_z_ordered = image_padded[jnp.array(self._encode_permutation)]

        norm = jnp.linalg.norm(image_z_ordered)
        if norm > 0:
            image_z_ordered = image_z_ordered / norm

        mps = _mps(image_z_ordered, n_state, self.max_bond_dim)

        qml.MPSPrep(mps, wires=state_wires, work_wires=work_wires, right_canonicalize=True)

    def decode(self, probabilities: jax.Array) -> jax.Array:
        pixel_probs = probabilities[..., jnp.array(self._decode_permutation)]
        return jnp.sqrt(pixel_probs)

    def _compute_permutations(self) -> tuple[list[int], list[int]]:
        _, height, width = self.image_shape
        n_rows = int(ceil(log2(height))) if height > 1 else 0
        n_columns = int(ceil(log2(width))) if width > 1 else 0
        total = 1 << (n_rows + n_columns)

        encode_perm: list[int] = []
        for z in range(total):
            r, c = _morton_code_to_position(z, n_rows, n_columns)
            if r < height and c < width:
                encode_perm.append(r * width + c)
            else:
                encode_perm.append(-1)

        decode_perm: list[int] = []
        for r in range(height):
            for c in range(width):
                decode_perm.append(_position_to_morton_code(r, c, n_rows, n_columns))

        return encode_perm, decode_perm


def _position_to_morton_code(row: int, column: int, n_rows: int, n_columns: int) -> int:
    """Translate the row-column position to a corresponding index in Morton code.
    Uses least-significant-bit-first layout (c0, r0, c1, r1, ...).
    """

    result = 0
    bit_pos = 0

    for i in range(max(n_rows, n_columns)):
        if i < n_columns:
            result |= ((column >> i) & 1) << bit_pos
            bit_pos += 1
        if i < n_rows:
            result |= ((row >> i) & 1) << bit_pos
            bit_pos += 1

    return result


def _morton_code_to_position(z: int, n_rows: int, n_columns: int) -> tuple[int, int]:
    """Extract row and col from a Morton code (Z-order index)."""

    row = 0
    col = 0
    bit_pos = 0

    for i in range(max(n_rows, n_columns)):
        if i < n_columns:
            col |= ((z >> bit_pos) & 1) << i
            bit_pos += 1
        if i < n_rows:
            row |= ((z >> bit_pos) & 1) << i
            bit_pos += 1

    return row, col


def _mps(state: jax.Array, n_qubits: int, max_bond_dim: int | None = None) -> list[jax.Array]:
    """Decompose the state vector into MPS. Requires the state vector to be normalized."""

    assert jnp.linalg.norm(state) == 1.0, 'MPS decomposition requires normalized state.'

    remaining = state.reshape(2, -1)
    decomposition: list[jax.Array] = []
    previous_bond = 1

    for i in range(n_qubits - 1):
        u, s, vh = jnp.linalg.svd(remaining, full_matrices=False)
        bond = len(s)

        if max_bond_dim is not None:
            bond = min(bond, max_bond_dim)
            u = u[:, :bond]
            s = s[:bond]
            vh = vh[:bond, :]

        if i == 0:
            decomposition.append(u)
        else:
            decomposition.append(u.reshape(previous_bond, 2, bond))

        remaining = jnp.diag(s) @ vh
        if i < n_qubits - 2:
            remaining = remaining.reshape(bond * 2, -1)

        previous_bond = bond

    decomposition.append(remaining.reshape(previous_bond, 2))
    return decomposition
