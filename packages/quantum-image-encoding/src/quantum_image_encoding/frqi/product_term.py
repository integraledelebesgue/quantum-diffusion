from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProductTerm:
    """A product term in an ESOP expression.

    Each bit position is one of: 0 (must be 0), 1 (must be 1), or don't-care.
    Represented as a pair of bitmasks:

    - mask: which bits are specified (1 = specified, 0 = don't-care)
    - value: the required value for specified bits
    """

    mask: int
    value: int
    num_variables: int

    def matches(self, minterm: int) -> bool:
        """Check if a minterm is covered by this product term."""
        return (minterm & self.mask) == self.value

    def control_wires_and_values(self, position_wires: list[int]) -> tuple[list[int], list[int]]:
        """Extract control wires and values for circuit generation, skipping don't-cares."""
        control_wires: list[int] = []
        control_values: list[int] = []

        for i, wire in enumerate(position_wires):
            bit_index = self.num_variables - 1 - i

            if self.mask & (1 << bit_index):
                control_wires.append(wire)
                control_values.append((self.value >> bit_index) & 1)

        return control_wires, control_values
