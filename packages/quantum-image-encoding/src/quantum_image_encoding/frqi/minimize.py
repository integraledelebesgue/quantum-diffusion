from typing import Literal

from .espresso import espresso_esop
from .product_term import ProductTerm
from .quine_mccluskey import quine_mccluskey_esop

__all__ = ['minimize_minterms', 'sort_esop_terms']


def sort_esop_terms(terms: list[ProductTerm]) -> list[ProductTerm]:
    """Sort ESOP terms using greedy nearest-neighbor to minimize X-gate transitions.

    Each term implies a target X-gate state per wire: True (needs X for control_value=0),
    False (no X for control_value=1), or None (don't-care, wire not controlled).
    The distance between consecutive terms is the number of specified wires whose
    required X state differs from the current state.
    """

    if len(terms) <= 1:
        return list(terms)

    num_variables = terms[0].num_variables

    def term_x_targets(term: ProductTerm) -> list[bool | None]:
        targets: list[bool | None] = []

        for i in range(num_variables):
            bit_index = num_variables - 1 - i

            if term.mask & (1 << bit_index):
                targets.append(not bool((term.value >> bit_index) & 1))
            else:
                targets.append(None)

        return targets

    all_targets = [term_x_targets(t) for t in terms]

    def distance(current_state: list[bool | None], target: list[bool | None]) -> int:
        return sum(1 for curr, tgt in zip(current_state, target) if tgt is not None and curr != tgt)

    current_state: list[bool | None] = [False for _ in range(num_variables)]
    remaining = set(range(len(terms)))
    order: list[int] = []

    for _ in range(len(terms)):
        best_idx = min(remaining, key=lambda i: distance(current_state, all_targets[i]))
        order.append(best_idx)
        remaining.discard(best_idx)

        for j in range(num_variables):
            if all_targets[best_idx][j] is not None:
                current_state[j] = all_targets[best_idx][j]

    return [terms[i] for i in order]


def minimize_minterms(
    minterms: list[int],
    num_variables: int,
    method: Literal['quine-mccluskey', 'espresso'],
) -> list[ProductTerm]:
    """Minimize a set of minterms into an ESOP expression.

    Args:
        minterms: List of minterm indices (integers 0..2^num_vars - 1).
        num_vars: Number of boolean variables.
        method: Minimization algorithm to use.

    Returns:
        List of ProductTerm forming an ESOP (XOR-sum-of-products) that
        covers exactly the given minterms.
    """
    if len(minterms) == 0:
        return []

    all_bits = (1 << num_variables) - 1

    if len(minterms) == 1:
        return [ProductTerm(mask=all_bits, value=minterms[0], num_variables=num_variables)]

    match method:
        case 'quine-mccluskey':
            return quine_mccluskey_esop(minterms, num_variables)

        case 'espresso':
            return espresso_esop(minterms, num_variables)

        case other:
            raise ValueError(f'Unknown minimization method: {other}')
