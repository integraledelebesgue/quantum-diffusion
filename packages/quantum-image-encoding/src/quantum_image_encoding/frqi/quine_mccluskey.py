from collections import defaultdict

from .minimize import ProductTerm

__all__ = ['quine_mccluskey_esop']


def quine_mccluskey_esop(minterms: list[int], num_vars: int) -> list[ProductTerm]:
    """Find an ESOP cover using Quine-McCluskey prime implicant generation
    and greedy GF(2) cover selection."""

    minterm_set = set(minterms)
    prime_implicants = _find_prime_implicants(minterm_set, num_vars)
    return _find_esop_cover(prime_implicants, minterm_set, num_vars)


def _find_prime_implicants(minterms: set[int], num_vars: int) -> list[tuple[int, int]]:
    """Standard Quine-McCluskey prime implicant generation.

    Groups terms by popcount, merges adjacent terms differing in exactly one
    specified bit. Returns list of (mask, value) tuples.
    """

    all_bits = (1 << num_vars) - 1
    current_terms: set[tuple[int, int]] = {(all_bits, m) for m in minterms}
    prime_implicants: set[tuple[int, int]] = set()

    while current_terms:
        groups: defaultdict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)

        for mask, value in current_terms:
            pc = bin(value & mask).count('1')
            groups[(mask, pc)].append((mask, value))

        merged: set[tuple[int, int]] = set()
        used: set[tuple[int, int]] = set()

        for (mask, pc), terms_in_group in groups.items():
            next_group = groups.get((mask, pc + 1), [])

            for mask_a, val_a in terms_in_group:
                for _mask_b, val_b in next_group:
                    diff = val_a ^ val_b

                    if diff & (diff - 1) == 0 and (diff & mask):
                        new_mask = mask & ~diff
                        new_value = val_a & new_mask
                        merged.add((new_mask, new_value))
                        used.add((mask_a, val_a))
                        used.add((mask, val_b))

        for term in current_terms:
            if term not in used:
                prime_implicants.add(term)

        current_terms = merged

    return list(prime_implicants)


def _find_esop_cover(
    prime_implicants: list[tuple[int, int]],
    minterms: set[int],
    num_variables: int,
) -> list[ProductTerm]:
    """Find ESOP cover using greedy GF(2) selection.

    Each minterm must be covered an odd number of times (XOR semantics).
    Greedily picks the implicant covering the most uncovered minterms,
    then XORs its coverage into the running state.
    """

    all_minterms = sorted(minterms)
    n_minterms = len(all_minterms)

    if not prime_implicants:
        return []

    # For each prime implicant, compute coverage bitmask over minterms
    implicant_covers: list[int] = []

    for mask, value in prime_implicants:
        cover = 0

        for idx, m in enumerate(all_minterms):
            if (m & mask) == value:
                cover |= 1 << idx

        implicant_covers.append(cover)

    # Greedy: pick implicant covering most remaining bits, XOR coverage
    remaining = (1 << n_minterms) - 1
    selected: list[int] = []

    while remaining:
        best_idx = -1
        best_count = -1

        for j, cover in enumerate(implicant_covers):
            overlap = bin(cover & remaining).count('1')

            if overlap > best_count:
                best_count = overlap
                best_idx = j

        if best_count <= 0:
            break

        selected.append(best_idx)
        remaining ^= implicant_covers[best_idx]

    return [ProductTerm(mask=prime_implicants[i][0], value=prime_implicants[i][1], num_variables=num_variables) for i in selected]
