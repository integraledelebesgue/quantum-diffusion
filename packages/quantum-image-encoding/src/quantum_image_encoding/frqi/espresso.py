from .product_term import ProductTerm

__all__ = ['espresso_esop']


def espresso_esop(minterms: list[int], num_variables: int) -> list[ProductTerm]:
    """ESOP minimization using an EXORCISM-style heuristic.

    1. Start with one fully-specified term per minterm
    2. Iteratively merge and reshape until no improvement
    """

    all_bits = (1 << num_variables) - 1
    terms: list[tuple[int, int]] = [(all_bits, m) for m in minterms]

    improved = True
    while improved:
        improved = _merge_pass(terms)
        improved |= _reshape_pass(terms)

    return [ProductTerm(mask=mask, value=value, num_variables=num_variables) for mask, value in terms]


def _merge_pass(terms: list[tuple[int, int]]) -> bool:
    """Merge pairs of terms with the same mask differing in exactly one specified bit.

    In ESOP, two terms (mask, v1) and (mask, v2) differing in one mask-bit
    combine into (mask & ~diff_bit, v1 & new_mask). Modifies terms in place.
    """

    improved = False
    i = 0

    while i < len(terms):
        merged = False
        mask_i, val_i = terms[i]

        for j in range(i + 1, len(terms)):
            mask_j, val_j = terms[j]

            if mask_i != mask_j:
                continue

            diff = val_i ^ val_j

            if diff != 0 and (diff & (diff - 1)) == 0 and (diff & mask_i):
                new_mask = mask_i & ~diff
                new_value = val_i & new_mask
                new_term = (new_mask, new_value)

                # Remove the two original terms
                terms.pop(j)
                terms.pop(i)

                # In ESOP, if the merged term already exists, they cancel (XOR)
                if new_term in terms:
                    terms.remove(new_term)
                else:
                    terms.insert(i, new_term)

                improved = True
                merged = True
                break

        if not merged:
            i += 1

    return improved


def _reshape_pass(terms: list[tuple[int, int]]) -> bool:
    """Reshape terms via distance-2 transformations with XOR cancellation.

    Two terms with the same mask differing in 2 bits can be replaced with
    two terms each having one additional don't-care, if this enables
    XOR cancellation with existing terms.
    """
    improved = False
    i = 0

    while i < len(terms):
        mask_i, val_i = terms[i]
        j = i + 1
        found = False

        while j < len(terms):
            mask_j, val_j = terms[j]

            if mask_i != mask_j:
                j += 1

                continue

            diff = val_i ^ val_j

            if bin(diff).count('1') != 2:
                j += 1
                continue

            bit_a = diff & (-diff)
            bit_b = diff ^ bit_a

            new_term_a = (mask_i & ~bit_a, val_i & (mask_i & ~bit_a))
            new_term_b = (mask_i & ~bit_b, val_i & (mask_i & ~bit_b))

            # Simulate the replacement and check for XOR cancellation
            candidate = list(terms)
            candidate.pop(max(i, j))
            candidate.pop(min(i, j))

            for new_term in [new_term_a, new_term_b]:
                if new_term in candidate:
                    candidate.remove(new_term)
                else:
                    candidate.append(new_term)

            if len(candidate) < len(terms):
                terms.clear()
                terms.extend(candidate)
                improved = True
                found = True
                break

            j += 1

        if found:
            i = 0  # Restart after modification
        else:
            i += 1

    return improved
