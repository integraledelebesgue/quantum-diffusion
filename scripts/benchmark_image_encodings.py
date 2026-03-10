from collections import Counter
from math import ceil, log2
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pennylane as qml
from quantum_datasets import Digits8x8

from quantum_image_encoding import FRQI, Amplitude, Encoding, RealKet

IMAGE_SIZES = [
    (2, 2),
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
]

ENCODINGS = [
    ("FRQI", lambda shape: FRQI(shape), "-o"),
    ("FRQI (Gray)", lambda shape: FRQI(shape, gray_code=True), "-s"),
    ("FRQI (QM)", lambda shape: FRQI(shape, minimize="quine-mccluskey"), "--^"),
    ("FRQI (Espresso)", lambda shape: FRQI(shape, minimize="espresso"), "--D"),
    ("Amplitude", lambda shape: Amplitude(shape), ":x"),
    ("Real Ket", lambda shape: RealKet(shape), "-v"),
    ("Real Ket (bd=4)", lambda shape: RealKet(shape, max_bond_dim=4), "-<"),
    ("Real Ket (bd=2)", lambda shape: RealKet(shape, max_bond_dim=2), "-P"),
]


def make_uniform_image(size: int, key: jax.Array) -> jax.Array:
    return jnp.ones(size) * 0.5


def make_two_color_image(size: int, key: jax.Array) -> jax.Array:
    image = (
        jax.random.randint(key, (size,), minval=0, maxval=2).astype(jnp.float32) * 0.5
    )

    if (image == 0.0).all():
        return image.at[-1].set(0.5)

    return image


def make_gaussian_noise_image(size: int, key: jax.Array) -> jax.Array:
    return jnp.clip(jax.random.normal(key, (size,)) * 0.25 + 0.5, 0.0, 1.0)


def make_uniform_noise_image(size: int, key: jax.Array) -> jax.Array:
    return jax.random.uniform(key, (size,))


def make_digit_image(size: int, key: jax.Array) -> jax.Array | None:
    if size != 64:
        return None

    dataset = Digits8x8(num_classes=10)
    image, _ = dataset[0]
    return image.flatten() / 16


IMAGE_TYPES = [
    ("Single color", make_uniform_image),
    ("Two colors", make_two_color_image),
    ("Gaussian noise", make_gaussian_noise_image),
    ("Uniform noise", make_uniform_noise_image),
    ("Digit", make_digit_image),
]


def decompose_operations(
    operations: list[qml.operation.Operation],
) -> list[qml.operation.Operation]:
    result: list[qml.operation.Operation] = []

    for operation in operations:
        if operation.has_decomposition:
            result.extend(decompose_operations(operation.decomposition()))
        else:
            result.append(operation)

    return result


def circuit_depth(operations: list[qml.operation.Operation]) -> int:
    wire_depths: dict[int, int] = {}

    for operation in operations:
        max_depth = max((wire_depths.get(w, 0) for w in operation.wires), default=0)
        new_depth = max_depth + 1

        for w in operation.wires:
            wire_depths[w] = new_depth

    return max(wire_depths.values(), default=0)


def count_gates(encoding: Encoding, image: jax.Array) -> dict[str, Any]:
    num_wires = encoding.num_wires()

    with qml.queuing.AnnotatedQueue() as q:
        encoding.encode(image, list(range(num_wires)))

    tape = qml.tape.QuantumScript.from_queue(q)

    elementary = decompose_operations(tape.operations)
    decomposed = Counter(op.name for op in elementary)

    return {
        "num_wires": num_wires,
        "decomposed": dict(decomposed),
        "num_gates_decomposed": sum(decomposed.values()),
        "depth_decomposed": circuit_depth(elementary),
    }


def format_gate_types(gate_types: dict[str, int]) -> str:
    return ", ".join(f"{name}: {count}" for name, count in sorted(gate_types.items()))


def collect_results() -> dict[str, tuple[list[str], dict[str, list[dict[str, Any]]]]]:
    key = jax.random.PRNGKey(42)
    all_results: dict[str, tuple[list[str], dict[str, list[dict[str, Any]]]]] = {}

    for image_type_name, make_image in IMAGE_TYPES:
        labels: list[str] = []
        results: dict[str, list[dict[str, Any]]] = {
            name: [] for name, _, _ in ENCODINGS
        }

        print(f"=== {image_type_name} ===")
        print()

        for height, width in IMAGE_SIZES:
            image_shape = (1, height, width)
            num_pixels = height * width
            num_position_qubits = int(ceil(log2(num_pixels)))

            key, subkey = jax.random.split(key)
            image = make_image(num_pixels, subkey)

            if image is None:
                continue

            labels.append(f"{height}x{width}")

            print(
                f"  --- {height}x{width} image ({num_pixels} pixels, {num_position_qubits} position qubits) ---"
            )
            print()

            for name, make_encoding, _ in ENCODINGS:
                result = count_gates(make_encoding(image_shape), image)
                results[name].append(result)

                print(f"    {name} ({result['num_wires']} qubits)")
                print(
                    f"      Gates:  {result['num_gates_decomposed']:>6}  {format_gate_types(result['decomposed'])}"
                )
                print(f"      Depth:  {result['depth_decomposed']:>6}")
                print()

        all_results[image_type_name] = (labels, results)

    return all_results


def plot(
    all_results: dict[str, tuple[list[str], dict[str, list[dict[str, Any]]]]],
    path: Path,
) -> None:
    image_types = list(all_results.keys())
    n_types = len(image_types)

    fig, axes = plt.subplots(
        2, n_types, figsize=(4 * n_types, 8), squeeze=False, sharey="row"
    )

    for col, image_type in enumerate(image_types):
        labels, results = all_results[image_type]
        x = range(len(labels))

        for (name, _, fmt), data in zip(ENCODINGS, results.values()):
            gate_counts = [d["num_gates_decomposed"] for d in data]
            depths = [d["depth_decomposed"] for d in data]

            axes[0, col].plot(x, gate_counts, fmt, label=name, markersize=6)
            axes[1, col].plot(x, depths, fmt, label=name, markersize=6)

        for row, (title, ylabel) in enumerate(
            [("Decomposed gates", "Gate count"), ("Circuit depth", "Depth")]
        ):
            ax = axes[row, col]
            ax.set_yscale("log")
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels)
            ax.set_xlabel("Image size")
            ax.set_title(f"{image_type} — {title}")
            ax.legend(fontsize="small")
            ax.grid(True, alpha=0.3)

            if col == 0:
                ax.set_ylabel(ylabel)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")


def main() -> None:
    all_results = collect_results()
    plot(all_results, Path("data/benchmark_encoding_gates.png"))


if __name__ == "__main__":
    main()
