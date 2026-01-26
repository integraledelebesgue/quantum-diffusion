from typing import SupportsIndex

import grain
import jax
import jax.numpy as jnp
from sklearn.datasets import load_digits


class Digits8x8(grain.sources.RandomAccessDataSource[tuple[jax.Array, jax.Array]]):
    def __init__(self, num_classes: int) -> None:
        images, labels = load_digits(n_class=num_classes, return_X_y=True)
        self.images = jnp.array(images).reshape(-1, 1, 8, 8)
        self.labels = jnp.array(labels)

    def __getitem__(self, record_key: SupportsIndex) -> tuple[jax.Array, jax.Array]:
        return self.images[record_key, :], self.labels[record_key]

    def __len__(self) -> int:
        return len(self.images)
