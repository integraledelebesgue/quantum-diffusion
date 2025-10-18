from typing import Literal

import einops
import torch
import torchvision
import torchvision.transforms as T
from sklearn import datasets

Dataset = Literal[
    "mnist_8x8",
    "mnist_28x28",
    "mnist_32x32",
    "cifar10_32x32",
    "fashion_28x28",
]


def get_by_name(
    dataset: Dataset,
    n_classes: int,
    ds_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    match dataset:
        case "mnist_8x8":
            return mnist_8x8(n_classes, ds_size)

        case "mnist_28x28":
            return mnist_28x28(n_classes, ds_size)

        case "mnist_32x32":
            return mnist_32x32(n_classes, ds_size)

        case "cifar10_32x32":
            return cifar10_32x32(n_classes, ds_size)

        case "fashion_28x28":
            return fashion_28x28(n_classes, ds_size)


def __normalize_labels(labels: torch.Tensor) -> torch.Tensor:
    return labels / labels.max()  # Normalize to [0, 1]


def mnist_8x8(
    n_classes: int = 10,
    ds_size: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    x_train, y_train = datasets.load_digits(n_class=n_classes, return_X_y=True)
    x_train /= 16
    x_train = x_train.reshape(-1, 64)
    x_train = torch.tensor(x_train, dtype=torch.double)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_train = __normalize_labels(y_train)
    x_train, y_train = x_train[:ds_size], y_train[:ds_size]
    return x_train, y_train, 8, 8


def mnist_28x28(
    n_classes: int = 10,
    ds_size: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    ds = torchvision.datasets.MNIST(
        root="~/mnist", download=True, transform=T.ToTensor()
    )
    idx = ds.targets < n_classes
    ds.data = ds.data[idx]
    ds.targets = ds.targets[idx]
    x_train, y_train = (
        torch.utils.data.DataLoader(ds, batch_size=ds_size, shuffle=False)
        .__iter__()
        .__next__()
    )
    y_train = __normalize_labels(y_train)
    x_train = x_train.flatten(start_dim=1)
    x_train = x_train.to(torch.double)
    y_train = y_train.to(torch.double)
    return x_train, y_train, 28, 28


def mnist_32x32(
    n_classes: int = 10,
    ds_size: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    tra = T.Compose(
        [
            T.ToTensor(),
            T.Resize((32, 32)),
        ]
    )
    ds = torchvision.datasets.MNIST(root="~/mnist", download=True, transform=tra)
    idx = ds.targets < n_classes
    ds.data = ds.data[idx]
    ds.targets = ds.targets[idx]
    x_train, y_train = (
        torch.utils.data.DataLoader(ds, batch_size=ds_size, shuffle=False)
        .__iter__()
        .__next__()
    )
    y_train = __normalize_labels(y_train)
    x_train = x_train.flatten(start_dim=1)
    x_train = x_train.to(torch.double)
    y_train = y_train.to(torch.double)
    return x_train, y_train, 32, 32


def cifar10_32x32(
    n_classes: int = 10,
    ds_size: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    transformation = T.Compose(
        [
            T.functional.rgb_to_grayscale,
            T.ToTensor(),
            T.Lambda(lambda x: torch.flatten(x)),
        ]
    )
    ds = torchvision.datasets.CIFAR10(
        root="~/cifar", download=True, transform=transformation
    )
    ds.targets = torch.tensor(ds.targets)
    idx = ds.targets < n_classes
    ds.data = ds.data[idx]
    ds.targets = ds.targets[idx]
    x_train, y_train = (
        torch.utils.data.DataLoader(ds, batch_size=ds_size, shuffle=False)
        .__iter__()
        .__next__()
    )
    y_train = __normalize_labels(y_train)
    x_train = x_train.to(torch.double)
    y_train = y_train.to(torch.double)
    return x_train, y_train, 32, 32


def fashion_28x28(
    n_classes: int = 10,
    ds_size: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    ds = torchvision.datasets.FashionMNIST(
        root="~/fashion", download=True, transform=T.ToTensor()
    )
    ds.targets = torch.tensor(ds.targets)
    idx = ds.targets < n_classes
    ds.data = ds.data[idx]
    ds.targets = ds.targets[idx]
    x_train, y_train = (
        torch.utils.data.DataLoader(ds, batch_size=ds_size, shuffle=False)
        .__iter__()
        .__next__()
    )
    x_train = einops.rearrange(x_train, "b 1 h w -> b (h w)")
    y_train = __normalize_labels(y_train)
    x_train = x_train.to(torch.double)
    y_train = y_train.to(torch.double)
    return x_train, y_train, 28, 28
