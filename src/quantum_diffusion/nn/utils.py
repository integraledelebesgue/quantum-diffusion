import math
import warnings

import einops
import torch


def autopad(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad the y image to the size of x"""
    xs, ys = x.shape, y.shape

    if xs < ys:
        warnings.warn("x is smaller than y. Padding x to match y")
        return autopad(y, x)

    y_padded = torch.nn.functional.pad(
        y,
        (
            math.ceil((xs[3] - ys[3]) / 2),
            math.floor((xs[3] - ys[3]) / 2),
            math.ceil((xs[2] - ys[2]) / 2),
            math.floor((xs[2] - ys[2]) / 2),
        ),
        mode="constant",
        value=0,
    )

    return x, y_padded


def __get_label_embedding_1(
    labels: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """Returns a mask for the labels"""
    batch = labels.shape[0]
    y = einops.repeat(labels, "b -> b w", w=width)

    mask = torch.arange(width, device=labels.device) / 20
    mask = einops.repeat(
        mask,
        "w -> b w",
        b=batch,
    )

    mask = torch.sin(y + mask)
    mask = mask * 0.1
    mask = einops.repeat(mask, "b w -> b 1 w h", h=height)

    return mask


def __get_label_embedding_2(
    labels: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """Returns a mask for the labels"""
    assert labels.unique().shape[0] == 2 and labels.min() == 0 and labels.max() == 1, (
        "Labels must be binary"
    )

    batch = labels.shape[0]
    mask = torch.zeros((batch, 1, width, height), device=labels.device)

    mask[
        :,
        :,
        : width // 2,
    ] = (labels == 0).reshape(batch, 1, 1, 1).float() * 0.1
    mask[:, :, width // 2 :] = (labels == 1).reshape(batch, 1, 1, 1).float() * 0.1

    return mask


get_label_embedding = __get_label_embedding_1
