"""
Module contains all neural networks used in the project.
Some are not used anymore, but are kept for reference and marked as deprecated.
"""

from collections.abc import Sequence
from typing import Any, Literal

import torch

from .conv import DeepConvDirectedMulti, DeepConvDirectedSingle, DeepConvUndirected
from .dense import DenseDirected, DenseUndirected
from .qconv import QConv2d
from .qdense import (
    QDense2Undirected,
    QDense4StatesAncilla,
    QDense4StatesUndirected,
    QDenseDirected,
    QDenseDirectedReupload,
    QDenseUndirected,
)
from .unet import UnetDirected, UNetUndirected
from .unet_simple import UnetDirectedS, UNetUndirectedS
from .utils import (
    autocrop,
    autopad,
    circuit_to_qasm,
    get_label_embedding,
    repeat_qasm,
    sample_from_qiskit,
)

MODELS = (
    "DeepConvDirectedMulti",
    "DeepConvDirectedSingle",
    "DeepConvUndirected",
    "DenseDirected",
    "DenseUndirected",
    "QConv2d",
    "QDense2Undirected",
    "QDense4StatesAncilla",
    "QDense4StatesUndirected",
    "QDenseDirected",
    "QDenseDirectedReupload",
    "QDenseUndirected",
    "UnetDirected",
    "UNetUndirected",
    "UnetDirectedS",
    "UNetUndirectedS",
)

Model = Literal[
    "DeepConvDirectedMulti",
    "DeepConvDirectedSingle",
    "DeepConvUndirected",
    "DenseDirected",
    "DenseUndirected",
    "QConv2d",
    "QDense2Undirected",
    "QDense4StatesAncilla",
    "QDense4StatesUndirected",
    "QDenseDirected",
    "QDenseDirectedReupload",
    "QDenseUndirected",
    "UnetDirected",
    "UNetUndirected",
    "UnetDirectedS",
    "UNetUndirectedS",
]


def get_by_name(model: Model, parameters: Sequence[Any]) -> torch.nn.Module:
    match model:
        case "DeepConvDirectedMulti":
            return DeepConvDirectedMulti(*parameters)
        case "DeepConvDirectedSingle":
            return DeepConvDirectedSingle(*parameters)
        case "DeepConvUndirected":
            return DeepConvUndirected(*parameters)
        case "DenseDirected":
            return DenseDirected(*parameters)
        case "DenseUndirected":
            return DenseUndirected(*parameters)
        case "QConv2d":
            return QConv2d(*parameters)
        case "QDense2Undirected":
            return QDense2Undirected(*parameters)
        case "QDense4StatesAncilla":
            return QDense4StatesAncilla(*parameters)
        case "QDense4StatesUndirected":
            return QDense4StatesUndirected(*parameters)
        case "QDenseDirected":
            return QDenseDirected(*parameters)
        case "QDenseDirectedReupload":
            return QDenseDirectedReupload(*parameters)
        case "QDenseUndirected":
            return QDenseUndirected(*parameters)
        case "UnetDirected":
            return UnetDirected(*parameters)
        case "UNetUndirected":
            return UNetUndirected(*parameters)
        case "UnetDirectedS":
            return UnetDirectedS(*parameters)
        case "UNetUndirectedS":
            return UNetUndirectedS(*parameters)


__all__ = [
    "Model",
    "get_by_name",
    "DeepConvDirectedMulti",
    "DeepConvDirectedSingle",
    "DeepConvUndirected",
    "DenseDirected",
    "DenseUndirected",
    "QConv2d",
    "QDense2Undirected",
    "QDense4StatesAncilla",
    "QDense4StatesUndirected",
    "QDenseDirected",
    "QDenseDirectedReupload",
    "QDenseUndirected",
    "UnetDirected",
    "UNetUndirected",
    "UnetDirectedS",
    "UNetUndirectedS",
    "autocrop",
    "autopad",
    "circuit_to_qasm",
    "get_label_embedding",
    "repeat_qasm",
    "sample_from_qiskit",
]
