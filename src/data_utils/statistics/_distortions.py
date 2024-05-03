from __future__ import annotations
import torch
import math
import numpy as np
from data_utils.arrays import make_numpy, NumpyConvertible
from typing import Optional


def entropy(x: NumpyConvertible, grid_width: Optional[float] = None) -> float:
    """Returns entropy of an array using the histogram as an estimate of the probability distribution.

    If x is a float array, the data is rounded to the nearest grid point in a grid where each point has distance grid_width to its neighbors. I.e.
    if grid_width = 1.0, this is equal to standard rounding. This is related to the max error as following:

        max_error = grid_width / 2
        grid_width = 2 * max_error

    If grid_width is None, the data is not quantised.
    """
    x = make_numpy(x)
    if grid_width is not None:  # type: ignore
        x = np.round(x / grid_width) * grid_width
    x = x.flatten()
    c = np.unique(x, return_counts=True)[1]

    return entropy_counts(c)


def entropy_counts(counts: np.ndarray | torch.Tensor) -> float:
    """Returns the entropy of a distribution given the counts of each value."""
    counts = make_numpy(counts)
    c_prob = counts / counts.sum()
    c_prob = c_prob[c_prob > 0]
    return np.sum(-np.log2(c_prob) * c_prob)


def _prepare_array(*arrays: NumpyConvertible) -> list[np.ndarray]:
    """Checks if all arrays have the same shape and returns them as numpy arrays."""
    if len(arrays) == 0:
        return []
    np_arrays = []
    for a in arrays:
        np_arrays.append(make_numpy(a).flatten())

    for a in np_arrays:
        if a.shape != np_arrays[0].shape:
            raise ValueError(
                f"Shapes of arrays do not match: {a.shape} != {arrays[0].shape}"
            )
    return np_arrays


def max_error(x: NumpyConvertible, y: NumpyConvertible) -> float:
    """Returns the maximum absolute error between x and y."""
    x, y = _prepare_array(x, y)
    return np.max(np.abs(x - y))


def mse(x: NumpyConvertible, y: NumpyConvertible) -> float:
    """Returns the mean squared error between x and y.

    Assumes x and y are numpy arrays or torch tensors with the same shape.

    MSE = 1/N * sum_i (x_i - y_i)^2
    """
    x, y = _prepare_array(x, y)
    return float(np.mean((x - y) ** 2))


def rmse(x, y) -> float:
    """Returns the root mean squared error between x and y."""
    return math.sqrt(mse(x, y))


def nll(mean: torch.Tensor, logvar: torch.Tensor, features: torch.Tensor):
    """Returns the negative log likelihood of a gaussian distribution with mean and logvar given features."""
    # logvar is log(sigma^2)
    distortion = (mean - features) ** 2 * torch.exp(-logvar)
    regulariser = math.log(2 * torch.pi) + logvar
    return 1 / 2 * (distortion + regulariser).mean()


def mse_from_psnr(psnr: float, pixel_max=1.0) -> float:
    """
    Calculates MSE from PSNR. Assumes pixel_min = 0.
    """
    return (pixel_max**2) / (10 ** (psnr / 10))


def psnr_from_mse(mse: float, pixel_max=1.0, pixel_min=0.0) -> float:
    """
    Calculates PSNR from MSE. Take care of the pixel_min and pixel_max parameters.
    """
    if mse == 0:
        return float("inf")
    return 20 * math.log10((pixel_max - pixel_min) / math.sqrt(mse))


def psnr(
    x: NumpyConvertible,
    y: NumpyConvertible,
    pixel_max=1.0,
    pixel_min=0.0,
    clip=False,
) -> float:
    """
    Calculates PSNR between x and y.
    """
    x = make_numpy(x)
    y = make_numpy(y)
    if clip:
        x = np.clip(x, pixel_min, pixel_max)
        y = np.clip(y, pixel_min, pixel_max)
    return psnr_from_mse(mse(x, y), pixel_max=pixel_max, pixel_min=pixel_min)
