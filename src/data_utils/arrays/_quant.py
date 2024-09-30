from typing import Optional, TypeVar, Union
import torch
import numpy as np
from ._arrays import make_numpy, NumpyConvertible


def quant_to_grid(
    x: NumpyConvertible, dx: Optional[float] = None, return_ints: bool = False
) -> tuple[np.ndarray, np.ndarray, float]:
    xn = make_numpy(x)

    unique_vals = np.unique(xn)
    if dx is None:
        dx = find_dx(xn)
    if dx == 0:
        return np.array([0]), np.array([xn.size]), 0
    assert isinstance(dx, float), f"dx must be a float, got {type(dx)}"

    grid_points = np.arange(np.min(unique_vals), np.max(unique_vals) + dx, dx)

    w_idx = np.round(xn / dx).astype(int)
    grid_idx = np.round(grid_points / dx).astype(int)

    counts = np.array([np.sum(w_idx == point) for point in grid_idx])
    grid_points = grid_idx * dx  # this restores 0

    if return_ints:
        grid_points = grid_idx
    return grid_points, counts, dx


def find_dx(x: np.ndarray) -> float:
    unique_vals = np.unique(x)
    if len(unique_vals) == 1:
        return 0.0
    dxarr = np.min(np.abs(np.diff(unique_vals)))
    larger_0 = dxarr > 0
    if not larger_0.any():
        return 0.0
    dx = float(dxarr[larger_0].min())  # Remove zero values
    return dx


def to_idx_space(x: NumpyConvertible, dx: Optional[float] = None) -> np.ndarray:
    x = make_numpy(x)
    if dx is None:
        dx = find_dx(x)
    if dx == 0:
        return np.zeros_like(x, dtype=int)
    return np.round(x / dx).astype(int)
