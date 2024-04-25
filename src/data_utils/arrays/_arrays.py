from __future__ import annotations
import torch
import numpy as np
from typing import Union
from torch.utils.data import Dataset, DataLoader

NumpyConvertible = Union[torch.Tensor, np.ndarray]

def take_batches(ds: Dataset | DataLoader, n: int):
    """Takes n batches from a dataset or dataloader."""
    batches = None
    for j, data in enumerate(ds):
        if j >= n:
            break
        if batches is None:
            batches = [[] for _ in range(len(data))] 
        for i, d in enumerate(data):
            if not isinstance(d, torch.Tensor):
                d = torch.tensor(d)
            batches[i].append(d)
    if batches is None:
        raise ValueError("Empty dataset passed.")
    
    batches_stacked = [torch.stack(batch) for batch in batches]
    return tuple(batches_stacked)



def flatten_coords(coords: torch.Tensor) -> torch.Tensor:
    assert coords.dim() == 4
    return coords.permute(0, 2, 3, 1).flatten(1, 2)


def subsample(x: torch.Tensor, stride: int) -> torch.Tensor:
    if stride == 1:
        return x
    if x.dim() == 2:
        return x[::stride, ::stride]
    if x.dim() == 3:
        return x[:, ::stride, ::stride]
    if x.dim() == 4:
        return x[:, :, ::stride, ::stride]
    raise ValueError(f"Cannot subsample tensor of dimension {x.dim()}")


def shuffle(x: NumpyConvertible, dim: int) -> torch.Tensor:
    idx = torch.randperm(x.shape[dim])

    t_shuffled = x[idx]
    return t_shuffled


def make_numpy(x: NumpyConvertible) -> np.ndarray:
    """Converts x to a numpy array. If x is a torch tensor, it is detached and moved to the cpu."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    assert isinstance(x, np.ndarray), f"x must be convertible to numpy, got {type(x)}"
    return x


def get_bytes_of_tensor(t: torch.Tensor):
    """Calculate total number of bytes to store `t`."""
    return t.nelement() * t.element_size()


def get_bytes_of_net(net: torch.nn.Module):
    """Calculate total number of bytes to store `net` parameters and buffers."""
    return sum(get_bytes_of_tensor(t) for t in net.parameters())


def order_by_first(x: np.ndarray | list, y: np.ndarray | list):
    """Orders x and y by the values of x."""
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    x_order = np.argsort(x)
    return x[x_order], y[x_order]


def round_grid(x: NumpyConvertible, grid_size: float) -> np.ndarray:
    """Round the values of `x` to the nearest multiple of `grid_size`."""
    x = make_numpy(x)
    return np.round(x / grid_size) * grid_size
