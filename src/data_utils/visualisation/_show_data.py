import torch
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import math
from data_utils.arrays import NumpyConvertible


def show_tensor(
    x: torch.Tensor | list[NumpyConvertible] | np.ndarray,
    imgs_per_row: int = 4,
    bar: bool = False,
    figsize: Optional[tuple[int, int]] = None,
    title: Optional[str] = None,
    clim: Optional[tuple[float, float]] = None,
    **kwargs,
):
    """Shows a single tensor or a grid of tensors. The images are immediately displayed.

    Args:
        x: The tensor or list of tensors to show. If a list is given, a grid of images is shown.
        imgs_per_row: The number of images per row in the grid.
        bar: Whether to show a color bar.
        figsize: The size of the figure,
        clim: The color limits for the color bar.
        **kwargs: Additional arguments passed to plt.imshow(). These are only used in the case of a single tensor. These are only used in the case of a single tensor.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(x, torch.Tensor):
        _show_single_tensor(x, bar, figsize, title, clim, **kwargs)
    elif isinstance(x, list):
        _show_grid(x, figsize, imgs_per_row=imgs_per_row, title=title)
    else:
        raise ValueError(f"Invalid type for x: {type(x)}")


def _show_single_tensor(
    x: torch.Tensor,
    bar: bool = False,
    figsize: Optional[tuple[int, int]] = None,
    title: Optional[str] = None,
    clim: Optional[tuple[float, float]] = None,
    **kwargs,
):
    """Shows a single tensor. The image is immediately displayed."""
    x = x.detach().cpu()
    plt.figure(figsize=figsize)
    plt.imshow(_to_wch(x), **kwargs)
    if bar:
        plt.colorbar()
        if clim is not None:
            plt.clim(*clim)
    if title is not None:
        plt.title(title)
    plt.show()


def _to_wch(x: torch.Tensor):
    """Converts a tensor to a WCH image. Tensor must be of shape (C, W, H) or (1, C, W, H)."""
    if len(x.shape) == 4:
        x = x.squeeze(0)
    if x.shape[0] == 1 or x.shape[0] == 3:
        x = x.permute(1, 2, 0)
    return x


def _show_grid(
    x: list[NumpyConvertible],
    figsize: Optional[tuple[int, int]] = None,
    imgs_per_row: int = 4,
    title: Optional[str] = None,
):
    """Shows a grid of tensors"""
    cols = min(len(x), imgs_per_row)
    rows = math.ceil(len(x) // cols)
    f, axarr = plt.subplots(rows, cols, figsize=figsize)

    axarr = axarr.flatten()

    for img, ax in zip(x, axarr):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        img = img.detach().cpu()
        ax.imshow(_to_wch(img))

    if title is not None:
        f.suptitle(title)
    plt.show()


def show_hist(
    x: NumpyConvertible,
    title: Optional[str] = None,
    bins: Optional[int] = None,
    range: Optional[tuple[float, float]] = None,
    *args,
    **kwargs,
):
    """Shows a histogram of the given tensor or numpy array. The histogram is immediately displayed.

    Args:
        x: The tensor or numpy array to show.
        title: The title of the histogram.
        bins: The number of bins.
        range: The range of the histogram.
        *args: Additional arguments passed to plt.hist().
        **kwargs: Additional arguments passed to plt.hist().
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().numpy()
    assert isinstance(x, np.ndarray)
    plt.hist(x.flatten(), bins=bins, range=range, *args, **kwargs)  # type:ignore
    if title is not None:
        plt.title(title)
    plt.show()
