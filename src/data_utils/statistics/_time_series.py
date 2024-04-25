import numpy as np
from data_utils.arrays import make_numpy, NumpyConvertible
from typing import Optional


def rank_time_series(x: np.ndarray, axis=-1):
    """Turn a set of a spatio-temporal time series (time, width, height) into ranked time series. Every entry
    is the rank of the entry in the time series, i.e. the smallest entry is 0, the second smallest 1, etc.

    The time series are ranked along the given axis."""
    assert len(x.shape) == 3
    return x.argsort(axis).argsort(axis)


def cosine_similarity(x: np.ndarray):
    """Calculates the cosine similarity of a x,y,t time ranked time series."""
    assert len(x.shape) == 3
    x = x.reshape(x.shape[0] * x.shape[1], x.shape[-1])
    norm = np.linalg.norm(x, axis=-1)
    return x @ x.T / np.outer(norm, norm)


def spearman_rho(x: np.ndarray):
    """Calculates the pairwise spearman-rho of an unranked, zero-mean unit standard variance x,y,t time series
    (passing an unnormalized time series is fine, but the spearman-rho is wrt. the normalized time series).
    """
    return cosine_similarity(rank_time_series(x))


def angular_distance(x: np.ndarray):
    """Calculates the angular distance of an unranked, zero-mean unit standard variance x,y,t time series."""
    return 2 * np.arccos(np.clip(spearman_rho(x), 0, 1)) / np.pi


def autocorr(
    ds: NumpyConvertible, x: int, y: int, t_max: Optional[int] = 100
) -> np.ndarray:
    """Calculates the autocorrelation function of a spatio-temporal time series (time, height, width) ds at position (x,y).

    Args:
        ds: The time series of shape (time, height, width).
        x: The x position.
        y: The y position.
        t_max: The maximum time lag to consider. If None, the full time series is used.
    Returns:
        The autocorrelation function at (x,y) of shape (t_max,).
    """
    ds = make_numpy(ds)
    if t_max is None:
        inp = ds[:None, 0, x, y].flatten()
    else:
        inp = ds[:t_max, 0, x, y].flatten()

    mean = inp.mean()
    var = np.var(inp)
    xp = inp - mean

    result = np.correlate(xp, xp, mode="full") / var / len(inp)
    midway = np.argmax(result)
    return result[midway:]


def detrend(x: NumpyConvertible) -> np.ndarray:
    """Removes the trend of the time series x. The time series returned is calculated by:
        x'[t] = x[t] - x[t-1] for t >= 1.

    Args:
        x: The time series of shape (time, ...)
    """
    x = make_numpy(x)
    npad = [(1, 0)] + [(0, 0)] * (len(x.shape) - 1)
    return x - np.pad(x, npad)[:-1]
