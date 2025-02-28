import numpy as np
from ..arrays._arrays import NumpyConvertible, make_numpy


def shannon(w: NumpyConvertible):
    """Returns shannon entropy of the given tensor."""
    w = make_numpy(w).flatten()
    _, c = np.unique(w, return_counts=True)
    p = c / c.sum()
    return -(p * np.log2(p)).sum()
