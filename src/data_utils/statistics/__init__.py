"""This module contains functions and classes that help evaluating various compression methods.
"""

from ._distortions import (
    psnr,
    rmse,
    mse,
    psnr_from_mse,
    mse_from_psnr,
    max_error,
    nll,
    entropy,
)
from ._time_series import (
    spearman_rho,
    cosine_similarity,
    rank_time_series,
    angular_distance,
    detrend,
    autocorr,
)

__all__ = [
    "psnr",
    "mse",
    "rmse",
    "nll",
    "max_error",
    "entropy",
    "psnr_from_mse",
    "mse_from_psnr",
    "spearman_rho",
    "cosine_similarity",
    "rank_time_series",
    "angular_distance",
    "detrend",
    "autocorr",
]
