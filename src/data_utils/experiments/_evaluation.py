from typing import Iterable, Literal, Optional
import pandas as pd
import numpy as np


def pareto_front(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    mode: Literal["over", "under"],
    nsteps: int = 100,
    grid: Literal["linear", "log", "continuous"] | np.ndarray = "continuous",
    logspace: Optional[bool] = None,
) -> pd.DataFrame:
    if logspace is not None:
        grid = "log" if logspace else "linear"
    pareto_rows = []
    if isinstance(grid, np.ndarray):
        threshhold_values = grid * df[y_column].max()
    else:
        if grid == "log":
            threshhold_values = np.logspace(
                np.log10(df[y_column].min()), np.log10(df[y_column].max()), nsteps
            )
            np.append(threshhold_values, df[y_column].max())
        elif grid == "linear":
            threshhold_values = np.linspace(
                df[y_column].min(), df[y_column].max(), nsteps
            )
            np.append(threshhold_values, df[y_column].max())
        else:
            threshhold_values = np.sort(df[y_column].unique())
            threshhold_values = (
                np.flip(threshhold_values) if mode == "under" else threshhold_values
            )

    for t in threshhold_values:
        mask = df[y_column] <= t if mode == "under" else df[y_column] >= t
        acceptable = df[mask]
        if len(acceptable) == 0:
            continue
        pareto_idx = acceptable[x_column].argmin()
        pareto_rows.append(acceptable.iloc[pareto_idx])
    return pd.DataFrame(pareto_rows).drop_duplicates()
