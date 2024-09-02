from typing import Iterable, Literal
import pandas as pd
import numpy as np


def pareto_front(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    mode: Literal["over", "under"],
    nsteps: int = 100,
    logspace: bool = False,
) -> pd.DataFrame:
    pareto_rows = []
    val_range = df[y_column].max() - df[y_column].min()
    if logspace:
        threshhold_values = np.logspace(
            np.log10(df[y_column].min()), np.log10(df[y_column].max()), nsteps
        )
    else:
        threshhold_values = np.linspace(df[y_column].min(), df[y_column].max(), nsteps)

    for t in threshhold_values:
        mask = df[y_column] < t if mode == "under" else df[y_column] > t
        acceptable = df[mask]
        if len(acceptable) == 0:
            continue
        pareto_idx = acceptable[x_column].argmin()
        pareto_rows.append(acceptable.iloc[pareto_idx])
    return pd.DataFrame(pareto_rows).drop_duplicates()
