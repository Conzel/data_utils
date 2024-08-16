from typing import Iterable, Literal
import pandas as pd


def pareto_front(df: pd.DataFrame, x_column: str, y_column: str, threshhold_values: Iterable[float], mode: Literal["over", "under"]) -> pd.DataFrame:
    pareto_rows = []
    for acc_threshhold in threshhold_values:
        mask = df[y_column] < acc_threshhold if mode == "under" else df[y_column] > acc_threshhold
        acceptable = df[mask]
        if len(acceptable) == 0:
            continue
        pareto_idx = acceptable[x_column].argmin()
        pareto_rows.append(acceptable.iloc[pareto_idx])
    return pd.DataFrame(pareto_rows).drop_duplicates()