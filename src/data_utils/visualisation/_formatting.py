from __future__ import annotations
import matplotlib.pyplot as plt
import torch
import math
from typing import Optional
import numpy as np
import pandas as pd
from py_markdown_table.markdown_table import markdown_table as _markdown_table  # type: ignore


def markdown_table(data: dict[str, list] | pd.DataFrame) -> str:
    """Create a markdown table from a dict of lists or DataFrame (must contain one entry per data point).

    The list is formatted for usage in Obsidian."""
    if isinstance(data, dict):
        data = pd.DataFrame(data)
    md = _markdown_table(data.to_dict(orient="records")).get_markdown()
    cleaned_table = (
        md.replace("+", "|")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
        .split("\n")[1:-1]
    )
    if len(cleaned_table) <= 3:
        raise ValueError(f"Table has no content, input {data}.")
    no_separators = cleaned_table[2::2]
    return "\n".join(cleaned_table[:2] + no_separators)
