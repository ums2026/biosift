from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd


def read_table(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read a CSV or TSV upload into a dataframe."""
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else "csv"
    if suffix in {"tsv", "txt"}:
        return pd.read_csv(io.BytesIO(file_bytes), sep="\t")
    return pd.read_csv(io.BytesIO(file_bytes))


def _safe_value(value: Any) -> str:
    if pd.isna(value):
        return "<missing>"
    text = str(value)
    return text[:120]


def profile_dataframe(df: pd.DataFrame, max_values: int = 8) -> dict[str, Any]:
    profiles: list[dict[str, Any]] = []

    for column in df.columns:
        series = df[column]
        non_missing = series.dropna()
        unique_values = non_missing.astype(str).unique().tolist()[:max_values]

        inferred_kind = "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical"
        if inferred_kind == "categorical" and non_missing.nunique() > max(30, len(df) * 0.6):
            inferred_kind = "identifier_or_text"

        item: dict[str, Any] = {
            "name": str(column),
            "dtype": str(series.dtype),
            "inferred_kind": inferred_kind,
            "missing_fraction": round(float(series.isna().mean()), 4),
            "unique_count": int(non_missing.nunique()),
            "sample_values": [_safe_value(value) for value in unique_values],
        }

        if pd.api.types.is_numeric_dtype(series) and len(non_missing) > 0:
            numeric = pd.to_numeric(non_missing, errors="coerce").dropna()
            if len(numeric) > 0:
                item["numeric_summary"] = {
                    "min": float(np.min(numeric)),
                    "median": float(np.median(numeric)),
                    "max": float(np.max(numeric)),
                }

        profiles.append(item)

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing_cells": int(df.isna().sum().sum()),
        "column_profiles": profiles,
    }
