from __future__ import annotations

import json

import pandas as pd

from biosift.models import ColumnMapping


def apply_mappings(df: pd.DataFrame, mappings: list[ColumnMapping]) -> pd.DataFrame:
    cleaned = df.copy()

    rename_map = {
        mapping.source_column: mapping.standard_field
        for mapping in mappings
        if mapping.standard_field != "unknown" and mapping.source_column in cleaned.columns
    }
    cleaned = cleaned.rename(columns=rename_map)

    for mapping in mappings:
        target = mapping.standard_field
        if target == "unknown" or target not in cleaned.columns:
            continue
        if mapping.normalized_values:
            cleaned[target] = cleaned[target].replace(mapping.normalized_values)

    return cleaned


def mappings_to_dataframe(mappings: list[ColumnMapping]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_column": mapping.source_column,
                "standard_field": mapping.standard_field,
                # Streamlit's ProgressColumn applies numeric formatting directly.
                # Store a true 0-100 display value here so 0.95 renders as 95%,
                # while the underlying ColumnMapping confidence remains 0-1.
                "evidence_score_percent": int(round(mapping.confidence * 100)),
                "needs_review": mapping.needs_review,
                "reasoning": mapping.reasoning_summary,
                "normalized_values_json": json.dumps(mapping.normalized_values, ensure_ascii=False),
            }
            for mapping in mappings
        ]
    )


def dataframe_to_mappings(edited: pd.DataFrame, original: list[ColumnMapping]) -> list[ColumnMapping]:
    original_by_source = {mapping.source_column: mapping for mapping in original}
    updated: list[ColumnMapping] = []

    for _, row in edited.iterrows():
        source = str(row["source_column"])
        old = original_by_source[source]

        try:
            normalized_values = json.loads(str(row.get("normalized_values_json", "{}")))
            if not isinstance(normalized_values, dict):
                normalized_values = old.normalized_values
        except json.JSONDecodeError:
            normalized_values = old.normalized_values

        updated.append(
            old.model_copy(
                update={
                    "standard_field": str(row["standard_field"]),
                    "needs_review": bool(row["needs_review"]),
                    "normalized_values": {str(k): str(v) for k, v in normalized_values.items()},
                }
            )
        )

    return updated
