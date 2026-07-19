from __future__ import annotations

import itertools
import math
from collections import defaultdict

import pandas as pd
from scipy.stats import chi2_contingency

from biosift.models import ColumnMapping, ValidationIssue


def _source_for(mappings: list[ColumnMapping], standard_field: str) -> str | None:
    for mapping in mappings:
        if mapping.standard_field == standard_field:
            return mapping.source_column
    return None


def _cramers_v(series_a: pd.Series, series_b: pd.Series) -> float:
    table = pd.crosstab(series_a, series_b)
    if table.empty or min(table.shape) < 2:
        return 0.0

    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    if n == 0:
        return 0.0

    phi2 = chi2 / n
    r, k = table.shape
    denominator = min(k - 1, r - 1)
    if denominator <= 0:
        return 0.0
    return float(math.sqrt(phi2 / denominator))


def _conditional_purity(series_a: pd.Series, series_b: pd.Series) -> float:
    table = pd.crosstab(series_a, series_b)
    if table.empty:
        return 0.0
    majority = table.max(axis=0).sum()
    total = table.to_numpy().sum()
    return float(majority / total) if total else 0.0


def validate_metadata(df: pd.DataFrame, mappings: list[ColumnMapping]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    mapped = {mapping.standard_field: mapping.source_column for mapping in mappings if mapping.standard_field != "unknown"}

    # Missing values in mapped fields.
    for field, column in mapped.items():
        if column not in df.columns:
            continue
        missing = int(df[column].isna().sum())
        if missing:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    title=f"Missing values in {field}",
                    description=f"{missing} row(s) are missing values in '{column}', mapped to {field}.",
                    recommendation="Resolve missing experimental labels before downstream modeling or explicitly document an imputation policy.",
                    evidence=[f"Missing fraction: {df[column].isna().mean():.1%}"],
                )
            )

    # Duplicate sample identifiers.
    sample_col = mapped.get("sample_id")
    if sample_col and sample_col in df.columns:
        duplicates = df.loc[df[sample_col].duplicated(keep=False), sample_col].dropna().astype(str).tolist()
        if duplicates:
            issues.append(
                ValidationIssue(
                    severity="critical",
                    title="Duplicate sample identifiers",
                    description=f"BioSift found {len(set(duplicates))} duplicated sample identifier value(s).",
                    recommendation="Assign a unique identifier to every biological sample and verify that duplicated rows are not accidental copies.",
                    evidence=[", ".join(sorted(set(duplicates))[:8])],
                )
            )

    # Subject leakage across dataset split.
    subject_col = mapped.get("subject_id")
    split_col = mapped.get("split")
    if subject_col and split_col and subject_col in df.columns and split_col in df.columns:
        grouped = df.dropna(subset=[subject_col, split_col]).groupby(subject_col)[split_col].nunique()
        leaked = grouped[grouped > 1].index.astype(str).tolist()
        if leaked:
            issues.append(
                ValidationIssue(
                    severity="critical",
                    title="Subject leakage across train and test sets",
                    description=f"{len(leaked)} subject(s) appear in multiple dataset partitions.",
                    recommendation="Create train, validation, and test splits at the subject level rather than the sample level.",
                    evidence=[f"Leaked subjects: {', '.join(leaked[:10])}"],
                )
            )

    # Treatment or condition confounding with batch.
    batch_col = mapped.get("batch")
    if batch_col and batch_col in df.columns:
        for factor in ["treatment", "condition"]:
            factor_col = mapped.get(factor)
            if not factor_col or factor_col not in df.columns:
                continue
            subset = df[[factor_col, batch_col]].dropna()
            if subset.empty:
                continue
            strength = _cramers_v(subset[factor_col], subset[batch_col])
            purity = _conditional_purity(subset[factor_col], subset[batch_col])
            if strength >= 0.65 or purity >= 0.85:
                issues.append(
                    ValidationIssue(
                        severity="critical",
                        title=f"Possible {factor}-batch confounding",
                        description=(
                            f"{factor.title()} is strongly associated with processing batch. "
                            "A downstream model may learn technical batch differences instead of biology."
                        ),
                        recommendation="Rebalance the study if possible, include batch in the statistical model, and visualize batch effects before interpreting treatment or condition signals.",
                        evidence=[f"Cramér's V: {strength:.2f}", f"Batch-to-factor majority purity: {purity:.1%}"],
                    )
                )

    # Inconsistent categories when multiple raw labels collapse to the same normalized label.
    for mapping in mappings:
        reverse: dict[str, list[str]] = defaultdict(list)
        for raw, normalized in mapping.normalized_values.items():
            reverse[normalized].append(raw)
        for normalized, raw_values in reverse.items():
            if len(raw_values) > 1:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        title=f"Inconsistent labels in {mapping.standard_field}",
                        description=f"Multiple raw labels appear to represent the same concept: {', '.join(raw_values)}.",
                        recommendation=f"Standardize these values to '{normalized}' and preserve the transformation in the audit log.",
                        evidence=[f"Suggested normalization: {raw_values} -> {normalized}"],
                    )
                )

    # Missing combinations among key experimental factors.
    factor_fields = [field for field in ["condition", "treatment", "timepoint"] if field in mapped]
    if len(factor_fields) >= 2:
        columns = [mapped[field] for field in factor_fields]
        subset = df[columns].dropna().astype(str)
        if not subset.empty:
            levels = [sorted(subset[column].unique().tolist()) for column in columns]
            expected = set(itertools.product(*levels))
            observed = set(map(tuple, subset[columns].drop_duplicates().to_numpy()))
            missing_combinations = sorted(expected - observed)
            if missing_combinations:
                rendered = [" | ".join(combo) for combo in missing_combinations[:8]]
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        title="Missing experimental combinations",
                        description=f"{len(missing_combinations)} combination(s) of the observed factor levels are absent.",
                        recommendation="Confirm whether these groups were intentionally omitted. Missing cells can make treatment, time, and condition effects non-identifiable.",
                        evidence=rendered,
                    )
                )

    # Low-confidence mappings.
    low_confidence = [mapping.source_column for mapping in mappings if mapping.needs_review]
    if low_confidence:
        issues.append(
            ValidationIssue(
                severity="info",
                title="Mappings need scientist review",
                description=f"{len(low_confidence)} mapping(s) were marked for human confirmation.",
                recommendation="Review uncertain mappings before generating the final standardized package.",
                evidence=[", ".join(low_confidence)],
            )
        )

    return issues
