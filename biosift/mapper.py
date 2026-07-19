from __future__ import annotations

import json
import os
import re
from typing import Any

from pydantic import BaseModel

from biosift.models import AIAnalysis, AllowedField, ColumnMapping, StudyDesign

DEFAULT_MODEL = "claude-opus-4-8"

# The passthrough column-level fields BioSift transmits to a model. These are
# schema descriptors, not data: a column name plus non-identifying aggregate
# statistics. Cell values (`sample_values`), numeric summaries, and protocol text
# are deliberately excluded so that no experimental or patient data leaves the
# machine. sanitize_profile_for_cloud() adds one more derived key, "value_pattern",
# which is a content-free fingerprint of value shapes (see _column_value_pattern).
CLOUD_SAFE_COLUMN_FIELDS = ("name", "dtype", "inferred_kind", "missing_fraction", "unique_count")

# Recognized measurement-unit tokens, used only to set a boolean flag in the
# value fingerprint (the tokens themselves are never transmitted).
_UNIT_TOKEN_RE = re.compile(
    r"(?<![a-z0-9])(nm|µm|um|mm|cm|mg|ug|ng|kg|ml|ul|hr|hrs|h|min|sec|s|d|day|days|wk|week|weeks|mol|mmol|nmol|percent)(?![a-z0-9])",
    re.IGNORECASE,
)


ALIASES: dict[str, set[str]] = {
    "sample_id": {"sample", "sampleid", "sample_id", "sid", "specimen", "specimen_id"},
    "subject_id": {
        "pt",
        "patient",
        "patient_id",
        "subject",
        "subject_id",
        "mouse",
        "animal_id",
        "donor",
        "donor_id",
        "organoid",
        "organoid_id",
        "line_id",
    },
    "condition": {"grp", "group", "condition", "disease", "phenotype", "status", "genotype"},
    "treatment": {"tx", "treatment", "drug", "compound", "arm", "stim", "stimulation"},
    "dose": {"dose", "dose_um", "dose_u_m", "dose_nm", "concentration", "conc"},
    "timepoint": {"tm", "time", "timepoint", "day", "collection_time", "visit"},
    "batch": {"lib", "batch", "library", "plate", "run", "sequencing_batch"},
    "biological_replicate": {"rep", "replicate", "bio_rep", "biological_replicate"},
    "technical_replicate": {"tech_rep", "technical_replicate"},
    "sex": {"sex", "gender"},
    "age": {"age", "age_years", "age_days"},
    "outcome": {"outcome", "label", "target", "response", "viability"},
    "split": {"set", "split", "partition", "fold"},
    "organism": {"organism", "species"},
    "tissue": {"tissue", "organ", "site"},
}

FIELD_PROTOCOL_TERMS: dict[str, set[str]] = {
    "sample_id": {"sample identifier", "sample id"},
    "subject_id": {"subject", "patient", "donor", "animal", "organoid line"},
    "condition": {"condition", "status", "genotype", "phenotype"},
    "treatment": {"treatment", "drug", "compound", "vehicle"},
    "dose": {"dose", "concentration", "nanomolar", "micromolar"},
    "timepoint": {"time point", "timepoint", "baseline", "hours", "days"},
    "batch": {"batch", "library preparation", "sequencing run"},
    "biological_replicate": {"biological replicate", "replicate"},
    "technical_replicate": {"technical replicate"},
    "sex": {"sex", "gender"},
    "age": {"age"},
    "outcome": {"outcome", "response", "viability"},
    "split": {"train", "test", "validation", "machine-learning split"},
    "organism": {"organism", "species"},
    "tissue": {"tissue", "organoid", "organ"},
}

CONFIDENCE_WEIGHTS = {
    "column_name": 0.45,
    "observed_values": 0.30,
    "protocol_support": 0.20,
    "data_type": 0.05,
}


class ClaudeColumnMapping(BaseModel):
    """The only decision Claude makes: which standard field a column maps to.

    Claude sees column names and structural statistics only. It never receives
    cell values or protocol text, so it does not score observed-value or
    protocol evidence and does not propose value normalizations — those are
    computed locally from data that stays on the machine.
    """

    source_column: str
    standard_field: AllowedField
    reasoning_summary: str


class ClaudeMappingResult(BaseModel):
    mappings: list[ClaudeColumnMapping]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _name_signal(column_name: str, field: str) -> float:
    name = _slug(column_name)
    aliases = ALIASES.get(field, set())
    if name in aliases:
        return 1.0
    if any(alias in name or name in alias for alias in aliases if len(alias) >= 3 and len(name) >= 2):
        return 0.7
    return 0.0


def _identifier_signal(column: dict[str, Any]) -> float:
    rows = max(int(column.get("row_count", 0)), int(column.get("unique_count", 0)), 1)
    unique_ratio = int(column.get("unique_count", 0)) / rows
    samples = [str(v) for v in column.get("sample_values", []) if v != "<missing>"]
    pattern_ratio = sum(bool(re.search(r"\d", value)) for value in samples) / max(len(samples), 1)
    return min(1.0, 0.65 * unique_ratio + 0.35 * pattern_ratio)


def _value_signal(column: dict[str, Any], field: str, protocol_text: str) -> float:
    values = [str(v).strip().lower() for v in column.get("sample_values", []) if v != "<missing>"]
    value_blob = " ".join(values)
    protocol_compact = _compact(protocol_text)

    if field in {"sample_id", "subject_id"}:
        return _identifier_signal(column)
    if field == "split":
        known = {"train", "test", "val", "validation", "holdout"}
        return 1.0 if values and all(_slug(v) in known for v in values) else 0.0
    if field == "sex":
        known = {"m", "f", "male", "female"}
        return 1.0 if values and all(_slug(v) in known for v in values) else 0.0
    if field == "timepoint":
        pattern = re.compile(r"^(bl|baseline|d\d+|day\s*\d+|\d+\s*[hdw])$", re.I)
        matches = sum(bool(pattern.match(v.replace(" ", ""))) for v in values)
        return matches / max(len(values), 1)
    if field == "dose":
        if column.get("inferred_kind") == "numeric":
            return 0.9
        pattern = re.compile(r"^\d+(\.\d+)?\s*(nm|um|µm|mg|ug|ng)?$", re.I)
        return sum(bool(pattern.match(v.replace(" ", ""))) for v in values) / max(len(values), 1)
    if field == "batch":
        if values and len(values) <= 12:
            return 0.8 if all(re.search(r"batch|run|plate|lib|^\d+$", v, re.I) for v in values) else 0.45
    if field in {"biological_replicate", "technical_replicate"}:
        return 1.0 if values and all(re.fullmatch(r"(?:r|rep)?\d+", _compact(v)) for v in values) else 0.5
    if field == "treatment":
        generic = {"vehicle", "veh", "control", "drug", "compound", "tram", "trametinib", "meki"}
        direct = any(token in value_blob for token in generic)
        protocol_match = any(_compact(v) in protocol_compact for v in values if len(_compact(v)) >= 3)
        return 1.0 if direct or protocol_match else 0.35 if column.get("inferred_kind") == "categorical" else 0.0
    if field == "condition":
        generic = {"control", "ctrl", "case", "wild", "mutant", "mut", "disease", "treated", "untreated"}
        direct = any(token in value_blob for token in generic)
        protocol_match = any(_compact(v) in protocol_compact for v in values if len(_compact(v)) >= 3)
        return 1.0 if direct or protocol_match else 0.35 if column.get("inferred_kind") == "categorical" else 0.0
    if field in {"organism", "tissue"}:
        return 0.7 if column.get("inferred_kind") == "categorical" else 0.0
    if field == "outcome":
        return 0.75 if column.get("inferred_kind") == "numeric" else 0.45
    if field == "age":
        return 0.8 if column.get("inferred_kind") == "numeric" else 0.3
    return 0.0


def _protocol_signal(column: dict[str, Any], field: str, protocol_text: str) -> float:
    if not protocol_text.strip():
        return 0.0

    lower = protocol_text.lower()
    compact_protocol = _compact(protocol_text)
    terms = FIELD_PROTOCOL_TERMS.get(field, set())
    term_match = any(term in lower for term in terms)

    values = [str(v) for v in column.get("sample_values", []) if v != "<missing>"]
    value_match = any(_compact(v) in compact_protocol for v in values if len(_compact(v)) >= 3)

    if term_match and value_match:
        return 1.0
    if term_match:
        return 0.7
    if value_match:
        return 0.45
    return 0.0


def _type_signal(column: dict[str, Any], field: str) -> float:
    kind = column.get("inferred_kind")
    if field in {"sample_id", "subject_id"}:
        return 1.0 if kind == "identifier_or_text" else 0.7
    if field in {"dose", "age", "outcome"}:
        return 1.0 if kind == "numeric" else 0.4
    if field in {"condition", "treatment", "timepoint", "batch", "sex", "split", "biological_replicate", "technical_replicate"}:
        return 1.0 if kind in {"categorical", "identifier_or_text"} else 0.35
    return 0.5


def calculate_confidence(
    column: dict[str, Any],
    field: str,
    protocol_text: str,
) -> tuple[float, dict[str, float]]:
    """Calculate a transparent evidence score, not a calibrated probability."""
    if field == "unknown":
        components = {name: 0.0 for name in CONFIDENCE_WEIGHTS}
        return 0.0, components

    components = {
        "column_name": _name_signal(column["name"], field),
        "observed_values": _value_signal(column, field, protocol_text),
        "protocol_support": _protocol_signal(column, field, protocol_text),
        "data_type": _type_signal(column, field),
    }
    score = sum(CONFIDENCE_WEIGHTS[name] * components[name] for name in CONFIDENCE_WEIGHTS)
    return round(max(0.0, min(1.0, score)), 3), {name: round(value, 3) for name, value in components.items()}


def _guess_field(column: dict[str, Any], protocol_text: str) -> tuple[str, str]:
    name = _slug(column["name"])
    values = [str(v).lower() for v in column.get("sample_values", [])]

    best_field = "unknown"
    best_score = 0.0
    for field, aliases in ALIASES.items():
        if name in aliases:
            return field, f"Column name '{column['name']}' matches a known alias for {field}."
        score = max((0.75 if alias in name or name in alias else 0.0) for alias in aliases)
        if score > best_score:
            best_field = field
            best_score = score

    value_blob = " ".join(values)
    compact_protocol = _compact(protocol_text)
    if any(token in value_blob for token in ["train", "test", "validation", "val"]):
        return "split", "Values resemble machine-learning partitions."
    if any(token in value_blob for token in ["vehicle", "veh", "trametinib", "tram", "meki", "drug"]):
        return "treatment", "Values resemble experimental treatment labels."
    if any(token in value_blob for token in ["control", "ctrl", "wild", "mutant", "case"]):
        return "condition", "Values resemble biological condition or genotype labels."
    if any(token in value_blob for token in ["baseline", "day1", "day3", "24h", "72h", "d1", "d3"]):
        return "timepoint", "Values resemble experimental time points."
    if all(re.fullmatch(r"[mf]", value) for value in values if value != "<missing>") and values:
        return "sex", "Values are consistent with sex labels."

    for field in ["treatment", "condition"]:
        if any(_compact(value) in compact_protocol for value in values if value != "<missing>" and len(_compact(value)) >= 3):
            return field, "Observed values are explicitly referenced in the supplied protocol."

    return best_field, "Mapped using column-name similarity and observed value patterns."


def _normalize_value(field: str, value: str) -> str:
    raw = str(value).strip()
    key = _compact(raw)

    dictionaries: dict[str, dict[str, str]] = {
        "condition": {
            "wt": "KRAS wild-type",
            "kraswt": "KRAS wild-type",
            "wildtype": "KRAS wild-type",
            "mut": "KRAS mutant",
            "krasmut": "KRAS mutant",
            "mutant": "KRAS mutant",
        },
        "treatment": {
            "veh": "vehicle",
            "vehicle": "vehicle",
            "vehiclecontrol": "vehicle",
            "tram": "trametinib",
            "trametinib": "trametinib",
            "meki": "trametinib",
        },
        "timepoint": {
            "bl": "0 h",
            "0h": "0 h",
            "baseline": "0 h",
            "d1": "24 h",
            "24h": "24 h",
            "day1": "24 h",
            "d3": "72 h",
            "72h": "72 h",
            "day3": "72 h",
        },
        "batch": {"1": "Batch 1", "2": "Batch 2", "batch1": "Batch 1", "batch2": "Batch 2"},
        "sex": {"m": "male", "male": "male", "f": "female", "female": "female"},
        "split": {"train": "train", "test": "test", "val": "validation", "validation": "validation"},
    }

    return dictionaries.get(field, {}).get(key, raw)


def _local_normalized_values(field: str, column: dict[str, Any]) -> dict[str, str]:
    """Build raw-to-canonical replacements from a column's observed values.

    Runs entirely on local data — the sample values it reads are never sent to a
    model. Used by both the offline mapper and the Claude-assisted path.
    """
    normalized_values: dict[str, str] = {}
    for value in column.get("sample_values", []):
        if value == "<missing>":
            continue
        normalized = _normalize_value(field, value)
        if normalized != value:
            normalized_values[value] = normalized
    return normalized_values


def _heuristic_study_design(protocol_text: str) -> StudyDesign:
    lower = protocol_text.lower()
    compact = _compact(protocol_text)

    organism = "Homo sapiens" if any(x in lower for x in ["human", "patient-derived", "homo sapiens"]) else "Unknown"
    tissue = "colorectal tumor organoid" if "colorectal" in lower and "organoid" in lower else "Unknown"

    conditions = []
    if "wild-type" in lower or "wild type" in lower:
        conditions.append("KRAS wild-type")
    if "mutant" in lower:
        conditions.append("KRAS mutant")

    treatments = []
    if "vehicle" in lower:
        treatments.append("vehicle")
    if "trametinib" in lower:
        treatments.append("trametinib")

    timepoints = []
    if "baseline" in lower:
        timepoints.append("0 h")
    if "24 hour" in lower or "24 h" in lower:
        timepoints.append("24 h")
    if "72 hour" in lower or "72 h" in lower:
        timepoints.append("72 h")

    return StudyDesign(
        title="MEK inhibitor response in colorectal cancer organoids" if "trametinib" in compact else "Uploaded biomedical study",
        organism=organism,
        tissue=tissue,
        conditions=conditions,
        treatments=treatments,
        doses=["0 nM", "100 nM"] if "100 nm" in lower or "100 nanomolar" in lower else [],
        timepoints=timepoints,
        batches=["Batch 1", "Batch 2"] if "batch 1" in lower and "batch 2" in lower else [],
        experimental_factors=["condition", "treatment", "timepoint", "batch"],
        summary="A patient-derived organoid study evaluating MEK-inhibitor response across KRAS status and collection time while controlling for batch.",
    )


def _profile_by_name(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for column in profile["column_profiles"]:
        enriched = dict(column)
        enriched["row_count"] = profile.get("rows", 0)
        result[column["name"]] = enriched
    return result


def _calibrate_mapping(mapping: ColumnMapping, column: dict[str, Any], protocol_text: str) -> ColumnMapping:
    confidence, components = calculate_confidence(column, mapping.standard_field, protocol_text)
    breakdown = (
        "Confidence evidence: "
        f"column name {components['column_name']:.0%}, "
        f"observed values {components['observed_values']:.0%}, "
        f"protocol support {components['protocol_support']:.0%}, "
        f"data type {components['data_type']:.0%}."
    )
    evidence = [item for item in mapping.evidence if not item.startswith("Confidence evidence:")]
    evidence.append(breakdown)
    return mapping.model_copy(
        update={
            "confidence": confidence,
            "confidence_components": components,
            "evidence": evidence,
            "needs_review": mapping.needs_review or confidence < 0.80 or mapping.standard_field == "unknown",
        }
    )


def calibrate_analysis(analysis: AIAnalysis, profile: dict[str, Any], protocol_text: str) -> AIAnalysis:
    columns = _profile_by_name(profile)
    calibrated = []
    for mapping in analysis.mappings:
        column = columns.get(mapping.source_column)
        if column is None:
            calibrated.append(mapping.model_copy(update={"confidence": 0.0, "needs_review": True}))
        else:
            calibrated.append(_calibrate_mapping(mapping, column, protocol_text))
    return analysis.model_copy(update={"mappings": calibrated})


def heuristic_analysis(profile: dict[str, Any], protocol_text: str) -> AIAnalysis:
    mappings: list[ColumnMapping] = []

    for column in profile["column_profiles"]:
        field, reason = _guess_field(column, protocol_text)
        normalized_values = _local_normalized_values(field, column)

        mappings.append(
            ColumnMapping(
                source_column=column["name"],
                standard_field=field,
                confidence=0.0,
                reasoning_summary=reason,
                evidence=[
                    f"Observed values: {', '.join(column.get('sample_values', [])[:6])}",
                    "Protocol context was checked for matching experimental terminology.",
                ],
                normalized_values=normalized_values,
                needs_review=field == "unknown",
            )
        )

    raw = AIAnalysis(study_design=_heuristic_study_design(protocol_text), mappings=mappings)
    return calibrate_analysis(raw, profile, protocol_text)


def _mask_token(value: str, cap: int = 16) -> str:
    """Turn a value into a content-free format template.

    Every letter becomes 'x' and every digit becomes '9'; separators and
    whitespace are preserved. Runs of four or more identical classes collapse to
    a short marker so exact lengths of long free text or large numbers are not
    revealed. The result describes shape only and cannot contain any original
    letters or digits.
    """
    masked_chars = []
    for char in value:
        if char.isalpha():
            masked_chars.append("x")
        elif char.isdigit():
            masked_chars.append("9")
        elif char.isspace():
            masked_chars.append(" ")
        else:
            masked_chars.append(char)
    masked = "".join(masked_chars)
    masked = re.sub(r"x{4,}", "x…", masked)
    masked = re.sub(r"9{4,}", "9…", masked)
    return masked[:cap]


def _cardinality_bucket(unique_count: int, row_count: int) -> str:
    """De-identified description of how many distinct values a column holds."""
    if unique_count <= 1:
        return "constant"
    if unique_count == 2:
        return "binary"
    if row_count > 0 and unique_count / row_count >= 0.8:
        return "identifier_like"
    if unique_count <= 12:
        return "few"
    if unique_count <= 30:
        return "several"
    return "many"


def _column_value_pattern(column: dict[str, Any], row_count: int, max_templates: int = 6) -> dict[str, Any]:
    """Build a content-free fingerprint of a column's value shapes.

    Reads the real sample values locally and emits only masked format templates,
    a cardinality bucket, and boolean character-class flags. No original value
    survives — this is what lets Claude reason about value shape without the data
    leaving the machine.
    """
    raw_values = [str(value) for value in column.get("sample_values", []) if value != "<missing>"]

    templates: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        template = _mask_token(value)
        if template not in seen:
            seen.add(template)
            templates.append(template)
        if len(templates) >= max_templates:
            break

    blob = " ".join(raw_values)
    has_decimal = any(re.search(r"\d\.\d", value) for value in raw_values)

    if column.get("inferred_kind") == "numeric":
        value_dtype = "numeric_decimal" if has_decimal else "numeric_integer"
    else:
        value_dtype = "categorical_text"

    return {
        "cardinality": _cardinality_bucket(int(column.get("unique_count", 0)), row_count),
        "value_dtype": value_dtype,
        "masked_examples": templates,
        "has_alpha": any(char.isalpha() for char in blob),
        "has_digit": any(char.isdigit() for char in blob),
        "has_decimal": has_decimal,
        "has_unit_token": bool(_UNIT_TOKEN_RE.search(blob)),
        "has_whitespace": any(char.isspace() for char in blob),
        "has_separator": any(char in "-_/:|." for char in blob),
    }


def sanitize_profile_for_cloud(profile: dict[str, Any]) -> dict[str, Any]:
    """Reduce a full local profile to schema-only metadata safe to send to a model.

    Keeps column names and non-identifying aggregate statistics
    (CLOUD_SAFE_COLUMN_FIELDS) and adds a content-free value-shape fingerprint
    (value_pattern). Drops cell values, numeric summaries, and every dataset-level
    content field so that no experimental or patient data leaves the machine.
    This is the single choke point through which anything sent to Claude passes.
    """
    row_count = int(profile.get("rows", 0))
    safe_columns = []
    for column in profile["column_profiles"]:
        safe_column = {key: column[key] for key in CLOUD_SAFE_COLUMN_FIELDS if key in column}
        safe_column["value_pattern"] = _column_value_pattern(column, row_count)
        safe_columns.append(safe_column)

    return {
        "column_count": profile.get("columns", len(safe_columns)),
        "column_schema": safe_columns,
    }


def claude_analysis(
    profile: dict[str, Any],
    protocol_text: str,
    api_key: str,
    model: str,
) -> AIAnalysis:
    """Ask Claude to choose a standard field per column from schema only.

    Claude receives nothing but sanitized column schema (names, structural
    statistics, and content-free value-shape fingerprints). Value normalization,
    study-design reconstruction, and all evidence scoring are performed locally
    from data that never leaves the machine.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    safe_profile = sanitize_profile_for_cloud(profile)

    system_prompt = """
You are BioSift, an evidence-grounded biomedical metadata curation agent.

For privacy, you are given ONLY schema-level metadata about each column. You do
NOT see any cell values and you do NOT see the study protocol; those stay on the
user's machine. For each column you receive:
- name: the exact source column name
- dtype / inferred_kind / missing_fraction / unique_count: aggregate statistics
- value_pattern: a content-free fingerprint of the column's value shapes:
  - cardinality: constant | binary | few | several | many | identifier_like
  - value_dtype: numeric_integer | numeric_decimal | categorical_text
  - masked_examples: format templates where every letter is 'x' and every digit
    is '9' (separators kept, long runs collapsed with '…'). For example "x9"
    means a letter followed by a digit (like a day code), "99x" a two-digit
    number with a unit letter, "x…" a single word, "999.9" a decimal number.
  - has_alpha / has_digit / has_decimal / has_unit_token / has_whitespace /
    has_separator: boolean shape flags (has_unit_token means a measurement unit
    like nM, mg, or h was detected).

Use the name together with these value-shape fingerprints to infer the concept.
For example: binary categorical values often indicate sex or condition;
identifier_like cardinality indicates sample_id or subject_id; numeric values
with has_unit_token suggest dose; short "x9"/"99x" templates suggest timepoints.

Your single task: map every column to exactly one allowed standard field. When
the name and fingerprint are not enough to decide, return 'unknown' rather than
guessing.

Rules:
1. Return exactly one mapping per input column, preserving the exact source name.
2. Use 'unknown' when the evidence is insufficient.
3. reasoning_summary is a short, user-facing justification based only on the
   column name and value-shape fingerprint. Never claim to have seen actual
   values, and do not invent specific values.
4. Never make clinical or diagnostic claims.
""".strip()

    payload = {
        "allowed_standard_fields": [
            "sample_id",
            "subject_id",
            "organism",
            "tissue",
            "condition",
            "treatment",
            "dose",
            "timepoint",
            "batch",
            "biological_replicate",
            "technical_replicate",
            "sex",
            "age",
            "outcome",
            "split",
            "unknown",
        ],
        "column_schema": safe_profile["column_schema"],
    }

    response = client.messages.parse(
        model=model,
        max_tokens=8192,
        thinking={"type": "adaptive"},
        system=system_prompt,
        messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
        output_format=ClaudeMappingResult,
    )

    parsed = response.parsed_output
    if parsed is None:
        raise RuntimeError("Claude returned no parsed BioSift mapping.")

    returned_by_source = {mapping.source_column: mapping for mapping in parsed.mappings}
    mappings: list[ColumnMapping] = []

    for column in profile["column_profiles"]:
        source_column = column["name"]
        claude_mapping = returned_by_source.get(source_column)
        if claude_mapping is None:
            field = "unknown"
            reasoning = "Claude did not return a mapping for this column; it was preserved for review."
        else:
            field = claude_mapping.standard_field
            reasoning = claude_mapping.reasoning_summary

        mappings.append(
            ColumnMapping(
                source_column=source_column,
                standard_field=field,
                confidence=0.0,
                reasoning_summary=reasoning,
                evidence=[
                    "Claude selected this field from the column name and a content-free value-shape fingerprint.",
                    "Actual values and protocol text were evaluated locally and never sent to any model.",
                ],
                # Normalization is computed locally from the real values, on-machine.
                normalized_values=_local_normalized_values(field, column),
                needs_review=field == "unknown",
            )
        )

    # Study design is reconstructed locally from the protocol, which never leaves
    # the machine. calibrate_analysis computes all four evidence components locally.
    raw = AIAnalysis(study_design=_heuristic_study_design(protocol_text), mappings=mappings)
    return calibrate_analysis(raw, profile, protocol_text)


def analyze_metadata(
    profile: dict[str, Any],
    protocol_text: str,
    api_key: str | None = None,
    model: str | None = None,
    allow_fallback: bool = True,
) -> tuple[AIAnalysis, str]:
    """Use Claude for field mapping when a key is present, otherwise offline mode.

    In both paths, cell values and protocol text stay local: Claude only ever
    receives sanitized column schema, and every evidence score is computed
    on-machine.
    """
    model = model or os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)

    if api_key:
        try:
            result = claude_analysis(profile, protocol_text, api_key, model)
            return result, f"Claude field mapping + local evidence scoring ({model})"
        except Exception as exc:
            if not allow_fallback:
                raise
            fallback = heuristic_analysis(profile, protocol_text)
            return fallback, f"Offline fallback after Claude error: {type(exc).__name__}: {exc}"

    return heuristic_analysis(profile, protocol_text), "Offline evidence-guided mapper (no API key)"
