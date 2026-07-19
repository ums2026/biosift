from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

AllowedField = Literal[
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
]

ALLOWED_FIELDS: list[str] = [
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
]


class StudyDesign(BaseModel):
    title: str = "Uploaded biomedical study"
    organism: str = "Unknown"
    tissue: str = "Unknown"
    conditions: list[str] = Field(default_factory=list)
    treatments: list[str] = Field(default_factory=list)
    doses: list[str] = Field(default_factory=list)
    timepoints: list[str] = Field(default_factory=list)
    batches: list[str] = Field(default_factory=list)
    experimental_factors: list[str] = Field(default_factory=list)
    summary: str = ""


class ColumnMapping(BaseModel):
    source_column: str
    standard_field: AllowedField
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_components: dict[str, float] = Field(default_factory=dict)
    reasoning_summary: str
    evidence: list[str] = Field(default_factory=list)
    normalized_values: dict[str, str] = Field(default_factory=dict)
    needs_review: bool = False

    @field_validator("source_column")
    @classmethod
    def source_column_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("source_column cannot be blank")
        return value


class AIAnalysis(BaseModel):
    study_design: StudyDesign
    mappings: list[ColumnMapping]


class ValidationIssue(BaseModel):
    severity: Literal["critical", "warning", "info"]
    title: str
    description: str
    recommendation: str
    evidence: list[str] = Field(default_factory=list)
