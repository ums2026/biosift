from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from biosift.demo_data import DEMO_PROTOCOL, get_demo_dataframe
from biosift.document_parser import extract_document_text
from biosift.exporter import build_export_zip
from biosift.mapper import analyze_metadata
from biosift.models import ALLOWED_FIELDS, AIAnalysis, ColumnMapping
from biosift.normalizer import apply_mappings, dataframe_to_mappings, mappings_to_dataframe
from biosift.profiler import profile_dataframe, read_table
from biosift.validators import validate_metadata

load_dotenv()


def get_config_value(name: str, default: str = "") -> str:
    """Read local environment variables or Streamlit Community Cloud secrets."""
    environment_value = os.getenv(name, "").strip()
    if environment_value:
        return environment_value
    try:
        secret_value = st.secrets.get(name, default)
    except Exception:
        secret_value = default
    return str(secret_value).strip()


st.set_page_config(
    page_title="BioSift",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container { padding-top: 2rem; padding-bottom: 3rem; }
.biosift-hero {
    padding: 1.4rem 1.6rem;
    border: 1px solid rgba(128,128,128,.22);
    border-radius: 18px;
    margin-bottom: 1.2rem;
    background: linear-gradient(135deg, rgba(88,101,242,.10), rgba(16,185,129,.08));
}
.biosift-kicker { letter-spacing: .08em; text-transform: uppercase; font-size: .78rem; opacity: .7; }
.issue-critical { border-left: 5px solid #d32f2f; padding-left: 1rem; }
.issue-warning { border-left: 5px solid #ed6c02; padding-left: 1rem; }
.issue-info { border-left: 5px solid #1976d2; padding-left: 1rem; }
.small-muted { color: rgba(128,128,128,.9); font-size: .9rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="biosift-hero">
  <div class="biosift-kicker">AI-native biomedical data intake</div>
  <h1 style="margin-bottom:.25rem">BioSift</h1>
  <p style="font-size:1.08rem; margin-bottom:0">
    Turn cryptic experimental metadata and study protocols into a standardized,
    reviewable, analysis-ready package.
  </p>
</div>
""",
    unsafe_allow_html=True,
)


def initialize_state() -> None:
    defaults = {
        "source_df": None,
        "protocol_text": "",
        "profile": None,
        "analysis": None,
        "engine_label": "",
        "mappings": None,
        "last_source": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()

with st.sidebar:
    st.header("Run BioSift")

    source_mode = st.radio(
        "Study source",
        ["One-click demo", "Upload my files"],
        help="The demo contains planted metadata problems so every important feature is visible.",
    )

    environment_key = get_config_value("ANTHROPIC_API_KEY")
    entered_key = st.text_input(
        "Anthropic API key (Claude)",
        type="password",
        placeholder="Uses ANTHROPIC_API_KEY from .env when left blank",
        help="When a key is available, BioSift asks Claude to choose the standard field for each column from the column names plus content-free value-shape fingerprints. Your actual data values and protocol text never leave this machine.",
    ).strip()
    api_key = entered_key or environment_key

    if api_key:
        key_source = "pasted key" if entered_key else ".env key"
        st.success(f"Claude mode active ({key_source})")
    else:
        st.info("No API key detected. BioSift will run in fully offline demo mode.")

    st.caption(
        "🔒 Privacy: Claude receives only column names, aggregate statistics, and "
        "content-free value-shape fingerprints (e.g. `x9`, `999.9`). Actual cell "
        "values, numeric summaries, and protocol text are never transmitted."
    )

    model = st.text_input(
        "Claude model",
        value=get_config_value("ANTHROPIC_MODEL", "claude-opus-4-8"),
        disabled=not bool(api_key),
        help="Any Claude model that supports Structured Outputs, e.g. claude-opus-4-8.",
    )

    allow_fallback = st.toggle(
        "Use offline fallback if Claude fails",
        value=True,
        disabled=not bool(api_key),
        help="Recommended for live demos. The engine label will clearly disclose when fallback was used.",
    )

    uploaded_metadata = None
    uploaded_protocol = None

    if source_mode == "Upload my files":
        uploaded_metadata = st.file_uploader("Metadata CSV or TSV", type=["csv", "tsv", "txt"])
        uploaded_protocol = st.file_uploader("Protocol PDF, TXT, or MD", type=["pdf", "txt", "md"])

    run_clicked = st.button("Analyze study", type="primary", use_container_width=True)

    st.divider()
    st.caption("Research demonstration only. BioSift does not provide clinical or diagnostic advice.")


if run_clicked:
    try:
        if source_mode == "One-click demo":
            df = get_demo_dataframe()
            protocol_text = DEMO_PROTOCOL
            source_label = "Built-in colorectal organoid drug-response demo"
        else:
            if uploaded_metadata is None:
                st.error("Upload a metadata CSV or TSV first.")
                st.stop()
            df = read_table(uploaded_metadata.getvalue(), uploaded_metadata.name)
            protocol_text = ""
            if uploaded_protocol is not None:
                protocol_text = extract_document_text(uploaded_protocol.getvalue(), uploaded_protocol.name)
            source_label = uploaded_metadata.name

        profile = profile_dataframe(df)

        spinner_text = (
            "Claude is mapping columns from schema only; values and protocol are scored locally..."
            if api_key
            else "Running the offline evidence-guided mapper..."
        )
        with st.spinner(spinner_text):
            analysis, engine_label = analyze_metadata(
                profile=profile,
                protocol_text=protocol_text,
                api_key=api_key or None,
                model=model,
                allow_fallback=allow_fallback,
            )

        st.session_state.source_df = df
        st.session_state.protocol_text = protocol_text
        st.session_state.profile = profile
        st.session_state.analysis = analysis
        st.session_state.engine_label = engine_label
        st.session_state.mappings = analysis.mappings
        st.session_state.last_source = source_label
        st.success("BioSift analysis complete.")

    except Exception as exc:
        st.exception(exc)


if st.session_state.source_df is None:
    st.info("Select the one-click demo in the sidebar and press **Analyze study**. The app will reveal a duplicate sample, subject leakage, label inconsistencies, a missing design cell, and batch confounding.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("1. Understand")
        st.write("Read cryptic columns together with protocol context.")
    with col2:
        st.subheader("2. Validate")
        st.write("Detect leakage, confounding, duplicates, and design gaps.")
    with col3:
        st.subheader("3. Export")
        st.write("Produce cleaned data, provenance, and reproducible code.")
    st.stop()


df: pd.DataFrame = st.session_state.source_df
profile: dict = st.session_state.profile
analysis: AIAnalysis = st.session_state.analysis
mappings: list[ColumnMapping] = st.session_state.mappings
claude_mapped = st.session_state.engine_label.startswith("Claude field mapping")

issues = validate_metadata(df, mappings)
cleaned_df = apply_mappings(df, mappings)

mapped_count = sum(mapping.standard_field != "unknown" for mapping in mappings)
review_count = sum(mapping.needs_review for mapping in mappings)
critical_count = sum(issue.severity == "critical" for issue in issues)

metric1, metric2, metric3, metric4, metric5 = st.columns(5)
metric1.metric("Samples", len(df))
metric2.metric("Columns", len(df.columns))
metric3.metric("Mapped fields", mapped_count)
metric4.metric("Needs review", review_count)
metric5.metric("Critical issues", critical_count)

st.caption(f"Source: {st.session_state.last_source} | Engine: {st.session_state.engine_label}")


overview_tab, mapping_tab, quality_tab, preview_tab, export_tab = st.tabs(
    ["Study overview", "Column mappings", "Quality report", "Before and after", "Export"]
)

with overview_tab:
    design = analysis.study_design
    left, right = st.columns([1.1, 1])

    with left:
        st.subheader(design.title)
        st.write(design.summary or "BioSift reconstructed this design from the uploaded files.")

        overview_rows = [
            ("Organism", design.organism),
            ("Tissue", design.tissue),
            ("Conditions", ", ".join(design.conditions) or "Unknown"),
            ("Treatments", ", ".join(design.treatments) or "Unknown"),
            ("Doses", ", ".join(design.doses) or "Not recovered"),
            ("Time points", ", ".join(design.timepoints) or "Not recovered"),
            ("Batches", ", ".join(design.batches) or "Not recovered"),
        ]
        st.dataframe(pd.DataFrame(overview_rows, columns=["Study element", "BioSift interpretation"]), hide_index=True, width="stretch")

    with right:
        st.subheader("Raw data profile")
        profile_rows = pd.DataFrame(profile["column_profiles"])[
            ["name", "inferred_kind", "missing_fraction", "unique_count", "sample_values"]
        ]
        st.dataframe(profile_rows, hide_index=True, width="stretch", height=420)

    with st.expander("View extracted protocol text"):
        st.text(st.session_state.protocol_text or "No protocol text was supplied.")

with mapping_tab:
    st.subheader("Review and edit AI-assisted mappings")
    st.write("Only the standard field, review flag, and normalization JSON are editable. Every change is reflected in the preview and export package.")

    with st.expander("How confidence is calculated"):
        if claude_mapped:
            st.markdown(
                """
The displayed score is a **locally computed evidence-strength score**, not a calibrated probability that the mapping is correct.

For privacy, Claude receives only column names, aggregate statistics, and content-free value-shape fingerprints (masked templates like `x9` or `999.9`, cardinality buckets, and character-class flags — never actual values). It does one thing: **choose the standard field for each column**. Everything that requires your actual data stays on this machine. The application computes all four evidence components locally, from your data and protocol:

- **45% column-name evidence**
- **30% observed-value evidence** *(computed on-machine)*
- **20% protocol support** *(computed on-machine)*
- **5% data-type compatibility**

The application performs the weighted arithmetic so the displayed total is internally consistent. Scores below 80% are automatically marked for review.
                """
            )
        else:
            st.markdown(
                """
No Claude key was available for this run. The displayed score is an **offline rule-based evidence-strength score**, not a calibrated probability.

The offline mapper both selects the field and evaluates the same four components using built-in aliases and value-pattern checks — entirely on this machine:

- **45% column-name evidence**
- **30% observed-value evidence**
- **20% protocol support**
- **5% data-type compatibility**

Add a Claude key in the sidebar or `.env` and rerun the study to have Claude select the semantic mapping. Even then, your data values and protocol never leave this machine.
                """
            )

    mapping_df = mappings_to_dataframe(mappings)
    edited_mapping_df = st.data_editor(
        mapping_df,
        hide_index=True,
        width="stretch",
        column_order=[
            "source_column",
            "standard_field",
            "evidence_score_percent",
            "needs_review",
            "reasoning",
            "normalized_values_json",
        ],
        disabled=["source_column", "evidence_score_percent", "reasoning"],
        column_config={
            "standard_field": st.column_config.SelectboxColumn("Standard field", options=ALLOWED_FIELDS, required=True),
            "evidence_score_percent": st.column_config.ProgressColumn(
                "Local evidence score",
                min_value=0,
                max_value=100,
                format="%d%%",
            ),
            "needs_review": st.column_config.CheckboxColumn("Needs review"),
            "normalized_values_json": st.column_config.TextColumn("Raw-to-canonical replacements", width="large"),
        },
        key="mapping_editor",
    )

    if st.button("Apply mapping edits"):
        try:
            st.session_state.mappings = dataframe_to_mappings(edited_mapping_df, mappings)
            st.success("Mapping edits applied.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not apply mapping edits: {exc}")

    st.divider()
    st.subheader("Evidence trail")
    for mapping in mappings:
        confidence_label = f"local evidence {mapping.confidence:.0%}"
        review_label = "Review" if mapping.needs_review else "Accepted"
        with st.expander(f"{mapping.source_column} → {mapping.standard_field} | {confidence_label} | {review_label}"):
            st.write(mapping.reasoning_summary)
            for evidence in mapping.evidence:
                st.write(f"• {evidence}")
            if mapping.confidence_components:
                st.markdown("**Locally computed evidence components**")
                component_labels = {
                    "column_name": "Column name",
                    "observed_values": "Observed values",
                    "protocol_support": "Protocol support",
                    "data_type": "Data type",
                }
                component_rows = [
                    {"Evidence source": component_labels.get(name, name), "Signal": f"{value:.0%}"}
                    for name, value in mapping.confidence_components.items()
                ]
                st.dataframe(pd.DataFrame(component_rows), hide_index=True, width="stretch")
            if mapping.normalized_values:
                st.json(mapping.normalized_values)

with quality_tab:
    st.subheader("Analysis-breaking issues")
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    sorted_issues = sorted(issues, key=lambda item: severity_order[item.severity])

    if not sorted_issues:
        st.success("No issues were detected by the current validation suite.")

    for issue in sorted_issues:
        icon = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}[issue.severity]
        st.markdown(f'<div class="issue-{issue.severity}">', unsafe_allow_html=True)
        st.markdown(f"### {icon} {issue.title}")
        st.write(issue.description)
        st.markdown(f"**Recommended action:** {issue.recommendation}")
        if issue.evidence:
            with st.expander("Evidence"):
                for item in issue.evidence:
                    st.write(f"• {item}")
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")

with preview_tab:
    st.subheader("Visible transformation")
    before_col, after_col = st.columns(2)
    with before_col:
        st.markdown("#### Before")
        st.dataframe(df.head(20), width="stretch", height=460)
    with after_col:
        st.markdown("#### After")
        st.dataframe(cleaned_df.head(20), width="stretch", height=460)

    st.subheader("Standardized category summaries")
    summary_rows = []
    for column in ["condition", "treatment", "timepoint", "batch", "split"]:
        if column in cleaned_df.columns:
            summary_rows.append(
                {
                    "field": column,
                    "values": ", ".join(cleaned_df[column].dropna().astype(str).value_counts().index.tolist()),
                    "missing": int(cleaned_df[column].isna().sum()),
                }
            )
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

with export_tab:
    st.subheader("Analysis-ready study package")
    st.write(
        "The ZIP includes the original metadata, standardized metadata, data dictionary, reconstructed study design, quality report, reproducible transformation script, and analysis configuration."
    )

    export_bytes = build_export_zip(
        original_df=df,
        cleaned_df=cleaned_df,
        mappings=mappings,
        study_design=analysis.study_design,
        issues=issues,
    )

    st.download_button(
        "Download BioSift package",
        data=export_bytes,
        file_name="biosift_analysis_package.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )

    st.markdown(
        """
**Package contents**

- `clean_metadata.csv`
- `data_dictionary.json`
- `study_design.json`
- `quality_report.html`
- `transformations.py`
- `analysis_config.yaml`
- `metadata.csv`
"""
    )
