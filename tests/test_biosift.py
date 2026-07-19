from biosift.demo_data import DEMO_PROTOCOL, get_demo_dataframe
from biosift.mapper import CONFIDENCE_WEIGHTS, heuristic_analysis
from biosift.normalizer import apply_mappings, mappings_to_dataframe
from biosift.profiler import profile_dataframe
from biosift.validators import validate_metadata


def test_demo_analysis_maps_core_fields():
    df = get_demo_dataframe()
    profile = profile_dataframe(df)
    analysis = heuristic_analysis(profile, DEMO_PROTOCOL)
    mapped = {item.standard_field for item in analysis.mappings}

    assert "sample_id" in mapped
    assert "subject_id" in mapped
    assert "condition" in mapped
    assert "treatment" in mapped
    assert "timepoint" in mapped
    assert "batch" in mapped
    assert "split" in mapped


def test_demo_detects_planted_critical_issues():
    df = get_demo_dataframe()
    analysis = heuristic_analysis(profile_dataframe(df), DEMO_PROTOCOL)
    issues = validate_metadata(df, analysis.mappings)
    titles = {issue.title for issue in issues}

    assert "Duplicate sample identifiers" in titles
    assert "Subject leakage across train and test sets" in titles
    assert any("batch confounding" in title.lower() for title in titles)


def test_normalization_produces_standard_columns():
    df = get_demo_dataframe()
    analysis = heuristic_analysis(profile_dataframe(df), DEMO_PROTOCOL)
    cleaned = apply_mappings(df, analysis.mappings)

    assert "condition" in cleaned.columns
    assert "treatment" in cleaned.columns
    assert "timepoint" in cleaned.columns
    assert "batch" in cleaned.columns
    assert "trametinib" in set(cleaned["treatment"].dropna())


def test_confidence_is_deterministic_and_auditable():
    df = get_demo_dataframe()
    analysis = heuristic_analysis(profile_dataframe(df), DEMO_PROTOCOL)
    mapping = next(item for item in analysis.mappings if item.source_column == "Tx")

    assert abs(sum(CONFIDENCE_WEIGHTS.values()) - 1.0) < 1e-9
    assert 0.0 <= mapping.confidence <= 1.0
    assert set(mapping.confidence_components) == set(CONFIDENCE_WEIGHTS)
    expected = sum(
        CONFIDENCE_WEIGHTS[name] * mapping.confidence_components[name]
        for name in CONFIDENCE_WEIGHTS
    )
    assert abs(mapping.confidence - expected) < 0.002



def test_demo_is_colorectal_organoid_example():
    analysis = heuristic_analysis(profile_dataframe(get_demo_dataframe()), DEMO_PROTOCOL)
    assert "colorectal cancer organoids" in analysis.study_design.title.lower()
    assert analysis.study_design.treatments == ["vehicle", "trametinib"]
    assert analysis.study_design.organism == "Homo sapiens"


def test_evidence_score_display_uses_zero_to_hundred_scale():
    df = get_demo_dataframe()
    analysis = heuristic_analysis(profile_dataframe(df), DEMO_PROTOCOL)
    display = mappings_to_dataframe(analysis.mappings)

    tx_mapping = next(item for item in analysis.mappings if item.source_column == "Tx")
    tx_display = int(display.loc[display["source_column"] == "Tx", "evidence_score_percent"].iloc[0])

    assert tx_display == round(tx_mapping.confidence * 100)
    assert 0 <= tx_display <= 100


def test_cloud_payload_excludes_experimental_and_patient_data():
    import json

    from biosift.mapper import CLOUD_SAFE_COLUMN_FIELDS, sanitize_profile_for_cloud

    profile = profile_dataframe(get_demo_dataframe())
    safe = sanitize_profile_for_cloud(profile)
    blob = json.dumps(safe)

    # Structural fields plus a content-free value_pattern: no cell values or
    # numeric summaries survive.
    assert "sample_values" not in blob
    assert "numeric_summary" not in blob
    allowed_keys = set(CLOUD_SAFE_COLUMN_FIELDS) | {"value_pattern"}
    for column in safe["column_schema"]:
        assert set(column).issubset(allowed_keys)
        # Masked examples describe shape only: letters -> 'x', digits -> '9',
        # so no original letter or digit can survive in a template.
        for template in column["value_pattern"]["masked_examples"]:
            assert not any(char.isalpha() and char != "x" for char in template)
            assert not any(char.isdigit() and char != "9" for char in template)

    # No distinctive observed value from the dataset may leak into the payload.
    # Short numeric/single-character values (e.g. "1", "0.0", "M") are excluded
    # because they coincide with the aggregate statistics that are legitimately
    # sent; the leakage risk is distinctive value strings like donor IDs or drug
    # names.
    column_names = {str(column["name"]) for column in profile["column_profiles"]}

    def _distinctive(value: str) -> bool:
        return len(value) >= 4 and any(ch.isalpha() for ch in value) and value not in column_names

    observed_values = {
        str(value)
        for column in profile["column_profiles"]
        for value in column.get("sample_values", [])
        if value != "<missing>"
    }
    leaked = {value for value in observed_values if _distinctive(value) and value in blob}
    assert not leaked


def test_claude_scoring_is_computed_locally(monkeypatch):
    import biosift.mapper as mapper_module
    from biosift.mapper import ClaudeColumnMapping, ClaudeMappingResult

    profile = profile_dataframe(get_demo_dataframe())

    class _FakeParsed:
        def __init__(self, parsed):
            self.parsed_output = parsed

    class _FakeMessages:
        def parse(self, **kwargs):
            # The payload must never carry values or protocol text.
            payload = kwargs["messages"][0]["content"]
            assert "sample_values" not in payload
            assert "protocol_text" not in payload
            assert "trametinib" not in payload
            mappings = [
                ClaudeColumnMapping(
                    source_column=column["name"],
                    standard_field="treatment" if column["name"] == "Tx" else "unknown",
                    reasoning_summary="Chosen from the column name and structure.",
                )
                for column in profile["column_profiles"]
            ]
            return _FakeParsed(ClaudeMappingResult(mappings=mappings))

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            self.messages = _FakeMessages()

    class _FakeAnthropic:
        Anthropic = _FakeClient

    monkeypatch.setitem(__import__("sys").modules, "anthropic", _FakeAnthropic)

    analysis = mapper_module.claude_analysis(profile, DEMO_PROTOCOL, "test-key", "test-model")
    tx = next(item for item in analysis.mappings if item.source_column == "Tx")

    # Score was recomputed locally from the four weighted components, not left at 0.
    assert tx.standard_field == "treatment"
    assert tx.confidence > 0.0
    expected = sum(
        CONFIDENCE_WEIGHTS[name] * tx.confidence_components[name] for name in CONFIDENCE_WEIGHTS
    )
    assert abs(tx.confidence - expected) < 0.002


def test_api_key_automatically_selects_claude(monkeypatch):
    import biosift.mapper as mapper_module

    profile = profile_dataframe(get_demo_dataframe())
    expected = heuristic_analysis(profile, DEMO_PROTOCOL)
    called = {"value": False}

    def fake_claude_analysis(profile, protocol_text, api_key, model):
        called["value"] = True
        assert api_key == "test-key"
        return expected

    monkeypatch.setattr(mapper_module, "claude_analysis", fake_claude_analysis)
    result, engine = mapper_module.analyze_metadata(
        profile=profile,
        protocol_text=DEMO_PROTOCOL,
        api_key="test-key",
        model="test-model",
        allow_fallback=False,
    )

    assert called["value"] is True
    assert result == expected
    assert engine.startswith("Claude field mapping")
