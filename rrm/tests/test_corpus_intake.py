"""Tests for rrm.corpus.intake module.

Public API tested:
    collect_human_written_review
    HumanWrittenIntakeResult
"""

from __future__ import annotations

import dataclasses

import pytest

from rrm.corpus.annotation import (
    AnnotationSubmissionStore,
    ControlledProtocolTruthStore,
    DispositionRegister,
    ReannotationRegister,
)
from rrm.corpus.intake import (
    HumanWrittenIntakeResult,
    collect_human_written_review,
)
from rrm.corpus.models import (
    AnnotationStatus,
    ConsentArtifact,
    SourceType,
)
from rrm.corpus.workflow import CorpusWorkflow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_workflow() -> CorpusWorkflow:
    return CorpusWorkflow(
        AnnotationSubmissionStore(),
        ControlledProtocolTruthStore(),
        DispositionRegister(),
        ReannotationRegister(),
    )


# ---------------------------------------------------------------------------
# A — consent_granted=False raises ValueError, no side effects
# ---------------------------------------------------------------------------


class TestConsentRequired:
    def test_consent_false_raises_value_error(self):
        wf = _make_workflow()
        with pytest.raises(ValueError, match="consent"):
            collect_human_written_review(
                workflow=wf,
                submitted_text="hello world",
                consent_granted=False,
                collected_at="2025-01-01T00:00:00Z",
                collection_method="web_form",
            )

    def test_consent_false_does_not_create_record(self, monkeypatch):
        wf = _make_workflow()
        calls = []

        def _failing_create(*args, **kwargs):
            calls.append((args, kwargs))
            raise RuntimeError("create_record should not be called")

        monkeypatch.setattr(wf, "create_record", _failing_create)
        with pytest.raises(ValueError, match="consent"):
            collect_human_written_review(
                workflow=wf,
                submitted_text="hello world",
                consent_granted=False,
                collected_at="2025-01-01T00:00:00Z",
                collection_method="web_form",
            )
        assert len(calls) == 0

    def test_non_bool_consent_raises_type_error(self):
        wf = _make_workflow()
        with pytest.raises(TypeError, match="consent_granted"):
            collect_human_written_review(
                workflow=wf,
                submitted_text="hello",
                consent_granted="yes",
                collected_at="2025-01-01T00:00:00Z",
                collection_method="web_form",
            )


# ---------------------------------------------------------------------------
# B — input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_blank_submitted_text_raises(self):
        wf = _make_workflow()
        with pytest.raises(ValueError, match="submitted_text"):
            collect_human_written_review(
                workflow=wf,
                submitted_text="   ",
                consent_granted=True,
                collected_at="2025-01-01T00:00:00Z",
                collection_method="web_form",
            )

    def test_empty_string_submitted_text_raises(self):
        wf = _make_workflow()
        with pytest.raises(ValueError, match="submitted_text"):
            collect_human_written_review(
                workflow=wf,
                submitted_text="",
                consent_granted=True,
                collected_at="2025-01-01T00:00:00Z",
                collection_method="web_form",
            )

    def test_blank_collection_method_raises(self):
        wf = _make_workflow()
        with pytest.raises(ValueError, match="collection_method"):
            collect_human_written_review(
                workflow=wf,
                submitted_text="hello",
                consent_granted=True,
                collected_at="2025-01-01T00:00:00Z",
                collection_method="   ",
            )

    def test_empty_collection_method_raises(self):
        wf = _make_workflow()
        with pytest.raises(ValueError, match="collection_method"):
            collect_human_written_review(
                workflow=wf,
                submitted_text="hello",
                consent_granted=True,
                collected_at="2025-01-01T00:00:00Z",
                collection_method="",
            )

    def test_blank_collected_at_raises(self):
        wf = _make_workflow()
        with pytest.raises(ValueError, match="collected_at"):
            collect_human_written_review(
                workflow=wf,
                submitted_text="hello",
                consent_granted=True,
                collected_at="   ",
                collection_method="web_form",
            )

    def test_invalid_workflow_type_raises(self):
        with pytest.raises(TypeError, match="workflow"):
            collect_human_written_review(
                workflow="not a workflow",
                submitted_text="hello",
                consent_granted=True,
                collected_at="2025-01-01T00:00:00Z",
                collection_method="web_form",
            )

    def test_non_string_submitted_text_raises(self):
        wf = _make_workflow()
        with pytest.raises(TypeError, match="submitted_text"):
            collect_human_written_review(
                workflow=wf,
                submitted_text=123,
                consent_granted=True,
                collected_at="2025-01-01T00:00:00Z",
                collection_method="web_form",
            )

    def test_non_string_collected_at_raises(self):
        wf = _make_workflow()
        with pytest.raises(TypeError, match="collected_at"):
            collect_human_written_review(
                workflow=wf,
                submitted_text="hello",
                consent_granted=True,
                collected_at=123,
                collection_method="web_form",
            )

    def test_non_string_collection_method_raises(self):
        wf = _make_workflow()
        with pytest.raises(TypeError, match="collection_method"):
            collect_human_written_review(
                workflow=wf,
                submitted_text="hello",
                consent_granted=True,
                collected_at="2025-01-01T00:00:00Z",
                collection_method=456,
            )

    def test_non_bool_retain_source_text_raw_raises(self):
        wf = _make_workflow()
        with pytest.raises(TypeError, match="retain_source_text_raw"):
            collect_human_written_review(
                workflow=wf,
                submitted_text="hello",
                consent_granted=True,
                collected_at="2025-01-01T00:00:00Z",
                collection_method="web_form",
                retain_source_text_raw="yes",
            )

    def test_non_string_consent_evidence_ref_raises(self):
        wf = _make_workflow()
        with pytest.raises(TypeError, match="consent_evidence_ref"):
            collect_human_written_review(
                workflow=wf,
                submitted_text="hello",
                consent_granted=True,
                collected_at="2025-01-01T00:00:00Z",
                collection_method="web_form",
                consent_evidence_ref=123,
            )


# ---------------------------------------------------------------------------
# C — successful intake
# ---------------------------------------------------------------------------


class TestSuccessfulIntake:
    def test_returns_human_written_intake_result(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="Great college!",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert isinstance(result, HumanWrittenIntakeResult)

    def test_record_registered_in_workflow(self, monkeypatch):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="Great college!",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        # Verify by re-fetching through create_record's internal store
        fetched = wf.get_record(result.record.review_id)
        assert fetched is not None
        assert fetched.review_id == result.record.review_id

    def test_source_type_is_human_written(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="Great college!",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert result.record.source_type == SourceType.HUMAN_WRITTEN_RMC

    def test_annotation_status_is_unannotated(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="Great college!",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert result.record.annotation_status == AnnotationStatus.UNANNOTATED

    def test_all_six_task_labels_are_none(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="Great college!",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert result.record.spam is None
        assert result.record.deception is None
        assert result.record.toxicity is None
        assert result.record.advertising is None
        assert result.record.off_topic is None
        assert result.record.pii is None

    def test_language_mix_is_none(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="Great college!",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert result.record.language_mix is None

    def test_college_category_is_none(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="Great college!",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert result.record.college_category is None

    def test_annotation_metadata_unset(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="Great college!",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert result.record.annotation_guide_version is None
        assert result.record.annotator_A_id is None
        assert result.record.annotator_B_id is None
        assert result.record.adjudicator_id is None
        assert result.record.finalized_at is None
        assert result.record.annotation_notes is None

    def test_experiment_metadata_unset(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="Great college!",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert result.record.experiment_id is None
        assert result.record.controlled_targets is None
        assert result.record.deception_truth is None
        assert result.record.control_protocol_id is None

    def test_derivation_metadata_unset(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="Great college!",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert result.record.parent_review_id is None
        assert result.record.derivation_type is None
        assert result.record.generation_method is None

    def test_split_metadata_unset(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="Great college!",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert result.record.dataset_version is None
        assert result.record.split_membership is None
        assert result.record.split_group_id is None

    def test_review_id_generated_and_nonempty(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="Great college!",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert result.record.review_id
        assert result.record.review_id.startswith("hw-")
        suffix = result.record.review_id[len("hw-"):]
        assert len(suffix) == 32
        int(suffix, 16)

    def test_contributor_pseudonym_generated(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="Great college!",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert result.record.contributor_pseudonym
        assert result.record.contributor_pseudonym.startswith("contrib-")

    def test_review_ids_are_unique(self):
        wf = _make_workflow()
        r1 = collect_human_written_review(
            workflow=wf,
            submitted_text="Review one",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        r2 = collect_human_written_review(
            workflow=wf,
            submitted_text="Review two",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert r1.record.review_id != r2.record.review_id
        assert r1.record.contributor_pseudonym != r2.record.contributor_pseudonym

    def test_review_text_set_to_submitted_text_clean(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="Great college!",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert result.record.review_text == "Great college!"

    def test_consent_status_is_consented(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="Great college!",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert result.record.consent_status == "CONSENTED"

    def test_collection_method_set(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="Great college!",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="mobile_app",
        )
        assert result.record.collection_method == "mobile_app"

    def test_created_at_set(self):
        wf = _make_workflow()
        ts = "2025-06-15T10:30:00Z"
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="hello",
            consent_granted=True,
            collected_at=ts,
            collection_method="web_form",
        )
        assert result.record.created_at == ts


# ---------------------------------------------------------------------------
# D — raw retention
# ---------------------------------------------------------------------------


class TestRawRetention:
    def test_retain_true_sets_source_text_raw(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="hello world",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
            retain_source_text_raw=True,
        )
        assert result.record.source_text_raw == "hello world"

    def test_retain_false_sets_source_text_raw_none(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="hello world",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
            retain_source_text_raw=False,
        )
        assert result.record.source_text_raw is None

    def test_default_retain_is_false(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="hello world",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert result.record.source_text_raw is None


# ---------------------------------------------------------------------------
# E — PII handling
# ---------------------------------------------------------------------------


class TestPIIHandling:
    def test_clean_text_no_pii_evidence(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="Great college with nice campus",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert result.pii_evidence is None
        assert result.record.pii_detected is False
        assert result.record.pii_categories is None
        assert result.record.pii_redaction_status is None
        assert result.record.pii_evidence_id is None
        assert result.record.review_text == "Great college with nice campus"
        assert result.record.export_text_redacted == "Great college with nice campus"

    def test_email_pii_detected_original_removed(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="Contact me at student42@example.com please",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert result.pii_evidence is not None
        assert result.record.pii_detected is True
        assert "EMAIL" in result.record.pii_categories
        assert result.record.pii_redaction_status == "redacted_surrogate"
        assert result.record.pii_evidence_id is not None
        # Exact original email must not survive in safe outputs
        assert "student42@example.com" not in result.record.review_text
        assert "student42@example.com" not in result.record.export_text_redacted

    def test_phone_pii_detected_original_removed(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="Call me at 9876543210",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert result.pii_evidence is not None
        assert result.record.pii_detected is True
        assert "PHONE" in result.record.pii_categories
        assert result.record.pii_redaction_status == "redacted_surrogate"
        # Exact original phone must not survive in safe outputs
        assert "9876543210" not in result.record.review_text
        assert "9876543210" not in result.record.export_text_redacted

    def test_url_pii_detected_original_removed(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="See https://test.example.com/details for info",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert result.pii_evidence is not None
        assert result.record.pii_detected is True
        assert "URL" in result.record.pii_categories
        assert result.record.pii_redaction_status == "redacted_surrogate"
        # Exact original URL must not survive in safe outputs
        assert "https://test.example.com/details" not in result.record.review_text
        assert "https://test.example.com/details" not in (
            result.record.export_text_redacted
        )

    def test_multi_pii_original_values_removed(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text=(
                "Email me at student42@example.com, "
                "call 9876543210, or visit https://test.example.com"
            ),
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert result.record.pii_detected is True
        assert "EMAIL" in result.record.pii_categories
        assert "PHONE" in result.record.pii_categories
        assert "URL" in result.record.pii_categories
        assert len(result.record.pii_categories) == 3
        # All original private values absent from both safe outputs
        assert "student42@example.com" not in result.record.review_text
        assert "9876543210" not in result.record.review_text
        assert "https://test.example.com" not in result.record.review_text
        assert "student42@example.com" not in result.record.export_text_redacted
        assert "9876543210" not in result.record.export_text_redacted
        assert "https://test.example.com" not in (
            result.record.export_text_redacted
        )

    def test_pii_evidence_completed_fields(self):
        wf = _make_workflow()
        ts = "2025-01-01T00:00:00Z"
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="Contact student42@example.com",
            consent_granted=True,
            collected_at=ts,
            collection_method="web_form",
        )
        assert result.pii_evidence is not None
        assert result.pii_evidence.review_id == result.record.review_id
        assert result.pii_evidence.redaction_status == "redacted_surrogate"
        assert result.pii_evidence.created_at == ts
        assert result.record.pii_evidence_id == result.pii_evidence.evidence_id

    def test_pii_evidence_id_matches_summary(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="Contact student42@example.com",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert result.pii_evidence is not None
        assert result.record.pii_evidence_id == result.pii_evidence.evidence_id

    def test_pii_label_remains_none_at_intake(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="Contact student42@example.com",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        # The pii annotation label must remain None during intake
        assert result.record.pii is None


# ---------------------------------------------------------------------------
# F — post-transform detect_pii call
# ---------------------------------------------------------------------------


class TestPostTransformDetectPII:
    def test_post_transform_detect_pii_called_with_safe_text(self, monkeypatch):
        wf = _make_workflow()
        calls = []

        def _mock_detect_pii(text):
            calls.append(text)
            return ()

        monkeypatch.setattr(
            "rrm.corpus.intake.detect_pii", _mock_detect_pii
        )
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="Contact student42@example.com",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert len(calls) == 1
        # Called with the safe review_text, not the original private value
        assert calls[0] == result.record.review_text
        assert "student42@example.com" not in calls[0]


# ---------------------------------------------------------------------------
# G — consent artifact
# ---------------------------------------------------------------------------


class TestConsentArtifact:
    def test_consent_artifact_created(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="hello",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert isinstance(result.consent_artifact, ConsentArtifact)

    def test_consent_artifact_fields(self):
        wf = _make_workflow()
        ts = "2025-01-01T00:00:00Z"
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="hello",
            consent_granted=True,
            collected_at=ts,
            collection_method="web_form",
            consent_evidence_ref="ref-123",
        )
        art = result.consent_artifact
        assert art.review_id == result.record.review_id
        assert art.consent_status == "CONSENTED"
        assert art.collected_at == ts
        assert art.method == "web_form"
        assert art.evidence_ref == "ref-123"

    def test_consent_artifact_no_evidence_ref(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="hello",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert result.consent_artifact.evidence_ref is None

    def test_result_record_is_frozen(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="hello",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.record.review_id = "changed"

    def test_consent_artifact_is_frozen(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="hello",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.consent_artifact.consent_status = "REVOKED"


# ---------------------------------------------------------------------------
# H — Hinglish / Roman-Hindi examples
# ---------------------------------------------------------------------------


class TestHinglishExamples:
    def test_hinglish_text_preserved(self):
        wf = _make_workflow()
        text = "Bhai campus life mast hai bhai"
        result = collect_human_written_review(
            workflow=wf,
            submitted_text=text,
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert result.record.review_text == text

    def test_roman_hindi_preserved(self):
        wf = _make_workflow()
        text = "Ye college ka hostel life bahut accha hai"
        result = collect_human_written_review(
            workflow=wf,
            submitted_text=text,
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert result.record.review_text == text

    def test_mixed_script_no_transliteration(self):
        wf = _make_workflow()
        text = "The canteen food is good, par fees zyada hai"
        result = collect_human_written_review(
            workflow=wf,
            submitted_text=text,
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert result.record.review_text == text
        assert "par" in result.record.review_text
        assert "zyada" in result.record.review_text


# ---------------------------------------------------------------------------
# I — sequential intake
# ---------------------------------------------------------------------------


class TestSequentialIntake:
    def test_multiple_intakes_independent(self):
        wf = _make_workflow()
        r1 = collect_human_written_review(
            workflow=wf,
            submitted_text="College A is great",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        r2 = collect_human_written_review(
            workflow=wf,
            submitted_text="College B is good",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert r1.record.review_id != r2.record.review_id
        assert r1.record.review_text == "College A is great"
        assert r2.record.review_text == "College B is good"

    def test_pii_label_remains_none_when_pii_detected(self):
        wf = _make_workflow()
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="My email is test@example.com",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        assert result.record.pii_detected is True
        assert result.record.pii is None


# ---------------------------------------------------------------------------
# J — package public API
# ---------------------------------------------------------------------------


class TestPackagePublicAPI:
    def test_imports_from_rrm_corpus(self):
        from rrm.corpus import (
            HumanWrittenIntakeResult as ExportedResult,
            collect_human_written_review as exported_collect,
        )
        assert ExportedResult is HumanWrittenIntakeResult
        assert exported_collect is collect_human_written_review
