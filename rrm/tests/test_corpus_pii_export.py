"""Tests for rrm.corpus.pii_adapter and rrm.corpus.export.

Public API tested:
    safe_surrogate_transform, redact_pii_in_text, build_pii_evidence_summary
    export_corpus, export_to_jsonl, export_metadata, EXPORT_WHITELIST
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile

import pytest

from rrm.corpus.export import (
    PORTABLE_CANDIDATE_FIELDS,
    EXPORT_WHITELIST,
    export_corpus,
    export_metadata,
    export_to_jsonl,
)
from rrm.corpus.models import (
    ALL_REASON_CODES,
    AnnotationStatus,
    CanonicalRecord,
    CollegeCategory,
    ControlledProtocolTruth,
    Disposition,
    LanguageMix,
    PIIEvidenceSummary,
    SourceType,
)
from rrm.corpus.annotation import (
    AnnotationSubmission,
    AnnotationSubmissionStore,
    ControlledProtocolTruthStore,
    DispositionRegister,
    GateCOperationalContext,
    ReannotationRegister,
)
from rrm.corpus.pii_adapter import (
    build_pii_evidence_summary,
    redact_pii_in_text,
    safe_surrogate_transform,
)


# ============================================================================
# PII Adapter tests
# ============================================================================


class TestSafeSurrogateTransform:
    def test_type_error_non_string(self):
        with pytest.raises(TypeError):
            safe_surrogate_transform(123)  # type: ignore

    def test_no_pii_unchanged(self):
        text = "This is a perfectly clean review with no issues at all."
        result = safe_surrogate_transform(text)
        assert result == text

    def test_email_replaced(self):
        text = "Contact me at user@example.com for details."
        result = safe_surrogate_transform(text)
        assert "user@example.com" not in result
        assert "example.test" in result

    def test_phone_replaced(self):
        text = "Call me at 555-123-4567 tomorrow."
        result = safe_surrogate_transform(text)
        assert "555-123-4567" not in result
        assert "+XX-XXXXX-XXXXX" in result

    def test_url_replaced(self):
        text = "Visit https://www.example.com for more info."
        result = safe_surrogate_transform(text)
        assert "https://www.example.com" not in result
        assert "redacted.example" in result

    def test_empty_string(self):
        result = safe_surrogate_transform("")
        assert result == ""

    def test_whitespace_only(self):
        result = safe_surrogate_transform("   ")
        assert result == "   "

    def test_multiple_pii_patterns(self):
        text = "Email me at john@example.com or visit https://example.org"
        result = safe_surrogate_transform(text)
        assert "john@example.com" not in result
        assert "https://example.org" not in result

    def test_not_preserving_original_values(self):
        """Original private values must NOT appear in the output."""
        original_email = "real.person.12345@gmail.com"
        text = f"Contact {original_email}"
        result = safe_surrogate_transform(text)
        assert original_email not in result


class TestRedactPiiInText:
    def test_type_error_non_string(self):
        with pytest.raises(TypeError):
            redact_pii_in_text(123)  # type: ignore

    def test_no_pii_unchanged(self):
        text = "Clean text with no patterns."
        result = redact_pii_in_text(text)
        assert result == text

    def test_email_redacted(self):
        text = "Contact user@example.com"
        result = redact_pii_in_text(text)
        assert "[EMAIL REDACTED]" in result
        assert "user@example.com" not in result

    def test_phone_redacted(self):
        text = "Call 555-123-4567"
        result = redact_pii_in_text(text)
        assert "[PHONE REDACTED]" in result

    def test_url_redacted(self):
        text = "Visit https://example.com"
        result = redact_pii_in_text(text)
        assert "[URL REDACTED]" in result

    def test_empty_string(self):
        result = redact_pii_in_text("")
        assert result == ""


class TestBuildPiiEvidenceSummary:
    def test_no_pii_returns_none(self):
        result = build_pii_evidence_summary("r-1", "clean text")
        assert result is None

    def test_pii_detected_returns_summary(self):
        text = "Contact user@example.com for info"
        result = build_pii_evidence_summary("r-1", text)
        assert result is not None
        assert isinstance(result, PIIEvidenceSummary)
        assert result.review_id == "r-1"
        assert result.match_count >= 1
        assert "EMAIL" in result.categories
        assert result.redaction_status == "detected_pending_redaction"

    def test_evidence_id_format(self):
        result = build_pii_evidence_summary("r-1", "contact@test.com")
        assert result is not None
        assert result.evidence_id.startswith("pii-evidence-")
        # The numeric suffix must be a zero-padded 6-digit integer
        suffix = result.evidence_id.replace("pii-evidence-", "")
        assert len(suffix) == 6
        assert suffix.isdigit()

    def test_type_error_non_string_review_id(self):
        with pytest.raises(TypeError):
            build_pii_evidence_summary(123, "text")  # type: ignore

    def test_type_error_non_string_text(self):
        with pytest.raises(TypeError):
            build_pii_evidence_summary("r-1", 456)  # type: ignore

    def test_no_original_values_in_summary(self):
        """PIIEvidenceSummary must NOT contain original private values."""
        text = "Real email: john.doe.private@gmail.com"
        result = build_pii_evidence_summary("r-1", text)
        assert result is not None
        # The summary only contains category labels, not the actual email
        assert "john.doe.private@gmail.com" not in result.categories
        assert result.evidence_id is not None


# ============================================================================
# Export tests
# ============================================================================


class TestPortableCandidateFields:
    """Exact 32-field contract for PORTABLE_CANDIDATE_FIELDS."""

    def test_is_tuple(self):
        assert isinstance(PORTABLE_CANDIDATE_FIELDS, tuple)

    def test_exactly_32_fields(self):
        assert len(PORTABLE_CANDIDATE_FIELDS) == 32

    def test_exact_field_order(self):
        expected = (
            "review_id",
            "review_text",
            "export_text_redacted",
            "source_type",
            "spam",
            "deception",
            "toxicity",
            "advertising",
            "off_topic",
            "pii",
            "language_mix",
            "college_category",
            "annotation_status",
            "annotation_guide_version",
            "finalized_at",
            "consent_status",
            "collection_method",
            "experiment_id",
            "controlled_targets",
            "deception_truth",
            "control_protocol_id",
            "parent_review_id",
            "derivation_type",
            "generation_method",
            "pii_detected",
            "pii_categories",
            "pii_redaction_status",
            "pii_evidence_id",
            "created_at",
            "dataset_version",
            "split_membership",
            "split_group_id",
        )
        assert PORTABLE_CANDIDATE_FIELDS == expected

    def test_export_text_redacted_included(self):
        assert "export_text_redacted" in PORTABLE_CANDIDATE_FIELDS

    def test_source_text_raw_excluded(self):
        assert "source_text_raw" not in PORTABLE_CANDIDATE_FIELDS

    def test_unsafe_fields_excluded(self):
        unsafe = {
            "source_text_raw",
            "contributor_pseudonym",
            "annotator_A_id",
            "annotator_B_id",
            "adjudicator_id",
            "annotation_notes",
        }
        for field in unsafe:
            assert field not in PORTABLE_CANDIDATE_FIELDS

    def test_export_whitelist_is_same_object(self):
        assert EXPORT_WHITELIST is PORTABLE_CANDIDATE_FIELDS


class TestExportCorpus:
    def _make_operational_context(self, review_id: str = "r-export") -> GateCOperationalContext:
        """Build a real context for a valid Human-Written record."""
        from rrm.corpus.annotation import (
            AnnotationSubmissionStore,
            ControlledProtocolTruthStore,
            DispositionRegister,
            ReannotationRegister,
            GateCOperationalContext,
            AnnotationSubmission,
        )

        sub_a = AnnotationSubmission(
            review_id=review_id,
            annotator_id="ann-a",
            annotation_guide_version="1.0",
            spam=0,
            deception=-1,
            toxicity=0,
            advertising=0,
            off_topic=0,
            pii=0,
            language_mix=LanguageMix.ENGLISH,
            college_category=None,
            submitted_at="2025-01-01T00:00:00Z",
        )
        sub_b = AnnotationSubmission(
            review_id=review_id,
            annotator_id="ann-b",
            annotation_guide_version="1.0",
            spam=0,
            deception=-1,
            toxicity=0,
            advertising=0,
            off_topic=0,
            pii=0,
            language_mix=LanguageMix.ENGLISH,
            college_category=None,
            submitted_at="2025-01-01T00:00:00Z",
        )
        submissions = AnnotationSubmissionStore()
        submissions.submit(sub_a)
        submissions.submit(sub_b)

        return GateCOperationalContext(
            submission_store=submissions,
            controlled_truth_store=ControlledProtocolTruthStore(),
            disposition_register=DispositionRegister(),
            reannotation_register=ReannotationRegister(),
            known_review_ids=frozenset({review_id}),
            inherited_target_evidence={},
        )

    def _make_valid_record(self, review_id="r-export") -> CanonicalRecord:
        return CanonicalRecord(
            review_id=review_id,
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            annotation_status=AnnotationStatus.FINAL,
            annotation_guide_version="1.0",
            finalized_at="2025-01-02T00:00:00Z",
            annotator_A_id="ann-a",
            annotator_B_id="ann-b",
            review_text="Test review text for export",
            created_at="2025-01-01T00:00:00Z",
            consent_status="CONSENTED",
            spam=0,
            deception=-1,
            toxicity=0,
            advertising=0,
            off_topic=0,
            pii=0,
            language_mix=LanguageMix.ENGLISH,
            college_category=None,
        )

    def test_export_returns_list_of_dicts(self):
        ctx = self._make_operational_context()
        records = [self._make_valid_record()]
        result = export_corpus(records, ctx)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)

    def test_export_all_32_keys_present(self):
        ctx = self._make_operational_context()
        records = [self._make_valid_record()]
        result = export_corpus(records, ctx)
        exported_keys = set(result[0].keys())
        assert exported_keys == set(PORTABLE_CANDIDATE_FIELDS)

    def test_export_none_values_preserved(self):
        """All 32 keys present even when value is None."""
        ctx = self._make_operational_context()
        records = [self._make_valid_record()]
        result = export_corpus(records, ctx)
        exported = result[0]
        # Fields known to be None in this minimal record
        assert "export_text_redacted" in exported
        assert exported["export_text_redacted"] is None
        assert "dataset_version" in exported
        assert exported["dataset_version"] is None
        assert "split_membership" in exported
        assert exported["split_membership"] is None
        assert "split_group_id" in exported
        assert exported["split_group_id"] is None

    def test_non_final_export_rejected(self):
        """Batch export raises if any record is not Gate-D eligible."""
        valid = self._make_valid_record("r-valid")
        unannotated = CanonicalRecord(
            review_id="r-unannotated",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            annotation_status=AnnotationStatus.UNANNOTATED,
            review_text="should be rejected",
            created_at="2025-01-01T00:00:00Z",
            consent_status="CONSENTED",
        )
        ctx = self._make_operational_context()
        ctx = dataclasses.replace(
            ctx,
            known_review_ids=frozenset({"r-valid", "r-unannotated"}),
        )
        with pytest.raises(ValueError, match="Gate-D export rejected"):
            export_corpus([valid, unannotated], ctx)

    def test_no_qc_bypass_parameter_exists(self):
        """export_corpus must not accept qc_gate or equivalent."""
        import inspect

        sig = inspect.signature(export_corpus)
        params = set(sig.parameters.keys())
        assert "qc_gate" not in params
        assert "skip_qc" not in params
        assert "validate" not in params
        assert "force_export" not in params

    def test_enum_serialized_to_string_value(self):
        ctx = self._make_operational_context()
        records = [self._make_valid_record()]
        result = export_corpus(records, ctx)
        assert result[0]["source_type"] == "HUMAN_WRITTEN_RMC"
        assert result[0]["language_mix"] == "ENGLISH"
        assert result[0]["annotation_status"] == "FINAL"

    def test_int_values_preserved(self):
        ctx = self._make_operational_context()
        records = [self._make_valid_record()]
        result = export_corpus(records, ctx)
        assert result[0]["spam"] == 0
        assert result[0]["deception"] == -1


class TestExportToJsonl:
    def test_writes_jsonl_file(self):
        from rrm.corpus.annotation import (
            AnnotationSubmissionStore,
            ControlledProtocolTruthStore,
            DispositionRegister,
            ReannotationRegister,
            GateCOperationalContext,
            AnnotationSubmission,
        )

        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            annotation_status=AnnotationStatus.FINAL,
            annotation_guide_version="1.0",
            finalized_at="2025-01-02T00:00:00Z",
            annotator_A_id="ann-a",
            annotator_B_id="ann-b",
            review_text="text",
            created_at="2025-01-01T00:00:00Z",
            consent_status="CONSENTED",
            spam=0,
            deception=-1,
            toxicity=0,
            advertising=0,
            off_topic=0,
            pii=0,
            language_mix=LanguageMix.ENGLISH,
        )
        sub_a = AnnotationSubmission(
            review_id="r-1",
            annotator_id="ann-a",
            annotation_guide_version="1.0",
            spam=0,
            deception=-1,
            toxicity=0,
            advertising=0,
            off_topic=0,
            pii=0,
            language_mix=LanguageMix.ENGLISH,
            submitted_at="2025-01-01T00:00:00Z",
        )
        sub_b = AnnotationSubmission(
            review_id="r-1",
            annotator_id="ann-b",
            annotation_guide_version="1.0",
            spam=0,
            deception=-1,
            toxicity=0,
            advertising=0,
            off_topic=0,
            pii=0,
            language_mix=LanguageMix.ENGLISH,
            submitted_at="2025-01-01T00:00:00Z",
        )
        submissions = AnnotationSubmissionStore()
        submissions.submit(sub_a)
        submissions.submit(sub_b)
        ctx = GateCOperationalContext(
            submission_store=submissions,
            controlled_truth_store=ControlledProtocolTruthStore(),
            disposition_register=DispositionRegister(),
            reannotation_register=ReannotationRegister(),
            known_review_ids=frozenset({"r-1"}),
            inherited_target_evidence={},
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as fh:
            tmp_path = fh.name

        try:
            count = export_to_jsonl([record], tmp_path, ctx)
            assert count == 1

            with open(tmp_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
            assert len(lines) == 1
            parsed = json.loads(lines[0])
            assert parsed["review_id"] == "r-1"
        finally:
            os.unlink(tmp_path)

    def test_type_error_non_string_path(self):
        from rrm.corpus.annotation import (
            AnnotationSubmissionStore,
            ControlledProtocolTruthStore,
            DispositionRegister,
            ReannotationRegister,
            GateCOperationalContext,
        )

        ctx = GateCOperationalContext(
            submission_store=AnnotationSubmissionStore(),
            controlled_truth_store=ControlledProtocolTruthStore(),
            disposition_register=DispositionRegister(),
            reannotation_register=ReannotationRegister(),
            known_review_ids=frozenset(),
            inherited_target_evidence={},
        )
        with pytest.raises(TypeError):
            export_to_jsonl([], 123, ctx)  # type: ignore


class TestImportFromJsonlAbsent:
    """import_from_jsonl is completely removed — not a stub, not a deprecation."""

    def test_not_in_corpus_export(self):
        """The symbol must not exist in rrm.corpus.export."""
        assert not hasattr(
            __import__("rrm.corpus.export", fromlist=["import_from_jsonl"]),
            "import_from_jsonl",
        )

    def test_not_in_corpus_package(self):
        """The symbol must not be re-exported from rrm.corpus."""
        assert not hasattr(
            __import__("rrm.corpus", fromlist=["import_from_jsonl"]),
            "import_from_jsonl",
        )


class TestExportMetadata:
    def test_excludes_review_text(self):
        record = CanonicalRecord(
            review_id="r-1",
            review_text="This is the review content that should not appear in metadata export.",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
        )
        meta = export_metadata(record)
        assert "review_text" not in meta
        assert "source_text_raw" not in meta

    def test_includes_identifying_fields(self):
        record = CanonicalRecord(
            review_id="r-meta",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            annotation_status=AnnotationStatus.FINAL,
            finalized_at="2025-01-01T00:00:00Z",
        )
        meta = export_metadata(record)
        assert meta["review_id"] == "r-meta"
        assert meta["source_type"] == "HUMAN_WRITTEN_RMC"
        assert meta["annotation_status"] == "FINAL"
        assert meta["finalized_at"] == "2025-01-01T00:00:00Z"
        # Annotator identity fields are excluded from portable export
        assert "annotator_A_id" not in meta
        assert "annotator_B_id" not in meta
        assert "adjudicator_id" not in meta
