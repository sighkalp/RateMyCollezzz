"""Tests for rrm.corpus.models — source-type enums and source-specific validation rules.

Public API tested:
    SourceType, LanguageMix, CollegeCategory, AnnotationStatus
    validate_synthetic_provenance (source-specific)
    gate_d_eligibility_qc (source-specific)
"""

from __future__ import annotations

from rrm.corpus.annotation import (
    AnnotationSubmission,
    AnnotationSubmissionStore,
    ControlledProtocolTruthStore,
    DispositionRegister,
    GateCOperationalContext,
    ReannotationRegister,
)

import pytest

from rrm.corpus.models import (
    AnnotationStatus,
    CanonicalRecord,
    CollegeCategory,
    ControlledProtocolTruth,
    LanguageMix,
    SourceType,
)
from rrm.corpus.validation import (
    gate_d_eligibility_qc,
    validate_synthetic_provenance,
)


def _make_simple_context(review_id: str = "r-1") -> GateCOperationalContext:
    """Minimal operational context with no submissions."""
    return GateCOperationalContext(
        submission_store=AnnotationSubmissionStore(),
        controlled_truth_store=ControlledProtocolTruthStore(),
        disposition_register=DispositionRegister(),
        reannotation_register=ReannotationRegister(),
        known_review_ids=frozenset({review_id}),
        inherited_target_evidence={},
    )


# ---------------------------------------------------------------------------
# SourceType enum
# ---------------------------------------------------------------------------


class TestSourceType:
    def test_three_members(self):
        assert len(list(SourceType)) == 3

    def test_human_written_value(self):
        assert SourceType.HUMAN_WRITTEN_RMC.value == "HUMAN_WRITTEN_RMC"

    def test_controlled_value(self):
        assert SourceType.CONTROLLED_RMC.value == "CONTROLLED_RMC"

    def test_synthetic_value(self):
        assert SourceType.SYNTHETIC_DERIVED_RMC.value == "SYNTHETIC_DERIVED_RMC"

    def test_membership(self):
        assert SourceType("HUMAN_WRITTEN_RMC") == SourceType.HUMAN_WRITTEN_RMC
        assert SourceType("CONTROLLED_RMC") == SourceType.CONTROLLED_RMC
        assert SourceType("SYNTHETIC_DERIVED_RMC") == SourceType.SYNTHETIC_DERIVED_RMC

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            SourceType("INVALID")


# ---------------------------------------------------------------------------
# LanguageMix enum
# ---------------------------------------------------------------------------


class TestLanguageMix:
    def test_five_members(self):
        assert len(list(LanguageMix)) == 5

    def test_members(self):
        expected = {
            "ENGLISH", "HINGLISH", "ROMAN_HINDI", "OTHER", "MIXED_OTHER"
        }
        assert {m.value for m in LanguageMix} == expected

    def test_membership(self):
        assert LanguageMix("HINGLISH") == LanguageMix.HINGLISH

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            LanguageMix("INVALID")


# ---------------------------------------------------------------------------
# CollegeCategory enum
# ---------------------------------------------------------------------------


class TestCollegeCategory:
    def test_ten_members(self):
        assert len(list(CollegeCategory)) == 10

    def test_members(self):
        expected = {
            "ACADEMICS", "FACULTY", "PLACEMENTS", "HOSTEL",
            "INFRASTRUCTURE", "FEES", "ADMINISTRATION", "CAMPUS_LIFE",
            "ADMISSIONS", "MULTI_TOPIC",
        }
        assert {m.value for m in CollegeCategory} == expected

    def test_membership(self):
        assert (
            CollegeCategory("PLACEMENTS") == CollegeCategory.PLACEMENTS
        )

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            CollegeCategory("INVALID")


# ---------------------------------------------------------------------------
# AnnotationStatus enum
# ---------------------------------------------------------------------------


class TestAnnotationStatus:
    def test_five_members(self):
        assert len(list(AnnotationStatus)) == 5

    def test_members(self):
        expected = {
            "UNANNOTATED", "ANNOTATING", "ADJUDICATION_REQUIRED",
            "FINAL", "EXCLUDED",
        }
        assert {m.value for m in AnnotationStatus} == expected

    def test_default_is_unannotated(self):
        rec = CanonicalRecord(review_id="r-1")
        assert rec.annotation_status == AnnotationStatus.UNANNOTATED


# ---------------------------------------------------------------------------
# Source-specific validation rules
# ---------------------------------------------------------------------------


class TestSourceSpecificValidation:
    """Source-type-specific validation rules."""

    def test_human_written_does_not_require_provenance(self):
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
        )
        is_valid, errors = validate_synthetic_provenance(record)
        assert is_valid is True
        assert errors == []

    def test_controlled_does_not_require_provenance(self):
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.CONTROLLED_RMC,
        )
        is_valid, errors = validate_synthetic_provenance(record)
        assert is_valid is True
        assert errors == []

    def test_synthetic_requires_parent_review_id(self):
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.SYNTHETIC_DERIVED_RMC,
            derivation_type="paraphrase",
            generation_method="llm",
        )
        is_valid, errors = validate_synthetic_provenance(record)
        assert is_valid is False
        assert any("parent_review_id" in e for e in errors)

    def test_synthetic_derivation_type_empty_string_fails(self):
        """Supplied derivation_type must be non-empty descriptive string."""
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.SYNTHETIC_DERIVED_RMC,
            parent_review_id="parent-1",
            derivation_type="",
        )
        is_valid, errors = validate_synthetic_provenance(record)
        assert is_valid is False
        assert any("derivation_type" in e for e in errors)

    def test_synthetic_generation_method_empty_string_fails(self):
        """Supplied generation_method must be non-empty descriptive string."""
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.SYNTHETIC_DERIVED_RMC,
            parent_review_id="parent-1",
            generation_method="",
        )
        is_valid, errors = validate_synthetic_provenance(record)
        assert is_valid is False
        assert any("generation_method" in e for e in errors)

    def test_synthetic_with_full_provenance_passes(self):
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.SYNTHETIC_DERIVED_RMC,
            parent_review_id="parent-1",
            derivation_type="paraphrase",
            generation_method="llm",
        )
        is_valid, errors = validate_synthetic_provenance(record)
        assert is_valid is True
        assert errors == []

    def test_controlled_requires_experiment_id_for_export(self):
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.CONTROLLED_RMC,
            annotation_status=AnnotationStatus.FINAL,
            annotator_A_id="ann-a",
            annotator_B_id="ann-b",
            review_text="text",
            created_at="2025-01-01T00:00:00Z",
        )
        ctx = _make_simple_context("r-1")
        is_eligible, errors = gate_d_eligibility_qc(record, ctx)
        assert is_eligible is False
        assert any("experiment_id" in e for e in errors)

    def test_human_written_requires_consented_for_export(self):
        # Provide valid A/B submissions so the ONLY failing condition
        # is the consent status.
        sub_a = AnnotationSubmission(
            review_id="r-1",
            annotator_id="ann-a",
            annotation_guide_version="1.0",
            spam=0,
            deception=0,
            toxicity=0,
            advertising=0,
            off_topic=0,
            pii=0,
            submitted_at="2025-01-01T00:00:00Z",
        )
        sub_b = AnnotationSubmission(
            review_id="r-1",
            annotator_id="ann-b",
            annotation_guide_version="1.0",
            spam=0,
            deception=0,
            toxicity=0,
            advertising=0,
            off_topic=0,
            pii=0,
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
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            annotation_status=AnnotationStatus.FINAL,
            annotator_A_id="ann-a",
            annotator_B_id="ann-b",
            review_text="text",
            created_at="2025-01-01T00:00:00Z",
            consent_status="WITHDRAWN",
        )
        is_eligible, errors = gate_d_eligibility_qc(record, ctx)
        assert is_eligible is False
        assert any("CONSENTED" in e for e in errors)
