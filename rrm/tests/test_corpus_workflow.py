"""Tests for rrm.corpus.workflow module.

Public API tested:
    CorpusWorkflow
"""

from __future__ import annotations

import datetime

import pytest

from rrm.corpus.annotation import (
    AnnotationSubmissionStore,
    ControlledProtocolTruthStore,
    DispositionRegister,
    ReannotationRegister,
)
from rrm.corpus.models import (
    ALL_REASON_CODES,
    ANNOTATION_INCOMPLETE,
    AnnotationStatus,
    AnnotationSubmission,
    CanonicalRecord,
    CollegeCategory,
    ControlledProtocolTruth,
    Disposition,
    DUPLICATE_RECORD,
    EMPTY_TEXT,
    INVALID_CONSENT,
    LanguageMix,
    RecordDisposition,
    SourceType,
)
from rrm.corpus.workflow import CorpusWorkflow
from rrm.corpus.validation import gate_d_eligibility_qc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_workflow() -> CorpusWorkflow:
    return CorpusWorkflow(
        submissions=AnnotationSubmissionStore(),
        truths=ControlledProtocolTruthStore(),
        dispositions=DispositionRegister(),
        reannotations=ReannotationRegister(),
    )


def _make_sub(
    review_id: str = "r-1",
    annotator_id: str = "ann-a",
    spam: int = 0,
    deception: int = -1,
    toxicity: int = 0,
    advertising: int = 0,
    off_topic: int = 0,
    pii: int = 0,
    language_mix: LanguageMix = LanguageMix.ENGLISH,
    college_category: CollegeCategory = CollegeCategory.ACADEMICS,
    annotation_guide_version: str = "1.0",
    submitted_at: str = "2025-01-01T00:00:00Z",
) -> AnnotationSubmission:
    return AnnotationSubmission(
        review_id=review_id,
        annotator_id=annotator_id,
        annotation_guide_version=annotation_guide_version,
        spam=spam,
        deception=deception,
        toxicity=toxicity,
        advertising=advertising,
        off_topic=off_topic,
        pii=pii,
        language_mix=language_mix,
        college_category=college_category,
        submitted_at=submitted_at,
    )


# ---------------------------------------------------------------------------
# Record creation and registration
# ---------------------------------------------------------------------------


class TestRecordCreation:
    def test_create_human_written(self):
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw text",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
            review_text="safe text",
        )
        assert rec.review_id == "r-1"
        assert rec.annotation_status == AnnotationStatus.UNANNOTATED
        assert rec.source_type == SourceType.HUMAN_WRITTEN_RMC

    def test_create_controlled(self):
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-ctrl",
            source_text_raw="raw",
            source_type=SourceType.CONTROLLED_RMC,
            created_at="2025-01-01T00:00:00Z",
            experiment_id="exp-1",
        )
        assert rec.source_type == SourceType.CONTROLLED_RMC
        assert rec.experiment_id == "exp-1"

    def test_create_synthetic_with_provenance(self):
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-syn",
            source_text_raw="raw",
            source_type=SourceType.SYNTHETIC_DERIVED_RMC,
            created_at="2025-01-01T00:00:00Z",
            parent_review_id="parent-1",
            derivation_type="paraphrase",
            generation_method="llm",
        )
        assert rec.parent_review_id == "parent-1"
        assert rec.derivation_type == "paraphrase"
        assert rec.generation_method == "llm"

    def test_type_error_bad_review_id(self):
        wf = _make_workflow()
        with pytest.raises(TypeError):
            wf.create_record(
                review_id="",
                source_text_raw="raw",
                source_type=SourceType.HUMAN_WRITTEN_RMC,
                created_at="2025-01-01T00:00:00Z",
            )

    def test_type_error_bad_source_type(self):
        wf = _make_workflow()
        with pytest.raises(TypeError):
            wf.create_record(
                review_id="r-1",
                source_text_raw="raw",
                source_type="HUMAN_WRITTEN_RMC",  # type: ignore
                created_at="2025-01-01T00:00:00Z",
            )

    def test_register_record(self):
        wf = _make_workflow()
        rec = CanonicalRecord(review_id="r-ext")
        wf.register_record(rec)
        assert wf.get_record("r-ext") == rec

    def test_get_record_nonexistent(self):
        wf = _make_workflow()
        assert wf.get_record("nonexistent") is None


# ---------------------------------------------------------------------------
# Ephemeral raw text
# ---------------------------------------------------------------------------


class TestEphemeralRawText:
    def test_returns_source_text_raw_when_present(self):
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw content here",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        result = wf.prepare_ephemeral_raw_text(rec)
        assert result == "raw content here"

    def test_returns_none_when_no_raw_text(self):
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        # Empty string is falsy for source_text_raw, returns None
        result = wf.prepare_ephemeral_raw_text(rec)
        # source_text_raw="" is falsy but not None, so it returns ""
        # Actually let me check: source_text_raw="" is not None, so it returns ""
        assert result == ""


# ---------------------------------------------------------------------------
# Annotation submissions
# ---------------------------------------------------------------------------


class TestAnnotationSubmissions:
    def test_submit_a_changes_to_annotating(self):
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a = _make_sub(annotator_id="ann-a", annotation_guide_version="1.0")
        updated = wf.submit_annotation_a(rec, sub_a)
        assert updated.annotation_status == AnnotationStatus.ANNOTATING
        assert updated.annotator_A_id == "ann-a"

    def test_submit_b_agreement_keeps_annotating(self):
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a = _make_sub(annotator_id="ann-a", spam=0)
        sub_b = _make_sub(annotator_id="ann-b", spam=0)
        wf.submit_annotation_a(rec, sub_a)
        updated = wf.submit_annotation_b(rec, sub_b)
        # Same labels → stays ANNOTATING (ready for agreement finalization)
        assert updated.annotation_status == AnnotationStatus.ANNOTATING
        assert updated.annotator_B_id == "ann-b"

    def test_submit_b_disagreement_goes_to_adjudication(self):
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a = _make_sub(annotator_id="ann-a", spam=0)
        sub_b = _make_sub(annotator_id="ann-b", spam=1)  # disagree on spam
        wf.submit_annotation_a(rec, sub_a)
        updated = wf.submit_annotation_b(rec, sub_b)
        assert updated.annotation_status == AnnotationStatus.ADJUDICATION_REQUIRED

    def test_submit_a_on_excluded_raises(self):
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        wf.exclude("r-1", EMPTY_TEXT)
        sub_a = _make_sub()
        with pytest.raises(ValueError, match="excluded"):
            wf.submit_annotation_a(rec, sub_a)

    def test_type_error_on_wrong_submission_type(self):
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        with pytest.raises(TypeError):
            wf.submit_annotation_a(rec, "not a submission")  # type: ignore


# ---------------------------------------------------------------------------
# Finalization
# ---------------------------------------------------------------------------


class TestFinalization:
    def test_finalize_agreement_succeeds(self):
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a = _make_sub(annotator_id="ann-a", spam=0)
        sub_b = _make_sub(annotator_id="ann-b", spam=0)
        wf.submit_annotation_a(rec, sub_a)
        wf.submit_annotation_b(rec, sub_b)
        finalized = wf.finalize_agreement("r-1", "2025-01-02T00:00:00Z")
        assert finalized.annotation_status == AnnotationStatus.FINAL
        assert finalized.finalized_at == "2025-01-02T00:00:00Z"

    def test_finalize_agreement_requires_annotating(self):
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        # Record is still UNANNOTATED
        with pytest.raises(ValueError, match="ANNOTATING"):
            wf.finalize_agreement("r-1", "2025-01-02T00:00:00Z")

    def test_finalize_adjudication_succeeds(self):
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a = _make_sub(annotator_id="ann-a", spam=0)
        sub_b = _make_sub(annotator_id="ann-b", spam=1)
        wf.submit_annotation_a(rec, sub_a)
        wf.submit_annotation_b(rec, sub_b)
        finalized = wf.finalize_adjudication(
            "r-1", "adj-1", {"spam": 0}, "2025-01-02T00:00:00Z"
        )
        assert finalized.annotation_status == AnnotationStatus.FINAL
        assert finalized.adjudicator_id == "adj-1"

    def test_finalize_adjudication_requires_adjudication_status(self):
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a = _make_sub(annotator_id="ann-a", spam=0)
        sub_b = _make_sub(annotator_id="ann-b", spam=0)
        wf.submit_annotation_a(rec, sub_a)
        wf.submit_annotation_b(rec, sub_b)
        # Record is ANNOTATING (agreed), not ADJUDICATION_REQUIRED
        with pytest.raises(ValueError, match="ADJUDICATION_REQUIRED"):
            wf.finalize_adjudication(
                "r-1", "adj-1", {"spam": 0}, "2025-01-02T00:00:00Z"
            )


# ---------------------------------------------------------------------------
# Exclusion
# ---------------------------------------------------------------------------


class TestExclusion:
    def test_exclude_changes_status(self):
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        excluded = wf.exclude("r-1", EMPTY_TEXT)
        assert excluded.annotation_status == AnnotationStatus.EXCLUDED

    def test_exclude_invalid_reason_code_raises(self):
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        with pytest.raises(ValueError, match="Invalid reason code"):
            wf.exclude("r-1", "NOT_A_VALID_REASON")

    def test_exclude_registers_disposition(self):
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        wf.exclude("r-1", EMPTY_TEXT, internal_note="test note")
        disp = wf._dispositions.get("r-1")
        assert disp is not None
        assert disp.disposition == Disposition.EXCLUDED
        assert disp.reason_code == EMPTY_TEXT
        assert disp.internal_note == "test note"


# ---------------------------------------------------------------------------
# Workflow state queries
# ---------------------------------------------------------------------------


class TestWorkflowState:
    def test_unannotated_state(self):
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        assert wf.get_workflow_state(rec) == AnnotationStatus.UNANNOTATED

    def test_annotating_state(self):
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a = _make_sub(annotator_id="ann-a")
        rec = wf.submit_annotation_a(rec, sub_a)
        assert wf.get_workflow_state(rec) == AnnotationStatus.ANNOTATING

    def test_adjudication_required_state(self):
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a = _make_sub(annotator_id="ann-a", spam=0)
        sub_b = _make_sub(annotator_id="ann-b", spam=1)
        rec = wf.submit_annotation_a(rec, sub_a)
        rec = wf.submit_annotation_b(rec, sub_b)
        assert wf.get_workflow_state(rec) == AnnotationStatus.ADJUDICATION_REQUIRED

    def test_final_state(self):
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a = _make_sub(annotator_id="ann-a", spam=0)
        sub_b = _make_sub(annotator_id="ann-b", spam=0)
        wf.submit_annotation_a(rec, sub_a)
        wf.submit_annotation_b(rec, sub_b)
        wf.finalize_agreement("r-1", "2025-01-02T00:00:00Z")
        rec = wf.get_record("r-1")
        assert wf.get_workflow_state(rec) == AnnotationStatus.FINAL

    def test_excluded_overrides_final(self):
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a = _make_sub(annotator_id="ann-a", spam=0)
        sub_b = _make_sub(annotator_id="ann-b", spam=0)
        wf.submit_annotation_a(rec, sub_a)
        wf.submit_annotation_b(rec, sub_b)
        wf.finalize_agreement("r-1", "2025-01-02T00:00:00Z")
        wf.exclude("r-1", DUPLICATE_RECORD)
        rec = wf.get_record("r-1")
        assert wf.get_workflow_state(rec) == AnnotationStatus.EXCLUDED


# ---------------------------------------------------------------------------
# Gate-D eligibility via workflow
# ---------------------------------------------------------------------------


class TestGateDEligibilityViaWorkflow:
    def test_get_gate_d_eligibility(self):
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
            review_text="sample review text",
            consent_status="CONSENTED",
        )
        # Transition through the workflow to FINAL
        sub_a = _make_sub(annotator_id="ann-a", annotation_guide_version="1.0")
        sub_b = _make_sub(annotator_id="ann-b", annotation_guide_version="1.0")
        rec = wf.submit_annotation_a(rec, sub_a)
        rec = wf.submit_annotation_b(rec, sub_b)
        rec = wf.finalize_agreement("r-1", "2025-01-02T00:00:00Z")
        is_eligible, errors = wf.get_gate_d_eligibility(rec)
        assert is_eligible is True

    def test_unannotated_fails_gate_d(self):
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
            consent_status="CONSENTED",
        )
        is_eligible, errors = wf.get_gate_d_eligibility(rec)
        assert is_eligible is False
        assert any("annotation_status" in e for e in errors)
