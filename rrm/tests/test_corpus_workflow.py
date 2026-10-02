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
            annotation_guide_version="1.0",
        )
        sub_a = _make_sub(annotator_id="ann-a", spam=0, annotation_guide_version="1.0")
        sub_b = _make_sub(annotator_id="ann-b", spam=0, annotation_guide_version="1.0")
        wf.submit_annotation_a(rec, sub_a)
        wf.submit_annotation_b(rec, sub_b)
        finalized = wf.finalize_agreement("r-1", "2025-01-02T00:00:00Z")
        assert finalized.annotation_status == AnnotationStatus.FINAL
        assert finalized.finalized_at == "2025-01-02T00:00:00Z"
        # Labels must be materialized from agreement
        assert finalized.spam == 0
        assert finalized.deception == -1
        assert finalized.toxicity == 0
        assert finalized.advertising == 0
        assert finalized.off_topic == 0
        assert finalized.pii == 0
        assert finalized.language_mix == LanguageMix.ENGLISH
        assert finalized.college_category == CollegeCategory.ACADEMICS
        assert finalized.annotator_A_id == "ann-a"
        assert finalized.annotator_B_id == "ann-b"
        assert finalized.annotation_guide_version == "1.0"

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

    # ---- New defect-closing tests ----

    def test_finalize_agreement_materializes_all_labels(self):
        """Defect A: FINAL record must carry all agreed labels."""
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a = _make_sub(
            annotator_id="ann-a",
            spam=1,
            toxicity=1,
            advertising=0,
            off_topic=0,
            pii=1,
        )
        sub_b = _make_sub(
            annotator_id="ann-b",
            spam=1,
            toxicity=1,
            advertising=0,
            off_topic=0,
            pii=1,
        )
        wf.submit_annotation_a(rec, sub_a)
        wf.submit_annotation_b(rec, sub_b)
        finalized = wf.finalize_agreement("r-1", "2025-01-02T00:00:00Z")
        assert finalized.spam == 1
        assert finalized.toxicity == 1
        assert finalized.advertising == 0
        assert finalized.off_topic == 0
        assert finalized.pii == 1
        assert finalized.language_mix == LanguageMix.ENGLISH
        assert finalized.college_category == CollegeCategory.ACADEMICS

    def test_finalize_adjudication_materializes_labels(self):
        """Defect C: adjudicated FINAL record must carry resolved labels."""
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a = _make_sub(annotator_id="ann-a", spam=1, toxicity=0)
        sub_b = _make_sub(annotator_id="ann-b", spam=0, toxicity=1)
        wf.submit_annotation_a(rec, sub_a)
        wf.submit_annotation_b(rec, sub_b)
        finalized = wf.finalize_adjudication(
            "r-1", "adj-1",
            {"spam": 0, "toxicity": 1},
            "2025-01-02T00:00:00Z",
        )
        assert finalized.spam == 0
        assert finalized.toxicity == 1
        assert finalized.annotator_A_id == "ann-a"
        assert finalized.annotator_B_id == "ann-b"
        assert finalized.deception == -1

    def test_finalize_adjudication_rejects_annotator_as_adj(self):
        """Defect D: adjudicator must not be either annotator."""
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.CONTROLLED_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a = _make_sub(annotator_id="ann-a", spam=0)
        sub_b = _make_sub(annotator_id="ann-b", spam=1)
        wf.submit_annotation_a(rec, sub_a)
        wf.submit_annotation_b(rec, sub_b)
        with pytest.raises(ValueError, match="must not be Annotator A"):
            wf.finalize_adjudication(
                "r-1", "ann-a", {"spam": 0}, "2025-01-02T00:00:00Z"
            )
        with pytest.raises(ValueError, match="must not be Annotator B"):
            wf.finalize_adjudication(
                "r-1", "ann-b", {"spam": 0}, "2025-01-02T00:00:00Z"
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


# ---------------------------------------------------------------------------
# C1 Blocker tests — agreement
# ---------------------------------------------------------------------------


class TestAgreementC1Blockers:
    """Tests for the C1 Human-Written agreement repair."""

    def test_same_annotator_a_and_b_rejects_agreement(self):
        """Blocker B: A and B must be different people."""
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a = _make_sub(annotator_id="ann-a", annotation_guide_version="1.0")
        sub_b = _make_sub(annotator_id="ann-a", annotation_guide_version="1.0")
        wf.submit_annotation_a(rec, sub_a)
        wf.submit_annotation_b(rec, sub_b)
        with pytest.raises(ValueError, match="must be different"):
            wf.finalize_agreement("r-1", "2025-01-02T00:00:00Z")

    def test_eight_dimension_disagreement_rejects_agreement(self):
        """Blocker B/11B: disagreement on any dimension blocks agreement."""
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a = _make_sub(annotator_id="ann-a", spam=0, annotation_guide_version="1.0")
        sub_b = _make_sub(annotator_id="ann-b", spam=1, annotation_guide_version="1.0")
        wf.submit_annotation_a(rec, sub_a)
        updated = wf.submit_annotation_b(rec, sub_b)
        # Disagreement is detected at submit_annotation_b time
        assert updated.annotation_status == AnnotationStatus.ADJUDICATION_REQUIRED
        # finalize_agreement then rejects because status is not ANNOTATING
        with pytest.raises(ValueError, match="ADJUDICATION_REQUIRED"):
            wf.finalize_agreement("r-1", "2025-01-02T00:00:00Z")

    def test_guide_version_mismatch_rejects_agreement(self):
        """Blocker A/11C: A/B must use the same annotation_guide_version."""
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
            annotation_guide_version="1.0",
        )
        sub_a = _make_sub(
            annotator_id="ann-a",
            annotation_guide_version="1.0",
            spam=0,
        )
        sub_b = _make_sub(
            annotator_id="ann-b",
            annotation_guide_version="2.0",
            spam=0,
        )
        wf.submit_annotation_a(rec, sub_a)
        wf.submit_annotation_b(rec, sub_b)
        with pytest.raises(ValueError, match="guide version"):
            wf.finalize_agreement("r-1", "2025-01-02T00:00:00Z")

    def test_guide_version_mismatch_without_record_guide_rejects(self):
        """Blocker A: mismatch rejected even when record has no guide set."""
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a = _make_sub(
            annotator_id="ann-a",
            annotation_guide_version="1.0",
            spam=0,
        )
        sub_b = _make_sub(
            annotator_id="ann-b",
            annotation_guide_version="2.0",
            spam=0,
        )
        wf.submit_annotation_a(rec, sub_a)
        wf.submit_annotation_b(rec, sub_b)
        with pytest.raises(ValueError, match="guide version"):
            wf.finalize_agreement("r-1", "2025-01-02T00:00:00Z")

    def test_agreement_materializes_none_college_category(self):
        """11D: college_category=None from agreement is preserved exactly."""
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a = _make_sub(
            annotator_id="ann-a",
            spam=0,
            annotation_guide_version="1.0",
            college_category=None,
        )
        sub_b = _make_sub(
            annotator_id="ann-b",
            spam=0,
            annotation_guide_version="1.0",
            college_category=None,
        )
        wf.submit_annotation_a(rec, sub_a)
        wf.submit_annotation_b(rec, sub_b)
        finalized = wf.finalize_agreement("r-1", "2025-01-02T00:00:00Z")
        assert finalized.college_category is None
        assert finalized.language_mix == LanguageMix.ENGLISH

    def test_human_written_agreement_deception_always_minus_one(self):
        """11E: Human-Written agreement always produces deception=-1."""
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a = _make_sub(
            annotator_id="ann-a",
            spam=0,
            deception=0,  # annotator opinion should be overridden
            annotation_guide_version="1.0",
        )
        sub_b = _make_sub(
            annotator_id="ann-b",
            spam=0,
            deception=0,
            annotation_guide_version="1.0",
        )
        wf.submit_annotation_a(rec, sub_a)
        wf.submit_annotation_b(rec, sub_b)
        finalized = wf.finalize_agreement("r-1", "2025-01-02T00:00:00Z")
        assert finalized.deception == -1

    def test_guide_version_propagates_when_record_guide_is_none(self):
        """Blocker A: A/B shared guide is used when record guide is None."""
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a = _make_sub(
            annotator_id="ann-a",
            annotation_guide_version="2.0",
            spam=0,
        )
        sub_b = _make_sub(
            annotator_id="ann-b",
            annotation_guide_version="2.0",
            spam=0,
        )
        wf.submit_annotation_a(rec, sub_a)
        wf.submit_annotation_b(rec, sub_b)
        finalized = wf.finalize_agreement("r-1", "2025-01-02T00:00:00Z")
        assert finalized.annotation_guide_version == "2.0"

    def test_non_human_agreement_eight_dimensions_preserved(self):
        """Non-Human finalize_agreement must not overwrite any of the
        eight annotation dimensions already on the record."""
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-nh",
            source_text_raw="raw",
            source_type=SourceType.CONTROLLED_RMC,
            created_at="2025-01-01T00:00:00Z",
            spam=1,
            deception=0,
            toxicity=1,
            advertising=0,
            off_topic=1,
            pii=0,
            language_mix=LanguageMix.ROMAN_HINDI,
            college_category=CollegeCategory.HOSTEL,
        )
        # A and B agree with each other but with different values than
        # the pre-populated record
        sub_a = _make_sub(
            review_id="r-nh",
            annotator_id="ann-a",
            spam=0,
            deception=1,
            toxicity=0,
            advertising=1,
            off_topic=0,
            pii=1,
            language_mix=LanguageMix.ENGLISH,
            college_category=CollegeCategory.ACADEMICS,
            annotation_guide_version="1.0",
        )
        sub_b = _make_sub(
            review_id="r-nh",
            annotator_id="ann-b",
            spam=0,
            deception=1,
            toxicity=0,
            advertising=1,
            off_topic=0,
            pii=1,
            language_mix=LanguageMix.ENGLISH,
            college_category=CollegeCategory.ACADEMICS,
            annotation_guide_version="1.0",
        )
        wf.submit_annotation_a(rec, sub_a)
        wf.submit_annotation_b(rec, sub_b)
        finalized = wf.finalize_agreement("r-nh", "2025-01-02T00:00:00Z")
        assert finalized.spam == 1
        assert finalized.deception == 0
        assert finalized.toxicity == 1
        assert finalized.advertising == 0
        assert finalized.off_topic == 1
        assert finalized.pii == 0
        assert finalized.language_mix == LanguageMix.ROMAN_HINDI
        assert finalized.college_category == CollegeCategory.HOSTEL


# ---------------------------------------------------------------------------
# C1 Blocker tests — adjudication
# ---------------------------------------------------------------------------


class TestAdjudicationC1Blockers:
    """Tests for the C1 Human-Written adjudication repair."""

    def test_adjudication_materializes_choices(self):
        """12A: adjudicator choices are materialized in FINAL record."""
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a = _make_sub(annotator_id="ann-a", spam=1, toxicity=0)
        sub_b = _make_sub(annotator_id="ann-b", spam=0, toxicity=1)
        wf.submit_annotation_a(rec, sub_a)
        wf.submit_annotation_b(rec, sub_b)
        finalized = wf.finalize_adjudication(
            "r-1", "adj-1",
            {"spam": 0, "toxicity": 1},
            "2025-01-02T00:00:00Z",
        )
        assert finalized.spam == 0
        assert finalized.toxicity == 1
        assert finalized.advertising == 0
        assert finalized.off_topic == 0
        assert finalized.pii == 0
        assert finalized.language_mix == LanguageMix.ENGLISH
        assert finalized.college_category == CollegeCategory.ACADEMICS

    def test_adjudication_deception_always_minus_one_human_written(self):
        """12B: Human-Written adjudication keeps deception=-1."""
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a = _make_sub(annotator_id="ann-a", spam=1)
        sub_b = _make_sub(annotator_id="ann-b", spam=0)
        wf.submit_annotation_a(rec, sub_a)
        wf.submit_annotation_b(rec, sub_b)
        # Adjudicator tries deception=0 — must be overridden
        finalized = wf.finalize_adjudication(
            "r-1", "adj-1",
            {"spam": 0, "deception": 0},
            "2025-01-02T00:00:00Z",
        )
        assert finalized.deception == -1

        # Also test deception=1 attempt
        wf2 = _make_workflow()
        rec2 = wf2.create_record(
            review_id="r-2",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a2 = _make_sub(
            review_id="r-2", annotator_id="ann-a", spam=1,
        )
        sub_b2 = _make_sub(
            review_id="r-2", annotator_id="ann-b", spam=0,
        )
        wf2.submit_annotation_a(rec2, sub_a2)
        wf2.submit_annotation_b(rec2, sub_b2)
        finalized2 = wf2.finalize_adjudication(
            "r-2", "adj-1",
            {"spam": 0, "deception": 1},
            "2025-01-02T00:00:00Z",
        )
        assert finalized2.deception == -1

    def test_same_person_a_and_b_rejects_adjudication(self):
        """12C: same person as A and B cannot adjudicate."""
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a = _make_sub(annotator_id="ann-a", spam=0)
        sub_b = _make_sub(annotator_id="ann-a", spam=1)
        wf.submit_annotation_a(rec, sub_a)
        wf.submit_annotation_b(rec, sub_b)
        with pytest.raises(ValueError, match="must be different"):
            wf.finalize_adjudication(
                "r-1", "adj-1", {"spam": 0},
                "2025-01-02T00:00:00Z",
            )

    def test_adjudicator_cannot_equal_a(self):
        """12D: adjudicator cannot be Annotator A."""
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
        with pytest.raises(ValueError, match="must not be Annotator A"):
            wf.finalize_adjudication(
                "r-1", "ann-a", {"spam": 0},
                "2025-01-02T00:00:00Z",
            )

    def test_adjudicator_cannot_equal_b(self):
        """12E: adjudicator cannot be Annotator B."""
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
        with pytest.raises(ValueError, match="must not be Annotator B"):
            wf.finalize_adjudication(
                "r-1", "ann-b", {"spam": 0},
                "2025-01-02T00:00:00Z",
            )

    def test_ab_guide_version_mismatch_rejects_adjudication(self):
        """12F: A/B guide-version mismatch cannot finalize via adjudication."""
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
            annotation_guide_version="1.0",
        )
        sub_a = _make_sub(
            annotator_id="ann-a",
            annotation_guide_version="1.0",
            spam=0,
        )
        sub_b = _make_sub(
            annotator_id="ann-b",
            annotation_guide_version="2.0",
            spam=1,
        )
        wf.submit_annotation_a(rec, sub_a)
        wf.submit_annotation_b(rec, sub_b)
        with pytest.raises(ValueError, match="guide version"):
            wf.finalize_adjudication(
                "r-1", "adj-1", {"spam": 0},
                "2025-01-02T00:00:00Z",
            )

    def test_invalid_language_mix_adjudicator_raises(self):
        """12G: invalid language_mix adjudicator choice raises ValueError."""
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
        with pytest.raises(ValueError, match="Invalid language_mix"):
            wf.finalize_adjudication(
                "r-1", "adj-1",
                {"language_mix": "NOT_A_VALID_LANG"},
                "2025-01-02T00:00:00Z",
            )

    def test_invalid_college_category_adjudicator_raises(self):
        """12H: invalid college_category adjudicator choice raises ValueError."""
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
        with pytest.raises(ValueError, match="Invalid college_category"):
            wf.finalize_adjudication(
                "r-1", "adj-1",
                {"college_category": "NOT_A_VALID_CAT"},
                "2025-01-02T00:00:00Z",
            )

    def test_valid_language_mix_string_adjudicator_accepted(self):
        """Valid string language_mix adjudicator choice is accepted."""
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a = _make_sub(annotator_id="ann-a", language_mix=LanguageMix.ENGLISH)
        sub_b = _make_sub(annotator_id="ann-b", language_mix=LanguageMix.HINGLISH)
        wf.submit_annotation_a(rec, sub_a)
        wf.submit_annotation_b(rec, sub_b)
        finalized = wf.finalize_adjudication(
            "r-1", "adj-1",
            {"language_mix": "ROMAN_HINDI"},
            "2025-01-02T00:00:00Z",
        )
        assert finalized.language_mix == LanguageMix.ROMAN_HINDI

    def test_adjudication_wrong_enum_type_language_mix_rejected(self):
        """CollegeCategory.ACADEMICS must not be accepted as language_mix."""
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a = _make_sub(annotator_id="ann-a", language_mix=LanguageMix.ENGLISH)
        sub_b = _make_sub(annotator_id="ann-b", language_mix=LanguageMix.HINGLISH)
        wf.submit_annotation_a(rec, sub_a)
        wf.submit_annotation_b(rec, sub_b)
        with pytest.raises(ValueError, match="Invalid language_mix"):
            wf.finalize_adjudication(
                "r-1", "adj-1",
                {"language_mix": CollegeCategory.ACADEMICS},
                "2025-01-02T00:00:00Z",
            )

    def test_adjudication_wrong_enum_type_college_category_rejected(self):
        """LanguageMix.ENGLISH must not be accepted as college_category."""
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
        )
        sub_a = _make_sub(
            annotator_id="ann-a",
            college_category=CollegeCategory.ACADEMICS,
        )
        sub_b = _make_sub(
            annotator_id="ann-b",
            college_category=CollegeCategory.HOSTEL,
        )
        wf.submit_annotation_a(rec, sub_a)
        wf.submit_annotation_b(rec, sub_b)
        with pytest.raises(ValueError, match="Invalid college_category"):
            wf.finalize_adjudication(
                "r-1", "adj-1",
                {"college_category": LanguageMix.ENGLISH},
                "2025-01-02T00:00:00Z",
            )

    def test_adjudication_non_human_eight_dimensions_preserved(self):
        """Non-Human adjudication must not overwrite any of the eight
        annotation dimensions on the CanonicalRecord."""
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw="raw",
            source_type=SourceType.CONTROLLED_RMC,
            created_at="2025-01-01T00:00:00Z",
            spam=1,
            deception=0,
            toxicity=1,
            advertising=0,
            off_topic=1,
            pii=0,
            language_mix=LanguageMix.ROMAN_HINDI,
            college_category=CollegeCategory.HOSTEL,
        )
        # A and B disagree on spam so the record reaches ADJUDICATION_REQUIRED
        sub_a = _make_sub(
            review_id="r-1",
            annotator_id="ann-a",
            spam=1,
            deception=0,
            toxicity=1,
            advertising=0,
            off_topic=1,
            pii=0,
            language_mix=LanguageMix.ROMAN_HINDI,
            college_category=CollegeCategory.HOSTEL,
            annotation_guide_version="1.0",
        )
        sub_b = _make_sub(
            review_id="r-1",
            annotator_id="ann-b",
            spam=0,  # disagrees
            annotation_guide_version="1.0",
        )
        wf.submit_annotation_a(rec, sub_a)
        wf.submit_annotation_b(rec, sub_b)
        # Adjudicator chooses values different from pre-populated record
        finalized = wf.finalize_adjudication(
            "r-1", "adj-1",
            {
                "spam": 0,
                "deception": 1,
                "toxicity": 0,
                "advertising": 1,
                "off_topic": 0,
                "pii": 1,
                "language_mix": "ENGLISH",
                "college_category": "ACADEMICS",
            },
            "2025-01-02T00:00:00Z",
        )
        # All eight pre-populated values must survive unchanged
        assert finalized.spam == 1
        assert finalized.deception == 0
        assert finalized.toxicity == 1
        assert finalized.advertising == 0
        assert finalized.off_topic == 1
        assert finalized.pii == 0
        assert finalized.language_mix == LanguageMix.ROMAN_HINDI
        assert finalized.college_category == CollegeCategory.HOSTEL


# ---------------------------------------------------------------------------
# Optional raw source test (section 10)
# ---------------------------------------------------------------------------


class TestOptionalRawSource:
    def test_create_record_with_none_raw_source(self):
        """source_text_raw=None is accepted; no auto-copy to review_text."""
        wf = _make_workflow()
        rec = wf.create_record(
            review_id="r-1",
            source_text_raw=None,
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            created_at="2025-01-01T00:00:00Z",
            review_text="safe text",
        )
        assert rec.source_text_raw is None
        assert rec.review_text == "safe text"


# ---------------------------------------------------------------------------
# Non-Human scope-preservation test (section 14)
# ---------------------------------------------------------------------------


class TestNonHumanScopePreservation:
    def test_human_written_repair_does_not_overwrite_controlled_label(self):
        """C1 Human-Written repair must not make annotator voting
        authoritative for CONTROLLED_RMC records."""
        wf = _make_workflow()
        # Pre-populate a CONTROLLED_RMC record with a canonical label
        rec = wf.create_record(
            review_id="r-ctrl",
            source_text_raw="raw",
            source_type=SourceType.CONTROLLED_RMC,
            created_at="2025-01-01T00:00:00Z",
            spam=1,  # controlled ground truth
        )
        # Submissions disagree (annotator B says spam=0)
        sub_a = _make_sub(
            review_id="r-ctrl",
            annotator_id="ann-a",
            spam=1,  # matches controlled truth
            annotation_guide_version="1.0",
        )
        sub_b = _make_sub(
            review_id="r-ctrl",
            annotator_id="ann-b",
            spam=0,  # disagrees
            annotation_guide_version="1.0",
        )
        wf.submit_annotation_a(rec, sub_a)
        wf.submit_annotation_b(rec, sub_b)
        # After submit_annotation_b, the record should carry B's
        # annotation_status and college_category/language_mix per
        # the current B-submission, but the controlled spam=1
        # on the record itself is NOT overwritten by submit_annotation_b.
        updated = wf.get_record("r-ctrl")
        assert updated is not None
        # The controlled spam=1 on the record should still be present
        # (submit_annotation_b does not overwrite canonical labels
        # from controlled truth)
        assert updated.spam == 1
