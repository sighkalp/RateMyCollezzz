"""Tests for rrm.corpus.models and rrm.corpus.validation.

Public API tested:
    SourceType, LanguageMix, CollegeCategory, AnnotationStatus, Disposition
    CanonicalRecord, AnnotationSubmission, ControlledProtocolTruth,
    RecordDisposition, ReannotationRequirement, ConsentArtifact, PIIEvidenceSummary
    INVALID_CONSENT, EMPTY_TEXT, MALFORMED_RECORD, DUPLICATE_RECORD,
    UNSAFE_PII, MISSING_PROVENANCE, BROKEN_PARENT_LINK,
    CONTROLLED_TRUTH_INCONSISTENCY, ANNOTATION_INCOMPLETE,
    UNRESOLVED_GUIDE_AMBIGUITY, ALL_REASON_CODES
    validate_record, validate_label, validate_annotation_submission,
    validate_controlled_truth, validate_synthetic_provenance,
    gate_d_eligibility_qc
"""

from __future__ import annotations

import dataclasses
import pytest

from rrm.corpus.models import (
    ALL_REASON_CODES,
    ANNOTATION_INCOMPLETE,
    BROKEN_PARENT_LINK,
    AnnotationStatus,
    AnnotationSubmission,
    CanonicalRecord,
    CollegeCategory,
    ConsentArtifact,
    ControlledProtocolTruth,
    Disposition,
    DUPLICATE_RECORD,
    EMPTY_TEXT,
    INVALID_CONSENT,
    LanguageMix,
    MALFORMED_RECORD,
    MISSING_PROVENANCE,
    PIIEvidenceSummary,
    ReannotationRequirement,
    RecordDisposition,
    SourceType,
    UNSAFE_PII,
    CONTROLLED_TRUTH_INCONSISTENCY,
    UNRESOLVED_GUIDE_AMBIGUITY,
)
from rrm.corpus.validation import (
    gate_d_eligibility_qc,
    validate_annotation_submission,
    validate_controlled_truth,
    validate_label,
    validate_record,
    validate_synthetic_provenance,
)
from rrm.corpus.annotation import (
    AnnotationSubmissionStore,
    ControlledProtocolTruthStore,
    DispositionRegister,
    ReannotationRegister,
    GateCOperationalContext,
)


# ---------------------------------------------------------------------------
# Reusable test helper
# ---------------------------------------------------------------------------

def _make_sub(
    review_id: str = "r-1",
    annotator_id: str = "ann-a",
    annotation_guide_version: str = "1.0",
    deception: int = 0,
    spam: int = 0,
) -> AnnotationSubmission:
    """Build a minimal valid AnnotationSubmission for tests."""
    return AnnotationSubmission(
        review_id=review_id,
        annotator_id=annotator_id,
        annotation_guide_version=annotation_guide_version,
        spam=spam,
        deception=deception,
        toxicity=0,
        advertising=0,
        off_topic=0,
        pii=0,
        submitted_at="2025-01-01T00:00:00Z",
    )


def _make_operational_context(
    review_id: str = "r-1",
    *,
    sub_a: AnnotationSubmission | None = None,
    sub_b: AnnotationSubmission | None = None,
    known_review_ids: frozenset | None = None,
    controlled_truth: ControlledProtocolTruth | None = None,
    inherited_targets: dict | None = None,
) -> GateCOperationalContext:
    """Build a real GateCOperationalContext for tests.

    Parameters
    ----------
    review_id : str
        The review ID this context is centered on.
    sub_a, sub_b : AnnotationSubmission or None
        Optional annotator submissions.  When not supplied, no
        submissions exist for *review_id* in the store.
    known_review_ids : frozenset or None
        Set of known review IDs.  Defaults to {review_id}.
    controlled_truth : ControlledProtocolTruth or None
        Optional controlled protocol truth.
    inherited_targets : dict or None
        Mapping review_id -> frozenset of target names with inherited
        evidence.

    Returns
    -------
    GateCOperationalContext
    """
    submissions = AnnotationSubmissionStore()
    if sub_a is not None:
        submissions.submit(sub_a)
    if sub_b is not None:
        submissions.submit(sub_b)

    truths = ControlledProtocolTruthStore()
    if controlled_truth is not None:
        truths.set_truth(controlled_truth)

    return GateCOperationalContext(
        submission_store=submissions,
        controlled_truth_store=truths,
        disposition_register=DispositionRegister(),
        reannotation_register=ReannotationRegister(),
        known_review_ids=known_review_ids or frozenset({review_id}),
        inherited_target_evidence=inherited_targets or {},
    )


# ---------------------------------------------------------------------------
# CanonicalRecord — construction and defaults
# ---------------------------------------------------------------------------


class TestCanonicalRecordConstruction:
    """CanonicalRecord construction and field defaults."""

    def test_minimal_construction(self):
        """CanonicalRecord can be created with only review_id."""
        rec = CanonicalRecord(review_id="r-001")
        assert rec.review_id == "r-001"

    def test_default_annotation_status(self):
        """Default annotation_status is UNANNOTATED."""
        rec = CanonicalRecord(review_id="r-001")
        assert rec.annotation_status == AnnotationStatus.UNANNOTATED

    def test_all_fields_none_by_default(self):
        """All non-required fields default to None."""
        rec = CanonicalRecord(review_id="r-001")
        assert rec.source_text_raw is None
        assert rec.review_text is None
        assert rec.export_text_redacted is None
        assert rec.source_type is None
        assert rec.spam is None
        assert rec.deception is None
        assert rec.toxicity is None
        assert rec.advertising is None
        assert rec.off_topic is None
        assert rec.pii is None
        assert rec.language_mix is None
        assert rec.college_category is None
        assert rec.annotation_guide_version is None
        assert rec.annotator_A_id is None
        assert rec.annotator_B_id is None
        assert rec.adjudicator_id is None
        assert rec.finalized_at is None
        assert rec.annotation_notes is None
        assert rec.consent_status is None
        assert rec.contributor_pseudonym is None
        assert rec.collection_method is None
        assert rec.experiment_id is None
        assert rec.controlled_targets is None
        assert rec.deception_truth is None
        assert rec.control_protocol_id is None
        assert rec.parent_review_id is None
        assert rec.derivation_type is None
        assert rec.generation_method is None
        assert rec.pii_detected is None
        assert rec.pii_categories is None
        assert rec.pii_redaction_status is None
        assert rec.pii_evidence_id is None
        assert rec.created_at is None
        assert rec.dataset_version is None
        assert rec.split_membership is None
        assert rec.split_group_id is None

    def test_frozen(self):
        """CanonicalRecord is frozen — cannot be mutated."""
        rec = CanonicalRecord(review_id="r-001")
        with pytest.raises(dataclasses.FrozenInstanceError):
            rec.review_id = "r-002"

    def test_full_construction(self):
        """CanonicalRecord accepts all 38 fields."""
        rec = CanonicalRecord(
            review_id="r-full",
            source_text_raw="raw text",
            review_text="safe text",
            export_text_redacted="redacted",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            spam=0,
            deception=-1,
            toxicity=1,
            advertising=0,
            off_topic=0,
            pii=0,
            language_mix=LanguageMix.ENGLISH,
            college_category=CollegeCategory.ACADEMICS,
            annotation_status=AnnotationStatus.FINAL,
            annotation_guide_version="1.0",
            annotator_A_id="ann-a",
            annotator_B_id="ann-b",
            adjudicator_id="adj-1",
            finalized_at="2025-01-01T00:00:00Z",
            annotation_notes="notes",
            consent_status="CONSENTED",
            contributor_pseudonym="contrib",
            collection_method="manual",
            experiment_id="exp-1",
            controlled_targets={"spam": 0, "deception": 1},
            deception_truth=0,
            control_protocol_id="proto-1",
            parent_review_id="parent",
            derivation_type="paraphrase",
            generation_method="template",
            pii_detected=False,
            pii_categories=["EMAIL"],
            pii_redaction_status="redacted",
            pii_evidence_id="evid-1",
            created_at="2025-01-01T00:00:00Z",
            dataset_version="v1.0",
            split_membership="train",
            split_group_id="group-1",
        )
        assert rec.review_id == "r-full"
        assert rec.deception == -1
        assert rec.deception_truth == 0
        assert rec.pii_categories == ["EMAIL"]
        assert rec.split_membership == "train"



# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestEnums:
    """Enum type tests."""

    def test_source_type_values(self):
        assert SourceType.HUMAN_WRITTEN_RMC.value == "HUMAN_WRITTEN_RMC"
        assert SourceType.CONTROLLED_RMC.value == "CONTROLLED_RMC"
        assert SourceType.SYNTHETIC_DERIVED_RMC.value == "SYNTHETIC_DERIVED_RMC"

    def test_language_mix_values(self):
        assert LanguageMix.ENGLISH.value == "ENGLISH"
        assert LanguageMix.HINGLISH.value == "HINGLISH"
        assert LanguageMix.ROMAN_HINDI.value == "ROMAN_HINDI"
        assert LanguageMix.OTHER.value == "OTHER"
        assert LanguageMix.MIXED_OTHER.value == "MIXED_OTHER"

    def test_college_category_values(self):
        assert CollegeCategory.ACADEMICS.value == "ACADEMICS"
        assert CollegeCategory.FACULTY.value == "FACULTY"
        assert CollegeCategory.PLACEMENTS.value == "PLACEMENTS"
        assert CollegeCategory.HOSTEL.value == "HOSTEL"
        assert CollegeCategory.INFRASTRUCTURE.value == "INFRASTRUCTURE"
        assert CollegeCategory.FEES.value == "FEES"
        assert CollegeCategory.ADMINISTRATION.value == "ADMINISTRATION"
        assert CollegeCategory.CAMPUS_LIFE.value == "CAMPUS_LIFE"
        assert CollegeCategory.ADMISSIONS.value == "ADMISSIONS"
        assert CollegeCategory.MULTI_TOPIC.value == "MULTI_TOPIC"

    def test_annotation_status_values(self):
        assert AnnotationStatus.UNANNOTATED.value == "UNANNOTATED"
        assert AnnotationStatus.ANNOTATING.value == "ANNOTATING"
        assert AnnotationStatus.ADJUDICATION_REQUIRED.value == "ADJUDICATION_REQUIRED"
        assert AnnotationStatus.FINAL.value == "FINAL"
        assert AnnotationStatus.EXCLUDED.value == "EXCLUDED"

    def test_disposition_values(self):
        assert Disposition.HOLD.value == "HOLD"
        assert Disposition.EXCLUDED.value == "EXCLUDED"


# ---------------------------------------------------------------------------
# Reason codes
# ---------------------------------------------------------------------------


class TestReasonCodes:
    """Frozen reason-code constants."""

    def test_all_ten_present(self):
        assert len(ALL_REASON_CODES) == 10

    def test_exact_spelling(self):
        expected = (
            "INVALID_CONSENT",
            "EMPTY_TEXT",
            "MALFORMED_RECORD",
            "DUPLICATE_RECORD",
            "UNSAFE_PII",
            "MISSING_PROVENANCE",
            "BROKEN_PARENT_LINK",
            "CONTROLLED_TRUTH_INCONSISTENCY",
            "ANNOTATION_INCOMPLETE",
            "UNRESOLVED_GUIDE_AMBIGUITY",
        )
        assert ALL_REASON_CODES == expected

    def test_individual_constants(self):
        assert INVALID_CONSENT == "INVALID_CONSENT"
        assert EMPTY_TEXT == "EMPTY_TEXT"
        assert MALFORMED_RECORD == "MALFORMED_RECORD"
        assert DUPLICATE_RECORD == "DUPLICATE_RECORD"
        assert UNSAFE_PII == "UNSAFE_PII"
        assert MISSING_PROVENANCE == "MISSING_PROVENANCE"
        assert BROKEN_PARENT_LINK == "BROKEN_PARENT_LINK"
        assert CONTROLLED_TRUTH_INCONSISTENCY == "CONTROLLED_TRUTH_INCONSISTENCY"
        assert ANNOTATION_INCOMPLETE == "ANNOTATION_INCOMPLETE"
        assert UNRESOLVED_GUIDE_AMBIGUITY == "UNRESOLVED_GUIDE_AMBIGUITY"


# ---------------------------------------------------------------------------
# Operational dataclasses
# ---------------------------------------------------------------------------


class TestOperationalDataclasses:
    """Internal operational dataclass tests."""

    def test_annotation_submission_frozen(self):
        sub = AnnotationSubmission(
            review_id="r-1",
            annotator_id="ann-a",
            annotation_guide_version="1.0",
            spam=0,
            deception=-1,
            toxicity=0,
            advertising=0,
            off_topic=0,
            pii=0,
            submitted_at="2025-01-01T00:00:00Z",
        )
        assert sub.review_id == "r-1"
        assert sub.deception == -1
        with pytest.raises(dataclasses.FrozenInstanceError):
            sub.spam = 1

    def test_controlled_truth_frozen(self):
        truth = ControlledProtocolTruth(
            review_id="r-1",
            control_protocol_id="proto-1",
            target_values={"spam": 0, "deception": 1},
        )
        assert truth.target_values == {"spam": 0, "deception": 1}
        with pytest.raises(dataclasses.FrozenInstanceError):
            truth.target_values = {}

    def test_record_disposition_frozen(self):
        disp = RecordDisposition(
            review_id="r-1",
            disposition=Disposition.HOLD,
            reason_code=INVALID_CONSENT,
            recorded_at="2025-01-01T00:00:00Z",
        )
        assert disp.disposition == Disposition.HOLD
        with pytest.raises(dataclasses.FrozenInstanceError):
            disp.reason_code = EMPTY_TEXT

    def test_reannotation_requirement_frozen(self):
        req = ReannotationRequirement(
            review_id="r-1",
            old_guide_version="1.0",
            required_guide_version="2.0",
            reason="Guide revised",
            recorded_at="2025-01-01T00:00:00Z",
        )
        assert req.required_guide_version == "2.0"
        with pytest.raises(dataclasses.FrozenInstanceError):
            req.reason = "changed"

    def test_consent_artifact_frozen(self):
        art = ConsentArtifact(
            review_id="r-1",
            consent_status="CONSENTED",
            collected_at="2025-01-01T00:00:00Z",
            method="online_form",
            evidence_ref="ref-1",
        )
        assert art.consent_status == "CONSENTED"
        with pytest.raises(dataclasses.FrozenInstanceError):
            art.evidence_ref = "new-ref"

    def test_pii_evidence_summary_frozen(self):
        summary = PIIEvidenceSummary(
            evidence_id="evid-1",
            review_id="r-1",
            categories=("EMAIL", "PHONE"),
            match_count=2,
            redaction_status="redacted",
            created_at="2025-01-01T00:00:00Z",
        )
        assert summary.match_count == 2
        assert len(summary.categories) == 2
        with pytest.raises(dataclasses.FrozenInstanceError):
            summary.match_count = 5


# ---------------------------------------------------------------------------
# validate_label
# ---------------------------------------------------------------------------


class TestValidateLabel:
    """Label domain validation."""

    def test_spam_valid_values(self):
        assert validate_label("spam", 0) is True
        assert validate_label("spam", 1) is True

    def test_spam_invalid_value(self):
        assert validate_label("spam", -1) is False
        assert validate_label("spam", 2) is False

    def test_deception_accepts_minus_one(self):
        assert validate_label("deception", -1) is True
        assert validate_label("deception", 0) is True
        assert validate_label("deception", 1) is True

    def test_deception_invalid_value(self):
        assert validate_label("deception", 2) is False

    def test_toxicity_valid_values(self):
        assert validate_label("toxicity", 0) is True
        assert validate_label("toxicity", 1) is True

    def test_advertising_valid_values(self):
        assert validate_label("advertising", 0) is True
        assert validate_label("advertising", 1) is True

    def test_off_topic_valid_values(self):
        assert validate_label("off_topic", 0) is True
        assert validate_label("off_topic", 1) is True

    def test_pii_valid_values(self):
        assert validate_label("pii", 0) is True
        assert validate_label("pii", 1) is True

    def test_unknown_label_returns_false(self):
        assert validate_label("nonexistent", 0) is False

    def test_type_error_non_string_label(self):
        with pytest.raises(TypeError):
            validate_label(123, 0)

    def test_type_error_non_int_value(self):
        with pytest.raises(TypeError):
            validate_label("spam", "0")


# ---------------------------------------------------------------------------
# validate_record
# ---------------------------------------------------------------------------


class TestValidateRecord:
    """CanonicalRecord structural and domain validation."""

    def test_minimal_valid_record(self):
        rec = CanonicalRecord(review_id="r-1")
        is_valid, errors = validate_record(rec)
        assert is_valid is True
        assert errors == []

    def test_empty_review_id_fails(self):
        rec = CanonicalRecord(review_id="")
        is_valid, errors = validate_record(rec)
        assert is_valid is False
        assert any("review_id" in e for e in errors)

    def test_non_string_review_id_fails(self):
        rec = CanonicalRecord(review_id=123)  # type: ignore
        is_valid, errors = validate_record(rec)
        assert is_valid is False

    def test_valid_labels(self):
        rec = CanonicalRecord(
            review_id="r-1",
            spam=1,
            toxicity=0,
            advertising=1,
            off_topic=0,
            pii=1,
        )
        is_valid, errors = validate_record(rec)
        assert is_valid is True
        assert errors == []

    def test_invalid_label_value(self):
        rec = CanonicalRecord(review_id="r-1", spam=2)
        is_valid, errors = validate_record(rec)
        assert is_valid is False
        assert any("spam" in e for e in errors)

    def test_invalid_deception_value(self):
        rec = CanonicalRecord(review_id="r-1", deception=2)
        is_valid, errors = validate_record(rec)
        assert is_valid is False
        assert any("deception" in e for e in errors)

    def test_valid_deception_minus_one(self):
        rec = CanonicalRecord(review_id="r-1", deception=-1)
        is_valid, errors = validate_record(rec)
        assert is_valid is True

    def test_invalid_label_type(self):
        rec = CanonicalRecord(review_id="r-1", spam="1")  # type: ignore
        is_valid, errors = validate_record(rec)
        assert is_valid is False
        assert any("spam" in e and "int" in e for e in errors)

    def test_invalid_source_type_type(self):
        rec = CanonicalRecord(review_id="r-1", source_type="HUMAN_WRITTEN_RMC")  # type: ignore
        is_valid, errors = validate_record(rec)
        assert is_valid is False
        assert any("source_type" in e for e in errors)

    def test_valid_source_type(self):
        rec = CanonicalRecord(
            review_id="r-1", source_type=SourceType.HUMAN_WRITTEN_RMC
        )
        is_valid, errors = validate_record(rec)
        assert is_valid is True

    def test_valid_annotation_status(self):
        rec = CanonicalRecord(review_id="r-1")
        is_valid, errors = validate_record(rec)
        assert is_valid is True

    def test_invalid_annotation_guide_version(self):
        rec = CanonicalRecord(
            review_id="r-1",
            annotation_guide_version="not-a-version",
        )
        is_valid, errors = validate_record(rec)
        assert is_valid is False
        assert any("annotation_guide_version" in e for e in errors)

    def test_valid_annotation_guide_version(self):
        rec = CanonicalRecord(
            review_id="r-1",
            annotation_guide_version="1.0",
        )
        is_valid, errors = validate_record(rec)
        assert is_valid is True

    def test_valid_deception_truth_zero(self):
        rec = CanonicalRecord(review_id="r-1", deception_truth=0)
        is_valid, errors = validate_record(rec)
        assert is_valid is True

    def test_invalid_deception_truth_minus_one(self):
        rec = CanonicalRecord(review_id="r-1", deception_truth=-1)
        is_valid, errors = validate_record(rec)
        assert is_valid is False
        assert any("deception_truth" in e for e in errors)

    def test_invalid_pii_detected_type(self):
        rec = CanonicalRecord(review_id="r-1", pii_detected="true")  # type: ignore
        is_valid, errors = validate_record(rec)
        assert is_valid is False
        assert any("pii_detected" in e for e in errors)

    def test_valid_created_at(self):
        rec = CanonicalRecord(
            review_id="r-1", created_at="2025-01-01T00:00:00Z"
        )
        is_valid, errors = validate_record(rec)
        assert is_valid is True

    def test_invalid_created_at(self):
        rec = CanonicalRecord(
            review_id="r-1", created_at="not-a-date"
        )
        is_valid, errors = validate_record(rec)
        assert is_valid is False
        assert any("created_at" in e for e in errors)

    def test_valid_controlled_targets(self):
        rec = CanonicalRecord(
            review_id="r-1",
            controlled_targets={"spam": 0, "deception": 1},
        )
        is_valid, errors = validate_record(rec)
        assert is_valid is True

    def test_invalid_controlled_target_key(self):
        rec = CanonicalRecord(
            review_id="r-1",
            controlled_targets={"nonexistent": 0},
        )
        is_valid, errors = validate_record(rec)
        assert is_valid is False
        assert any("canonical label" in e for e in errors)

    def test_invalid_controlled_target_value(self):
        rec = CanonicalRecord(
            review_id="r-1",
            controlled_targets={"spam": 2},
        )
        is_valid, errors = validate_record(rec)
        assert is_valid is False
        assert any("spam" in e for e in errors)


# ---------------------------------------------------------------------------
# validate_annotation_submission
# ---------------------------------------------------------------------------


class TestValidateAnnotationSubmission:
    """AnnotationSubmission validation."""

    def test_valid_submission(self):
        sub = AnnotationSubmission(
            review_id="r-1",
            annotator_id="ann-a",
            annotation_guide_version="1.0",
            spam=0,
            deception=-1,
            toxicity=1,
            advertising=0,
            off_topic=0,
            pii=0,
            submitted_at="2025-01-01T00:00:00Z",
        )
        is_valid, errors = validate_annotation_submission(sub)
        assert is_valid is True
        assert errors == []

    def test_invalid_label_value(self):
        sub = AnnotationSubmission(
            review_id="r-1",
            annotator_id="ann-a",
            annotation_guide_version="1.0",
            spam=2,
            deception=0,
            toxicity=0,
            advertising=0,
            off_topic=0,
            pii=0,
            submitted_at="2025-01-01T00:00:00Z",
        )
        is_valid, errors = validate_annotation_submission(sub)
        assert is_valid is False
        assert any("spam" in e for e in errors)

    def test_empty_annotator_id(self):
        sub = AnnotationSubmission(
            review_id="r-1",
            annotator_id="",
            annotation_guide_version="1.0",
            spam=0,
            deception=0,
            toxicity=0,
            advertising=0,
            off_topic=0,
            pii=0,
            submitted_at="2025-01-01T00:00:00Z",
        )
        is_valid, errors = validate_annotation_submission(sub)
        assert is_valid is False
        assert any("annotator_id" in e for e in errors)

    def test_invalid_guide_version(self):
        sub = AnnotationSubmission(
            review_id="r-1",
            annotator_id="ann-a",
            annotation_guide_version="bad",
            spam=0,
            deception=0,
            toxicity=0,
            advertising=0,
            off_topic=0,
            pii=0,
            submitted_at="2025-01-01T00:00:00Z",
        )
        is_valid, errors = validate_annotation_submission(sub)
        assert is_valid is False
        assert any("annotation_guide_version" in e for e in errors)

    def test_empty_submitted_at(self):
        sub = AnnotationSubmission(
            review_id="r-1",
            annotator_id="ann-a",
            annotation_guide_version="1.0",
            spam=0,
            deception=0,
            toxicity=0,
            advertising=0,
            off_topic=0,
            pii=0,
            submitted_at="",
        )
        is_valid, errors = validate_annotation_submission(sub)
        assert is_valid is False
        assert any("submitted_at" in e for e in errors)


# ---------------------------------------------------------------------------
# validate_controlled_truth
# ---------------------------------------------------------------------------


class TestValidateControlledTruth:
    """ControlledProtocolTruth validation."""

    def test_valid_truth_with_controlled_record(self):
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.CONTROLLED_RMC,
        )
        truth = ControlledProtocolTruth(
            review_id="r-1",
            control_protocol_id="proto-1",
            target_values={"spam": 0, "deception": 1},
        )
        is_valid, errors = validate_controlled_truth(truth, record)
        assert is_valid is True
        assert errors == []

    def test_requires_controlled_source(self):
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
        )
        truth = ControlledProtocolTruth(
            review_id="r-1",
            control_protocol_id="proto-1",
            target_values={"spam": 0},
        )
        is_valid, errors = validate_controlled_truth(truth, record)
        assert is_valid is False
        assert any("CONTROLLED_RMC" in e for e in errors)

    def test_empty_control_protocol_id(self):
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.CONTROLLED_RMC,
        )
        truth = ControlledProtocolTruth(
            review_id="r-1",
            control_protocol_id="",
            target_values={"spam": 0},
        )
        is_valid, errors = validate_controlled_truth(truth, record)
        assert is_valid is False
        assert any("control_protocol_id" in e for e in errors)

    def test_invalid_target_key(self):
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.CONTROLLED_RMC,
        )
        truth = ControlledProtocolTruth(
            review_id="r-1",
            control_protocol_id="proto-1",
            target_values={"nonexistent": 0},
        )
        is_valid, errors = validate_controlled_truth(truth, record)
        assert is_valid is False
        assert any("canonical label" in e for e in errors)

    def test_invalid_target_value(self):
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.CONTROLLED_RMC,
        )
        truth = ControlledProtocolTruth(
            review_id="r-1",
            control_protocol_id="proto-1",
            target_values={"spam": 2},
        )
        is_valid, errors = validate_controlled_truth(truth, record)
        assert is_valid is False
        assert any("spam" in e for e in errors)

    def test_deception_truth_rejects_minus_one(self):
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.CONTROLLED_RMC,
        )
        truth = ControlledProtocolTruth(
            review_id="r-1",
            control_protocol_id="proto-1",
            target_values={"deception": -1},
        )
        is_valid, errors = validate_controlled_truth(truth, record)
        assert is_valid is False
        assert any("deception" in e for e in errors)


# ---------------------------------------------------------------------------
# validate_synthetic_provenance
# ---------------------------------------------------------------------------


class TestValidateSyntheticProvenance:
    """Synthetic provenance validation."""

    def test_human_written_record_passes(self):
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
        )
        is_valid, errors = validate_synthetic_provenance(record)
        assert is_valid is True
        assert errors == []

    def test_synthetic_with_full_provenance(self):
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

    def test_synthetic_missing_parent(self):
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.SYNTHETIC_DERIVED_RMC,
            derivation_type="paraphrase",
            generation_method="llm",
        )
        is_valid, errors = validate_synthetic_provenance(record)
        assert is_valid is False
        assert any("parent_review_id" in e for e in errors)

    def test_synthetic_missing_derivation_type(self):
        """derivation_type is independently conditional — None is valid."""
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.SYNTHETIC_DERIVED_RMC,
            parent_review_id="parent-1",
            generation_method="llm",
        )
        is_valid, errors = validate_synthetic_provenance(record)
        assert is_valid is True
        assert errors == []

    def test_synthetic_missing_generation_method(self):
        """generation_method is independently conditional — None is valid."""
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.SYNTHETIC_DERIVED_RMC,
            parent_review_id="parent-1",
            derivation_type="paraphrase",
        )
        is_valid, errors = validate_synthetic_provenance(record)
        assert is_valid is True
        assert errors == []

    def test_synthetic_only_parent_review_id_valid(self):
        """Only parent_review_id required; both conditionals None → valid."""
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.SYNTHETIC_DERIVED_RMC,
            parent_review_id="parent-1",
        )
        is_valid, errors = validate_synthetic_provenance(record)
        assert is_valid is True
        assert errors == []

    def test_synthetic_empty_derivation_type_invalid(self):
        """Empty derivation_type when supplied → invalid."""
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.SYNTHETIC_DERIVED_RMC,
            parent_review_id="parent-1",
            derivation_type="",
        )
        is_valid, errors = validate_synthetic_provenance(record)
        assert is_valid is False
        assert any("derivation_type" in e for e in errors)

    def test_synthetic_empty_generation_method_invalid(self):
        """Empty generation_method when supplied → invalid."""
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.SYNTHETIC_DERIVED_RMC,
            parent_review_id="parent-1",
            generation_method="",
        )
        is_valid, errors = validate_synthetic_provenance(record)
        assert is_valid is False
        assert any("generation_method" in e for e in errors)

    def test_synthetic_whitespace_derivation_type_invalid(self):
        """Whitespace-only derivation_type when supplied → invalid."""
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.SYNTHETIC_DERIVED_RMC,
            parent_review_id="parent-1",
            derivation_type="   ",
        )
        is_valid, errors = validate_synthetic_provenance(record)
        assert is_valid is False
        assert any("derivation_type" in e for e in errors)

    def test_synthetic_whitespace_generation_method_invalid(self):
        """Whitespace-only generation_method when supplied → invalid."""
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.SYNTHETIC_DERIVED_RMC,
            parent_review_id="parent-1",
            generation_method="  ",
        )
        is_valid, errors = validate_synthetic_provenance(record)
        assert is_valid is False
        assert any("generation_method" in e for e in errors)


# ---------------------------------------------------------------------------
# gate_d_eligibility_qc
# ---------------------------------------------------------------------------


class TestGateDEligibilityQC:
    """Gate-D eligibility quality control."""

    def test_final_with_both_annotators_passes(self):
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            annotation_status=AnnotationStatus.FINAL,
            annotator_A_id="ann-a",
            annotator_B_id="ann-b",
            review_text="some review text",
            created_at="2025-01-01T00:00:00Z",
            consent_status="CONSENTED",
            deception=-1,
        )
        sub_a = _make_sub(review_id="r-1", annotator_id="ann-a")
        sub_b = _make_sub(review_id="r-1", annotator_id="ann-b")
        ctx = _make_operational_context("r-1", sub_a=sub_a, sub_b=sub_b)
        is_eligible, errors = gate_d_eligibility_qc(record, ctx)
        assert is_eligible is True
        assert errors == []

    def test_unannotated_fails(self):
        record = CanonicalRecord(
            review_id="r-1",
            review_text="text",
            created_at="2025-01-01T00:00:00Z",
        )
        ctx = _make_operational_context("r-1")
        is_eligible, errors = gate_d_eligibility_qc(record, ctx)
        assert is_eligible is False
        assert any("FINAL" in e for e in errors)

    def test_missing_annotator_a_fails(self):
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            annotation_status=AnnotationStatus.FINAL,
            annotator_B_id="ann-b",
            review_text="text",
            created_at="2025-01-01T00:00:00Z",
            consent_status="CONSENTED",
            deception=-1,
        )
        sub_b = _make_sub(review_id="r-1", annotator_id="ann-b")
        ctx = _make_operational_context("r-1", sub_b=sub_b)
        is_eligible, errors = gate_d_eligibility_qc(record, ctx)
        assert is_eligible is False
        assert any("annotator" in e.lower() for e in errors)

    def test_missing_review_text_fails(self):
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.HUMAN_WRITTEN_RMC,
            annotation_status=AnnotationStatus.FINAL,
            annotator_A_id="ann-a",
            annotator_B_id="ann-b",
            created_at="2025-01-01T00:00:00Z",
            consent_status="CONSENTED",
            deception=-1,
        )
        sub_a = _make_sub(review_id="r-1", annotator_id="ann-a")
        sub_b = _make_sub(review_id="r-1", annotator_id="ann-b")
        ctx = _make_operational_context("r-1", sub_a=sub_a, sub_b=sub_b)
        is_eligible, errors = gate_d_eligibility_qc(record, ctx)
        assert is_eligible is False
        assert any("review_text" in e for e in errors)

    def test_human_written_requires_consent(self):
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
        sub_a = _make_sub(review_id="r-1", annotator_id="ann-a")
        sub_b = _make_sub(review_id="r-1", annotator_id="ann-b")
        ctx = _make_operational_context("r-1", sub_a=sub_a, sub_b=sub_b)
        is_eligible, errors = gate_d_eligibility_qc(record, ctx)
        assert is_eligible is False
        assert any("CONSENTED" in e for e in errors)

    def test_controlled_requires_experiment_id(self):
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.CONTROLLED_RMC,
            annotation_status=AnnotationStatus.FINAL,
            annotator_A_id="ann-a",
            annotator_B_id="ann-b",
            review_text="text",
            created_at="2025-01-01T00:00:00Z",
        )
        ctx = _make_operational_context("r-1")
        is_eligible, errors = gate_d_eligibility_qc(record, ctx)
        assert is_eligible is False
        assert any("experiment_id" in e for e in errors)

    def test_controlled_with_experiment_id_passes(self):
        record = CanonicalRecord(
            review_id="r-1",
            source_type=SourceType.CONTROLLED_RMC,
            annotation_status=AnnotationStatus.FINAL,
            annotation_guide_version="1.0",
            finalized_at="2025-01-02T00:00:00Z",
            annotator_A_id="ann-a",
            annotator_B_id="ann-b",
            review_text="text",
            created_at="2025-01-01T00:00:00Z",
            experiment_id="exp-1",
            control_protocol_id="protocol-1",
            controlled_targets={"spam": 1},
            spam=1,
            deception=-1,
            deception_truth=None,
        )
        sub_a = _make_sub(review_id="r-1", annotator_id="ann-a", spam=1)
        sub_b = _make_sub(review_id="r-1", annotator_id="ann-b", spam=1)
        truth = ControlledProtocolTruth(
            review_id="r-1",
            control_protocol_id="protocol-1",
            target_values={"spam": 1},
        )
        ctx = _make_operational_context(
            "r-1",
            sub_a=sub_a,
            sub_b=sub_b,
            controlled_truth=truth,
        )
        is_eligible, errors = gate_d_eligibility_qc(record, ctx)
        assert is_eligible is True
        assert errors == []
