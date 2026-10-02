"""Canonical types, enums, and frozen dataclasses for the RMC corpus.

This module is the single owner of:
- canonical enum types (SourceType, LanguageMix, CollegeCategory,
  AnnotationStatus, Disposition)
- the CanonicalRecord frozen dataclass (38 fields in exact order)
- internal operational dataclasses
- frozen reason-code constants

Architectural note:
    RRM UNDERSTANDS. TRUST DECIDES.

    These types represent data structure only.  Validation, policy
    decisions, and moderation actions belong elsewhere.

Public API:
    SourceType, LanguageMix, CollegeCategory, AnnotationStatus, Disposition
    CanonicalRecord
    AnnotationSubmission, ControlledProtocolTruth, RecordDisposition,
    ReannotationRequirement, ConsentArtifact, PIIEvidenceSummary
    INVALID_CONSENT, EMPTY_TEXT, MALFORMED_RECORD, DUPLICATE_RECORD,
    UNSAFE_PII, MISSING_PROVENANCE, BROKEN_PARENT_LINK,
    CONTROLLED_TRUTH_INCONSISTENCY, ANNOTATION_INCOMPLETE,
    UNRESOLVED_GUIDE_AMBIGUITY
    ALL_REASON_CODES
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Canonical enum types
# ---------------------------------------------------------------------------

class SourceType(enum.Enum):
    """Source class for an RMC record."""

    HUMAN_WRITTEN_RMC = "HUMAN_WRITTEN_RMC"
    CONTROLLED_RMC = "CONTROLLED_RMC"
    SYNTHETIC_DERIVED_RMC = "SYNTHETIC_DERIVED_RMC"


class LanguageMix(enum.Enum):
    """Language mix classification for an RMC record."""

    ENGLISH = "ENGLISH"
    HINGLISH = "HINGLISH"
    ROMAN_HINDI = "ROMAN_HINDI"
    OTHER = "OTHER"
    MIXED_OTHER = "MIXED_OTHER"


class CollegeCategory(enum.Enum):
    """College topic category for an RMC record."""

    ACADEMICS = "ACADEMICS"
    FACULTY = "FACULTY"
    PLACEMENTS = "PLACEMENTS"
    HOSTEL = "HOSTEL"
    INFRASTRUCTURE = "INFRASTRUCTURE"
    FEES = "FEES"
    ADMINISTRATION = "ADMINISTRATION"
    CAMPUS_LIFE = "CAMPUS_LIFE"
    ADMISSIONS = "ADMISSIONS"
    MULTI_TOPIC = "MULTI_TOPIC"


class AnnotationStatus(enum.Enum):
    """Annotation workflow state for an RMC record."""

    UNANNOTATED = "UNANNOTATED"
    ANNOTATING = "ANNOTATING"
    ADJUDICATION_REQUIRED = "ADJUDICATION_REQUIRED"
    FINAL = "FINAL"
    EXCLUDED = "EXCLUDED"


class Disposition(enum.Enum):
    """Record disposition for Gate-D QC."""

    HOLD = "HOLD"
    EXCLUDED = "EXCLUDED"


# ---------------------------------------------------------------------------
# Frozen reason-code constants
# ---------------------------------------------------------------------------

#: Record was excluded due to invalid or missing consent.
INVALID_CONSENT: str = "INVALID_CONSENT"

#: Record was excluded because review_text is empty or blank.
EMPTY_TEXT: str = "EMPTY_TEXT"

#: Record was excluded because it is structurally malformed.
MALFORMED_RECORD: str = "MALFORMED_RECORD"

#: Record was excluded because it is an exact or near-duplicate.
DUPLICATE_RECORD: str = "DUPLICATE_RECORD"

#: Record was excluded because unsafe PII was detected.
UNSAFE_PII: str = "UNSAFE_PII"

#: Record was excluded due to missing required provenance fields.
MISSING_PROVENANCE: str = "MISSING_PROVENANCE"

#: Record was excluded because parent_review_id is invalid or missing
#: for a synthetic record.
BROKEN_PARENT_LINK: str = "BROKEN_PARENT_LINK"

#: Record was excluded because controlled target values are inconsistent
#: with the control protocol.
CONTROLLED_TRUTH_INCONSISTENCY: str = "CONTROLLED_TRUTH_INCONSISTENCY"

#: Record requires reannotation because the annotation guide was revised
#: after the original annotation.
ANNOTATION_INCOMPLETE: str = "ANNOTATION_INCOMPLETE"

#: Record requires reannotation because guide ambiguity caused disagreement
#: and the guide was subsequently revised.
UNRESOLVED_GUIDE_AMBIGUITY: str = "UNRESOLVED_GUIDE_AMBIGUITY"

#: Tuple of all valid reason codes in canonical order.
ALL_REASON_CODES: Tuple[str, ...] = (
    INVALID_CONSENT,
    EMPTY_TEXT,
    MALFORMED_RECORD,
    DUPLICATE_RECORD,
    UNSAFE_PII,
    MISSING_PROVENANCE,
    BROKEN_PARENT_LINK,
    CONTROLLED_TRUTH_INCONSISTENCY,
    ANNOTATION_INCOMPLETE,
    UNRESOLVED_GUIDE_AMBIGUITY,
)

#: Set of valid reason codes for fast membership testing.
_VALID_REASON_CODES: frozenset[str] = frozenset(ALL_REASON_CODES)


# ---------------------------------------------------------------------------
# CanonicalRecord — 38 fields in exact order
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CanonicalRecord:
    """Immutable RMC canonical record with exactly 38 fields.

    Fields are listed below in their canonical order.  This order MUST
    be preserved in all serialization, export, and display code.

    Attributes
    ----------
    review_id : str
        Unique record identifier.  The only required field.
    source_text_raw : str, optional
        Secure raw source text.  Never public, never neural input.
        Available only where governance permits.
    review_text : str, optional
        Canonical safe text for annotation, training, and runtime.
        Sole neural input.  Contains no real private values.
    export_text_redacted : str, optional
        Export-safe text.  Never neural input.
        May be more aggressively redacted than review_text.
    source_type : SourceType, optional
        Source class identifier.
    spam : int, optional
        spam label: 0 or 1.
    deception : int, optional
        deception label: -1, 0, or 1.
        -1 = UNKNOWN / insufficient ground truth.
    toxicity : int, optional
        toxicity label: 0 or 1.
    advertising : int, optional
        advertising label: 0 or 1.
    off_topic : int, optional
        off_topic label: 0 or 1.
    pii : int, optional
        pii label: 0 or 1.
    language_mix : LanguageMix, optional
        Language mix classification.  None when annotation is UNANNOTATED.
    college_category : CollegeCategory, optional
        College topic category.  None when no topic applies or
        annotation is UNANNOTATED.
    annotation_status : AnnotationStatus
        Annotation workflow state.  Defaults to UNANNOTATED.
    annotation_guide_version : str, optional
        Semantic version of the annotation guide used for final annotation.
    annotator_A_id : str, optional
        First annotator identifier (pseudonymous).
    annotator_B_id : str, optional
        Second annotator identifier (pseudonymous).
    adjudicator_id : str, optional
        Adjudicator identifier (pseudonymous).  Present only when
        adjudication occurred.
    finalized_at : str, optional
        ISO 8601 timestamp of finalization.  None until status = FINAL.
    annotation_notes : str, optional
        Free-text annotator or adjudicator notes.
    consent_status : str, optional
        Consent state for Human-Written RMC.
        "CONSENTED" is the Gate-D-eligible value.
        NOT an enum — plain optional string per Gate B contract.
    contributor_pseudonym : str, optional
        Pseudonymous contributor identifier.
    collection_method : str, optional
        How the review was collected.
    experiment_id : str, optional
        Controlled experiment identifier.  CONTROLLED_RMC only.
    controlled_targets : dict[str, int], optional
        Experimentally established ground-truth targets.
        Maps canonical label names to authoritative int values.
    deception_truth : int, optional
        Authoritative deception ground truth from experiment protocol.
        0 or 1 only.  Rejects -1.
    control_protocol_id : str, optional
        Protocol identifier for controlled ground truth.
    parent_review_id : str, optional
        Parent review identifier.  SYNTHETIC_DERIVED_RMC only.
    derivation_type : str, optional
        How the derivative was created.
    generation_method : str, optional
        Generation method for synthetic data.
    pii_detected : bool, optional
        Whether PII patterns were detected in the text.
    pii_categories : list[str], optional
        PII category labels (e.g., "EMAIL", "PHONE", "URL").
        Never contains the original private matched values.
    pii_redaction_status : str, optional
        Redaction/transformation status for PII.
    pii_evidence_id : str, optional
        Evidence identifier for PII detection.
    created_at : str, optional
        ISO 8601 record creation timestamp.
    dataset_version : str, optional
        Frozen dataset version identifier.  POPULATED_LATER by Gate D.
    split_membership : str, optional
        Train / validation / test assignment.  POPULATED_LATER by Gate E.
    split_group_id : str, optional
        Leakage-safe grouping identifier.  POPULATED_LATER by Gate E.

    Notes
    -----
    This dataclass is frozen.  Workflow state changes produce new
    CanonicalRecord instances.  Do not mutate in place.
    """

    # Field 1
    review_id: str

    # Field 2
    source_text_raw: Optional[str] = None

    # Field 3
    review_text: Optional[str] = None

    # Field 4
    export_text_redacted: Optional[str] = None

    # Field 5
    source_type: Optional[SourceType] = None

    # Field 6
    spam: Optional[int] = None

    # Field 7
    deception: Optional[int] = None

    # Field 8
    toxicity: Optional[int] = None

    # Field 9
    advertising: Optional[int] = None

    # Field 10
    off_topic: Optional[int] = None

    # Field 11
    pii: Optional[int] = None

    # Field 12
    language_mix: Optional[LanguageMix] = None

    # Field 13
    college_category: Optional[CollegeCategory] = None

    # Field 14
    annotation_status: AnnotationStatus = dataclasses.field(
        default=AnnotationStatus.UNANNOTATED
    )

    # Field 15
    annotation_guide_version: Optional[str] = None

    # Field 16
    annotator_A_id: Optional[str] = None

    # Field 17
    annotator_B_id: Optional[str] = None

    # Field 18
    adjudicator_id: Optional[str] = None

    # Field 19
    finalized_at: Optional[str] = None

    # Field 20
    annotation_notes: Optional[str] = None

    # Field 21
    consent_status: Optional[str] = None

    # Field 22
    contributor_pseudonym: Optional[str] = None

    # Field 23
    collection_method: Optional[str] = None

    # Field 24
    experiment_id: Optional[str] = None

    # Field 25
    controlled_targets: Optional[Dict[str, int]] = None

    # Field 26
    deception_truth: Optional[int] = None

    # Field 27
    control_protocol_id: Optional[str] = None

    # Field 28
    parent_review_id: Optional[str] = None

    # Field 29
    derivation_type: Optional[str] = None

    # Field 30
    generation_method: Optional[str] = None

    # Field 31
    pii_detected: Optional[bool] = None

    # Field 32
    pii_categories: Optional[List[str]] = None

    # Field 33
    pii_redaction_status: Optional[str] = None

    # Field 34
    pii_evidence_id: Optional[str] = None

    # Field 35
    created_at: Optional[str] = None

    # Field 36
    dataset_version: Optional[str] = None

    # Field 37
    split_membership: Optional[str] = None

    # Field 38
    split_group_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal operational dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class AnnotationSubmission:
    """Immutable record of a single annotator's submission.

    Attributes
    ----------
    review_id : str
        The record being annotated.
    annotator_id : str
        Pseudonymous annotator identifier.
    annotation_guide_version : str
        Semantic version of the guide used (e.g., "1.0").
    spam : int
        0 or 1.
    deception : int
        -1, 0, or 1.  Accepts -1 for UNKNOWN per annotation rules.
    toxicity : int
        0 or 1.
    advertising : int
        0 or 1.
    off_topic : int
        0 or 1.
    pii : int
        0 or 1.
    language_mix : LanguageMix, optional
        Language mix classification.
    college_category : CollegeCategory, optional
        College topic category.
    annotation_note : str, optional
        Free-text note from the annotator.
    submitted_at : str
        ISO 8601 timestamp of submission.
    """

    review_id: str
    annotator_id: str
    annotation_guide_version: str
    spam: int
    deception: int
    toxicity: int
    advertising: int
    off_topic: int
    pii: int
    language_mix: Optional[LanguageMix] = None
    college_category: Optional[CollegeCategory] = None
    annotation_note: Optional[str] = None
    submitted_at: str = ""


@dataclasses.dataclass(frozen=True)
class ControlledProtocolTruth:
    """Authoritative ground truth from a controlled experimental protocol.

    This type is used for CONTROLLED_RMC records where the experimental
    protocol establishes definitive label values, not annotator voting.

    Attributes
    ----------
    review_id : str
        The record this truth applies to.
    control_protocol_id : str
        Identifier for the control protocol.
    target_values : dict[str, int]
        Mapping of canonical label names to authoritative int values.
        deception_truth must be 0 or 1 (never -1).
    """

    review_id: str
    control_protocol_id: str
    target_values: Dict[str, int]


@dataclasses.dataclass(frozen=True)
class RecordDisposition:
    """Immutable record of a Gate-D QC disposition decision.

    Attributes
    ----------
    review_id : str
        The record being dispositioned.
    disposition : Disposition
        HOLD or EXCLUDED.
    reason_code : str
        One of the 10 canonical reason codes.
    internal_note : str, optional
        Internal explanation (not exported).
    recorded_at : str
        ISO 8601 timestamp of the disposition.
    """

    review_id: str
    disposition: Disposition
    reason_code: str
    internal_note: Optional[str] = None
    recorded_at: str = ""


@dataclasses.dataclass(frozen=True)
class ReannotationRequirement:
    """Immutable record of a reannotation requirement.

    Triggered when the annotation guide is revised after initial annotation.

    Attributes
    ----------
    review_id : str
        The record requiring reannotation.
    old_guide_version : str
        Version of the guide used for the original annotation.
    required_guide_version : str
        Version of the guide that must be used for reannotation.
    reason : str
        Explanation of why reannotation is required.
    recorded_at : str
        ISO 8601 timestamp when the requirement was recorded.
    """

    review_id: str
    old_guide_version: str
    required_guide_version: str
    reason: str
    recorded_at: str = ""


@dataclasses.dataclass(frozen=True)
class ConsentArtifact:
    """Immutable record of consent evidence for Human-Written RMC.

    Attributes
    ----------
    review_id : str
        The record this consent applies to.
    consent_status : str
        Consent state string (e.g., "CONSENTED").
    collected_at : str
        ISO 8601 timestamp of consent collection.
    method : str
        How consent was obtained.
    evidence_ref : str, optional
        Reference to stored consent evidence.
    """

    review_id: str
    consent_status: str
    collected_at: str
    method: str
    evidence_ref: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class PIIEvidenceSummary:
    """Immutable summary of PII evidence for a record.

    IMPORTANT: This object must contain NO original private matched values.

    Attributes
    ----------
    evidence_id : str
        Unique evidence identifier.
    review_id : str
        The record this evidence belongs to.
    categories : tuple[str, ...]
        PII category labels (e.g., "EMAIL", "PHONE", "URL").
    match_count : int
        Total number of PII patterns detected.
    redaction_status : str
        Redaction or transformation status.
    created_at : str
        ISO 8601 timestamp.
    """

    evidence_id: str
    review_id: str
    categories: Tuple[str, ...]
    match_count: int
    redaction_status: str
    created_at: str
