"""RateMyCollezzz Review Model Corpus (RMC) — production corpus package.

This package implements the Gate C production corpus collection and annotation
infrastructure for the RMC.

Architectural note:
    RRM UNDERSTANDS. TRUST DECIDES.

    This package is data infrastructure for the RRM.  It produces annotated
    corpus records and evidence metadata.  It does NOT make moderation,
    trust, or policy decisions.

Public API:
    All types and functions from the corpus modules are re-exported here.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
from rrm.corpus.models import (
    ALL_REASON_CODES,
    ANNOTATION_INCOMPLETE,
    BROKEN_PARENT_LINK,
    AnnotationStatus,
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
    AnnotationSubmission,
    SourceType,
    CanonicalRecord,
    UNRESOLVED_GUIDE_AMBIGUITY,
    UNSAFE_PII,
    CONTROLLED_TRUTH_INCONSISTENCY,
)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
from rrm.corpus.validation import (
    validate_annotation_submission,
    validate_controlled_truth,
    validate_label,
    validate_record,
    validate_synthetic_provenance,
    gate_d_eligibility_qc,
)

# ---------------------------------------------------------------------------
# Annotation stores and helpers
# ---------------------------------------------------------------------------
from rrm.corpus.annotation import (
    AnnotationSubmissionStore,
    ControlledProtocolTruthStore,
    DispositionRegister,
    ReannotationRegister,
    compute_eight_dimension_disagreement,
    needs_adjudication,
    adjudicator_decision,
)

# ---------------------------------------------------------------------------
# PII adapter
# ---------------------------------------------------------------------------
from rrm.corpus.pii_adapter import (
    build_pii_evidence_summary,
    redact_pii_in_text,
    safe_surrogate_transform,
)

# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------
from rrm.corpus.workflow import CorpusWorkflow

# ---------------------------------------------------------------------------
# Intake adapter
# ---------------------------------------------------------------------------
from rrm.corpus.intake import (
    HumanWrittenIntakeResult,
    collect_human_written_review,
)

# ---------------------------------------------------------------------------
# Collection store (Gate-C restricted local persistence)
# ---------------------------------------------------------------------------
from rrm.corpus.collection_store import (
    CollectionSession,
    HumanWrittenCollectionStore,
)

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
from rrm.corpus.export import (
    PORTABLE_CANDIDATE_FIELDS,
    EXPORT_WHITELIST,
    export_corpus,
    export_metadata,
    export_to_jsonl,
)
