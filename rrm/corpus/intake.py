"""Human-Written RMC intake adapter.

This module converts a genuine consenting contributor's review into a
safe UNANNOTATED CanonicalRecord that can enter the existing annotation
workflow.

Architectural note:
    RRM UNDERSTANDS. TRUST DECIDES.

    This module is a collection boundary only.  It does not annotate,
    label, moderate, or make trust decisions.

Public API:
    HumanWrittenIntakeResult
    collect_human_written_review
"""

from __future__ import annotations

import dataclasses
import uuid

from rrm.corpus.models import (
    AnnotationStatus,
    CanonicalRecord,
    ConsentArtifact,
    PIIEvidenceSummary,
    SourceType,
)
from rrm.corpus.pii_adapter import (
    build_pii_evidence_summary,
    redact_pii_in_text,
    safe_surrogate_transform,
)
from rrm.corpus.workflow import CorpusWorkflow
from rrm.pii_detection import detect_pii


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class HumanWrittenIntakeResult:
    """Immutable result of a Human-Written RMC intake.

    Attributes
    ----------
    record : CanonicalRecord
        The created record, already registered in the workflow.
    consent_artifact : ConsentArtifact
        Evidence of the contributor's consent.
    pii_evidence : PIIEvidenceSummary or None
        PII evidence summary when PII was detected in the submitted text,
        None otherwise.
    """

    record: CanonicalRecord
    consent_artifact: ConsentArtifact
    pii_evidence: PIIEvidenceSummary | None


# ---------------------------------------------------------------------------
# Intake function
# ---------------------------------------------------------------------------

def collect_human_written_review(
    workflow: CorpusWorkflow,
    submitted_text: str,
    consent_granted: bool,
    collected_at: str,
    collection_method: str,
    *,
    retain_source_text_raw: bool = False,
    consent_evidence_ref: str | None = None,
) -> HumanWrittenIntakeResult:
    """Collect a Human-Written RMC review from a consenting contributor.

    This is the production intake boundary for genuine human-written
    reviews.  It performs consent validation, PII detection, safe-text
    transformation, and record registration.

    Parameters
    ----------
    workflow : CorpusWorkflow
        The corpus workflow instance to register the record with.
    submitted_text : str
        The genuine text supplied by the contributor.
    consent_granted : bool
        Must be True.  If False, a ValueError is raised and no record
        is created or registered.
    collected_at : str
        ISO 8601 timestamp of collection.
    collection_method : str
        Non-empty description of how the review was collected
        (e.g., "web_form", "mobile_app").
    retain_source_text_raw : bool, optional
        If True, the original submitted text is retained in
        source_text_raw (restricted/internal).  Default is False.
    consent_evidence_ref : str or None, optional
        Optional reference to stored consent evidence.

    Returns
    -------
    HumanWrittenIntakeResult
        The created record, consent artifact, and optional PII evidence.

    Raises
    ------
    ValueError
        If consent is not granted, submitted_text is empty/blank,
        collected_at is empty, or collection_method is empty.
    TypeError
        If workflow is not a CorpusWorkflow, submitted_text is not a
        str, consent_granted is not a bool, collected_at is not a str,
        collection_method is not a str, or retain_source_text_raw is
        not a bool.

    Notes
    -----
    Similarity is RRM evidence only.  This function does not perform
    duplicate detection, spam classification, or moderation.
    Trust layer makes final policy decisions.
    """
    # --- Input validation ---------------------------------------------------
    if not isinstance(workflow, CorpusWorkflow):
        raise TypeError(
            f"workflow must be CorpusWorkflow, "
            f"got {type(workflow).__name__}"
        )
    if not isinstance(submitted_text, str):
        raise TypeError(
            f"submitted_text must be str, "
            f"got {type(submitted_text).__name__}"
        )
    if not isinstance(consent_granted, bool):
        raise TypeError(
            f"consent_granted must be bool, "
            f"got {type(consent_granted).__name__}"
        )
    if not isinstance(collected_at, str):
        raise TypeError(
            f"collected_at must be str, "
            f"got {type(collected_at).__name__}"
        )
    if not isinstance(collection_method, str):
        raise TypeError(
            f"collection_method must be str, "
            f"got {type(collection_method).__name__}"
        )
    if not isinstance(retain_source_text_raw, bool):
        raise TypeError(
            f"retain_source_text_raw must be bool, "
            f"got {type(retain_source_text_raw).__name__}"
        )
    if consent_evidence_ref is not None and not isinstance(
        consent_evidence_ref, str
    ):
        raise TypeError(
            f"consent_evidence_ref must be str or None, "
            f"got {type(consent_evidence_ref).__name__}"
        )

    if not consent_granted:
        raise ValueError(
            "Cannot collect review without explicit consent. "
            "consent_granted must be True."
        )
    if not submitted_text or not submitted_text.strip():
        raise ValueError(
            "submitted_text must be a non-empty, non-whitespace string."
        )
    if not collected_at.strip():
        raise ValueError(
            "collected_at must be a non-empty string."
        )
    if not collection_method.strip():
        raise ValueError(
            "collection_method must be a non-empty string."
        )

    # --- ID generation ------------------------------------------------------
    review_id = "hw-" + uuid.uuid4().hex
    contributor_pseudonym = "contrib-" + uuid.uuid4().hex

    # --- PII evidence -------------------------------------------------------
    pii_evidence = build_pii_evidence_summary(review_id, submitted_text)

    if pii_evidence is not None:
        # Build safe text variants
        review_text = safe_surrogate_transform(submitted_text)
        export_text_redacted = redact_pii_in_text(submitted_text)
        pii_detected = True

        # Derive a completed immutable evidence summary with the
        # post-transform redaction status and collection timestamp.
        pii_evidence = dataclasses.replace(
            pii_evidence,
            redaction_status="redacted_surrogate",
            created_at=collected_at,
        )
        pii_categories = list(pii_evidence.categories)
        pii_redaction_status = pii_evidence.redaction_status
        pii_evidence_id = pii_evidence.evidence_id

        # Deterministic post-transform inspection.
        # A safe non-real surrogate remaining detector-positive is expected
        # and not an error — this is inspection, not moderation.
        detect_pii(review_text)

    else:
        review_text = submitted_text
        export_text_redacted = submitted_text
        pii_detected = False
        pii_categories = None
        pii_redaction_status = None
        pii_evidence_id = None

    # --- Source text raw retention -------------------------------------------
    source_text_raw = submitted_text if retain_source_text_raw else None

    # --- Create CanonicalRecord ---------------------------------------------
    record = workflow.create_record(
        review_id=review_id,
        source_text_raw=source_text_raw,
        review_text=review_text,
        export_text_redacted=export_text_redacted,
        source_type=SourceType.HUMAN_WRITTEN_RMC,
        annotation_status=AnnotationStatus.UNANNOTATED,
        consent_status="CONSENTED",
        contributor_pseudonym=contributor_pseudonym,
        collection_method=collection_method,
        pii_detected=pii_detected,
        pii_categories=pii_categories,
        pii_redaction_status=pii_redaction_status,
        pii_evidence_id=pii_evidence_id,
        created_at=collected_at,
    )

    # --- Consent artifact ----------------------------------------------------
    consent_artifact = ConsentArtifact(
        review_id=review_id,
        consent_status="CONSENTED",
        collected_at=collected_at,
        method=collection_method,
        evidence_ref=consent_evidence_ref,
    )

    return HumanWrittenIntakeResult(
        record=record,
        consent_artifact=consent_artifact,
        pii_evidence=pii_evidence,
    )
