"""Annotation workflow orchestration for the RMC corpus.

This module provides the CorpusWorkflow class that orchestrates the complete
annotation lifecycle:

    UNANNOTATED → ANNOTATING → ADJUDICATION_REQUIRED → FINAL
                                                      → EXCLUDED

Architectural note:
    RRM UNDERSTANDS. TRUST DECIDES.

    This module orchestrates data flow and state transitions.  It does NOT
    make moderation or trust decisions.

Public API:
    CorpusWorkflow
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional, Tuple, Union

from rrm.corpus.annotation import (
    AnnotationSubmissionStore,
    ControlledProtocolTruthStore,
    DispositionRegister,
    ReannotationRegister,
    compute_eight_dimension_disagreement,
    needs_adjudication,
)
from rrm.corpus.models import (
    ALL_REASON_CODES,
    AnnotationStatus,
    AnnotationSubmission,
    CanonicalRecord,
    CollegeCategory,
    ControlledProtocolTruth,
    Disposition,
    LanguageMix,
    PIIEvidenceSummary,
    RecordDisposition,
    ReannotationRequirement,
    SourceType,
)
from rrm.corpus.validation import gate_d_eligibility_qc


class CorpusWorkflow:
    """Orchestrates the annotation workflow for RMC records.

    Parameters
    ----------
    submissions : AnnotationSubmissionStore
        Store for annotation submissions.
    truths : ControlledProtocolTruthStore
        Store for controlled protocol truths.
    dispositions : DispositionRegister
        Register for record dispositions.
    reannotations : ReannotationRegister
        Register for reannotation requirements.

    Notes
    -----
    State machine:

    ::

        UNANNOTATED
            ↓  first submission
        ANNOTATING
            ↓  second submission agrees on all dimensions
        FINAL (via agreement)
            ↓
        EXCLUDED

        ANNOTATING
            ↓  second submission disagrees on any dimension
        ADJUDICATION_REQUIRED
            ↓  adjudicator resolves
        FINAL (via adjudication)
            ↓
        EXCLUDED

    All state transitions produce new CanonicalRecord instances.
    The workflow maintains an internal record store.
    """

    def __init__(
        self,
        submissions: AnnotationSubmissionStore,
        truths: ControlledProtocolTruthStore,
        dispositions: DispositionRegister,
        reannotations: ReannotationRegister,
    ) -> None:
        self._submissions = submissions
        self._truths = truths
        self._dispositions = dispositions
        self._reannotations = reannotations
        self._records: Dict[str, CanonicalRecord] = {}

    # ------------------------------------------------------------------
    # Record management
    # ------------------------------------------------------------------

    def register_record(self, record: CanonicalRecord) -> None:
        """Register a CanonicalRecord with the workflow.

        Parameters
        ----------
        record : CanonicalRecord
            The record to register.

        Raises
        ------
        TypeError
            If *record* is not a CanonicalRecord.
        """
        if not isinstance(record, CanonicalRecord):
            raise TypeError(
                f"register_record expects CanonicalRecord, "
                f"got {type(record).__name__}"
            )
        self._records[record.review_id] = record

    def get_record(self, review_id: str) -> Optional[CanonicalRecord]:
        """Retrieve a record by ID.

        Parameters
        ----------
        review_id : str
            The review identifier.

        Returns
        -------
        CanonicalRecord or None
        """
        return self._records.get(review_id)

    def _update_record(self, record: CanonicalRecord) -> None:
        """Replace a record in the internal store."""
        self._records[record.review_id] = record

    # ------------------------------------------------------------------
    # Record creation
    # ------------------------------------------------------------------

    def create_record(
        self,
        review_id: str,
        source_text_raw: str,
        source_type: SourceType,
        created_at: str,
        review_text: Optional[str] = None,
        export_text_redacted: Optional[str] = None,
        **kwargs,
    ) -> CanonicalRecord:
        """Create a new CanonicalRecord in UNANNOTATED status.

        Parameters
        ----------
        review_id : str
            Unique record identifier.
        source_text_raw : str
            Secure raw source text.  Never public, never neural input.
        source_type : SourceType
            Source class identifier.
        created_at : str
            ISO 8601 record creation timestamp.
        review_text : str, optional
            Canonical safe text for annotation and training.
        export_text_redacted : str, optional
            Export-safe text.
        **kwargs
            Additional fields to set on the record.

        Returns
        -------
        CanonicalRecord
            New record in UNANNOTATED status.

        Raises
        ------
        TypeError
            If required arguments have wrong types.
        """
        if not isinstance(review_id, str) or not review_id.strip():
            raise TypeError("review_id must be a non-empty string")
        if not isinstance(source_text_raw, str):
            raise TypeError(
                f"source_text_raw must be str, "
                f"got {type(source_text_raw).__name__}"
            )
        if not isinstance(source_type, SourceType):
            raise TypeError(
                f"source_type must be SourceType, "
                f"got {type(source_type).__name__}"
            )
        if not isinstance(created_at, str) or not created_at.strip():
            raise TypeError("created_at must be a non-empty string")

        record = CanonicalRecord(
            review_id=review_id,
            source_text_raw=source_text_raw,
            source_type=source_type,
            created_at=created_at,
            review_text=review_text,
            export_text_redacted=export_text_redacted,
            **kwargs,
        )
        self.register_record(record)
        return record

    # ------------------------------------------------------------------
    # Ephemeral raw text
    # ------------------------------------------------------------------

    def prepare_ephemeral_raw_text(
        self,
        record: CanonicalRecord,
    ) -> Optional[str]:
        """Return ephemeral raw text for annotation if available.

        Parameters
        ----------
        record : CanonicalRecord
            The record to prepare raw text for.

        Returns
        -------
        str or None
            The raw text when governance permits, None otherwise.

        Notes
        -----
        source_text_raw is available only where governance permits.
        When unavailable, annotators see review_text instead.
        """
        if record.source_text_raw is not None:
            return record.source_text_raw
        return None

    # ------------------------------------------------------------------
    # Annotation submissions
    # ------------------------------------------------------------------

    def submit_annotation_a(
        self,
        record: CanonicalRecord,
        submission: AnnotationSubmission,
    ) -> CanonicalRecord:
        """Record Annotator A's submission.

        Parameters
        ----------
        record : CanonicalRecord
            The record being annotated.
        submission : AnnotationSubmission
            Annotator A's submission.

        Returns
        -------
        CanonicalRecord
            Updated record with ANNOTATING status.

        Raises
        ------
        TypeError
            If arguments have wrong types.
        ValueError
            If record is already dispositioned as EXCLUDED.
        """
        if not isinstance(submission, AnnotationSubmission):
            raise TypeError(
                f"submit_annotation_a expects AnnotationSubmission, "
                f"got {type(submission).__name__}"
            )

        disp = self._dispositions.get(record.review_id)
        if disp is not None and disp.disposition == Disposition.EXCLUDED:
            raise ValueError(
                f"Cannot annotate excluded record {record.review_id}"
            )

        self._submissions.submit(submission)

        updated = dataclasses.replace(
            record,
            annotation_status=AnnotationStatus.ANNOTATING,
            annotator_A_id=submission.annotator_id,
            annotation_guide_version=submission.annotation_guide_version,
        )
        self._update_record(updated)
        return updated

    def submit_annotation_b(
        self,
        record: CanonicalRecord,
        submission: AnnotationSubmission,
    ) -> CanonicalRecord:
        """Record Annotator B's submission and check for disagreement.

        Parameters
        ----------
        record : CanonicalRecord
            The record being annotated.
        submission : AnnotationSubmission
            Annotator B's submission.

        Returns
        -------
        CanonicalRecord
            Updated record.  Status becomes ADJUDICATION_REQUIRED if
            any dimension disagrees, otherwise ANNOTATING (ready for
            finalization via agreement).

        Raises
        ------
        TypeError
            If arguments have wrong types.
        ValueError
            If record is already dispositioned as EXCLUDED.
        """
        if not isinstance(submission, AnnotationSubmission):
            raise TypeError(
                f"submit_annotation_b expects AnnotationSubmission, "
                f"got {type(submission).__name__}"
            )

        disp = self._dispositions.get(record.review_id)
        if disp is not None and disp.disposition == Disposition.EXCLUDED:
            raise ValueError(
                f"Cannot annotate excluded record {record.review_id}"
            )

        self._submissions.submit(submission)

        a_sub = self._submissions.get_a(record.review_id)
        if a_sub is None:
            raise ValueError(
                f"Annotator A has not submitted for {record.review_id}"
            )

        disagreement = compute_eight_dimension_disagreement(a_sub, submission)
        has_disagreement = needs_adjudication(disagreement)

        if has_disagreement:
            new_status = AnnotationStatus.ADJUDICATION_REQUIRED
        else:
            new_status = AnnotationStatus.ANNOTATING

        updated = dataclasses.replace(
            record,
            annotation_status=new_status,
            annotator_B_id=submission.annotator_id,
            language_mix=submission.language_mix or record.language_mix,
            college_category=(
                submission.college_category or record.college_category
            ),
        )
        self._update_record(updated)
        return updated

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def finalize_agreement(
        self,
        review_id: str,
        finalized_at: str,
    ) -> CanonicalRecord:
        """Finalize a record when both annotators agree.

        Parameters
        ----------
        review_id : str
            The review identifier.
        finalized_at : str
            ISO 8601 finalization timestamp.

        Returns
        -------
        CanonicalRecord
            Updated record with FINAL status.

        Raises
        ------
        ValueError
            If the record is not in ANNOTATING status, or if both
            annotators have not submitted, or if a disposition blocks.
        """
        record = self._records.get(review_id)
        if record is None:
            raise ValueError(f"Unknown record: {review_id}")

        if record.annotation_status != AnnotationStatus.ANNOTATING:
            raise ValueError(
                f"Cannot finalize via agreement: record status is "
                f"{record.annotation_status.value}, expected ANNOTATING"
            )

        if not self._submissions.has_both(review_id):
            raise ValueError(
                f"Cannot finalize via agreement: both annotators must "
                f"have submitted for {review_id}"
            )

        disp = self._dispositions.get(review_id)
        if disp is not None and disp.disposition == Disposition.HOLD:
            raise ValueError(
                f"Cannot finalize: record {review_id} is on HOLD"
            )

        # For Human-Written RMC: deception is ordinary annotator opinion
        # and cannot establish ground truth → must be -1
        extra_kwargs: Dict[str, Any] = {}
        if record.source_type == SourceType.HUMAN_WRITTEN_RMC:
            extra_kwargs["deception"] = -1

        # Carry the guide version from submissions if not already on record
        sub_a = self._submissions.get_a(review_id)
        sub_b = self._submissions.get_b(review_id)
        if (
            record.annotation_guide_version is None
            and sub_a is not None
            and sub_b is not None
        ):
            if sub_a.annotation_guide_version == sub_b.annotation_guide_version:
                extra_kwargs["annotation_guide_version"] = (
                    sub_a.annotation_guide_version
                )

        updated = dataclasses.replace(
            record,
            annotation_status=AnnotationStatus.FINAL,
            finalized_at=finalized_at,
            **extra_kwargs,
        )
        self._update_record(updated)
        return updated

    def finalize_adjudication(
        self,
        review_id: str,
        adjudicator_id: str,
        adjudicator_choices: Dict[str, Optional[Union[int, str]]],
        finalized_at: str,
    ) -> CanonicalRecord:
        """Finalize a record via adjudicator decision.

        Parameters
        ----------
        review_id : str
            The review identifier.
        adjudicator_id : str
            Adjudicator identifier (pseudonymous).
        adjudicator_choices : dict[str, int | str | None]
            Adjudicator's choices for the disagreed dimensions.
        finalized_at : str
            ISO 8601 finalization timestamp.

        Returns
        -------
        CanonicalRecord
            Updated record with FINAL status and adjudicator fields.

        Raises
        ------
        ValueError
            If the record is not in ADJUDICATION_REQUIRED status, or
            if a disposition blocks.
        """
        record = self._records.get(review_id)
        if record is None:
            raise ValueError(f"Unknown record: {review_id}")

        if record.annotation_status != AnnotationStatus.ADJUDICATION_REQUIRED:
            raise ValueError(
                f"Cannot finalize via adjudication: record status is "
                f"{record.annotation_status.value}, expected "
                f"ADJUDICATION_REQUIRED"
            )

        disp = self._dispositions.get(review_id)
        if disp is not None and disp.disposition == Disposition.HOLD:
            raise ValueError(
                f"Cannot finalize: record {review_id} is on HOLD"
            )

        updated = dataclasses.replace(
            record,
            annotation_status=AnnotationStatus.FINAL,
            adjudicator_id=adjudicator_id,
            finalized_at=finalized_at,
        )
        self._update_record(updated)
        return updated

    # ------------------------------------------------------------------
    # Exclusion
    # ------------------------------------------------------------------

    def exclude(
        self,
        review_id: str,
        reason: str,
        internal_note: Optional[str] = None,
        recorded_at: Optional[str] = None,
    ) -> CanonicalRecord:
        """Mark a record as EXCLUDED with a reason code.

        Parameters
        ----------
        review_id : str
            The review identifier.
        reason : str
            One of the 10 canonical reason codes.
        internal_note : str, optional
            Internal explanation (not exported).
        recorded_at : str, optional
            ISO 8601 timestamp.  Defaults to empty string.

        Returns
        -------
        CanonicalRecord
            Updated record with EXCLUDED status and disposition.

        Raises
        ------
        ValueError
            If *reason* is not one of the 10 canonical reason codes.
        """
        if reason not in ALL_REASON_CODES:
            raise ValueError(
                f"Invalid reason code '{reason}'. "
                f"Must be one of: {ALL_REASON_CODES}"
            )

        record = self._records.get(review_id)
        if record is None:
            raise ValueError(f"Unknown record: {review_id}")

        import datetime

        now = recorded_at or datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()

        disposition = RecordDisposition(
            review_id=review_id,
            disposition=Disposition.EXCLUDED,
            reason_code=reason,
            internal_note=internal_note,
            recorded_at=now,
        )
        self._dispositions.register(disposition)

        updated = dataclasses.replace(
            record,
            annotation_status=AnnotationStatus.EXCLUDED,
        )
        self._update_record(updated)
        return updated

    # ------------------------------------------------------------------
    # Workflow state query
    # ------------------------------------------------------------------

    def get_workflow_state(
        self, record: CanonicalRecord
    ) -> AnnotationStatus:
        """Return the current effective workflow state.

        Parameters
        ----------
        record : CanonicalRecord
            The record to query.

        Returns
        -------
        AnnotationStatus
            The effective workflow state.
        """
        # Disposition EXCLUDED overrides everything.
        disp = self._dispositions.get(record.review_id)
        if disp is not None and disp.disposition == Disposition.EXCLUDED:
            return AnnotationStatus.EXCLUDED

        # Return whatever is on the record.
        return record.annotation_status

    def get_gate_d_eligibility(
        self, record: CanonicalRecord
    ) -> Tuple[bool, list]:
        """Return Gate-D eligibility for a record.

        Parameters
        ----------
        record : CanonicalRecord
            The record to check.

        Returns
        -------
        tuple[bool, list[str]]
            (is_eligible, list_of_issues).
        """
        from rrm.corpus.annotation import GateCOperationalContext
        from rrm.corpus.validation import gate_d_eligibility_qc

        context = GateCOperationalContext(
            submission_store=self._submissions,
            controlled_truth_store=self._truths,
            disposition_register=self._dispositions,
            reannotation_register=self._reannotations,
            known_review_ids=frozenset(self._records.keys()),
            inherited_target_evidence={},
        )
        return gate_d_eligibility_qc(record, context)
