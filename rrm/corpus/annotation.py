"""Annotation stores, disagreement detection, and adjudication helpers.

This module provides immutable stores for annotation submissions, controlled
protocol truths, record dispositions, and reannotation requirements.

It also provides:
- eight-dimension disagreement detection between two submissions
- adjudicator decision application

Architectural note:
    RRM UNDERSTANDS. TRUST DECIDES.

    These stores and helpers record evidence about annotation progress.
    They do NOT make moderation or trust decisions.

Public API:
    AnnotationSubmissionStore
    ControlledProtocolTruthStore
    DispositionRegister
    ReannotationRegister
    compute_eight_dimension_disagreement
    needs_adjudication
    adjudicator_decision
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple, Union

from rrm.corpus.models import (
    AnnotationStatus,
    AnnotationSubmission,
    CollegeCategory,
    ControlledProtocolTruth,
    Disposition,
    LanguageMix,
    RecordDisposition,
    ReannotationRequirement,
)


# ---------------------------------------------------------------------------
# AnnotationSubmissionStore
# ---------------------------------------------------------------------------

class AnnotationSubmissionStore:
    """Stores annotation submissions indexed by review_id.

    This store is mutable (submissions are added over time) but the
    stored AnnotationSubmission objects are themselves immutable.

    Parameters
    ----------
    None
    """

    def __init__(self) -> None:
        self._store: Dict[str, List[AnnotationSubmission]] = {}

    def submit(self, sub: AnnotationSubmission) -> None:
        """Record an annotation submission.

        Parameters
        ----------
        sub : AnnotationSubmission
            The submission to store.

        Raises
        ------
        TypeError
            If *sub* is not an AnnotationSubmission.
        """
        if not isinstance(sub, AnnotationSubmission):
            raise TypeError(
                f"submit expects AnnotationSubmission, "
                f"got {type(sub).__name__}"
            )
        rid = sub.review_id
        if rid not in self._store:
            self._store[rid] = []
        self._store[rid].append(sub)

    def get_for_review(
        self, review_id: str
    ) -> Tuple[AnnotationSubmission, ...]:
        """Return all submissions for a review, in submission order.

        Parameters
        ----------
        review_id : str
            The review identifier.

        Returns
        -------
        tuple[AnnotationSubmission, ...]
            All submissions for this review, or empty tuple.
        """
        return tuple(self._store.get(review_id, []))

    def get_a(self, review_id: str) -> Optional[AnnotationSubmission]:
        """Return the first submission (Annotator A) for a review.

        Parameters
        ----------
        review_id : str
            The review identifier.

        Returns
        -------
        AnnotationSubmission or None
        """
        subs = self._store.get(review_id, [])
        return subs[0] if subs else None

    def get_b(self, review_id: str) -> Optional[AnnotationSubmission]:
        """Return the second submission (Annotator B) for a review.

        Parameters
        ----------
        review_id : str
            The review identifier.

        Returns
        -------
        AnnotationSubmission or None
        """
        subs = self._store.get(review_id, [])
        return subs[1] if len(subs) > 1 else None

    def has_both(self, review_id: str) -> bool:
        """Return True when both Annotator A and B have submitted.

        Parameters
        ----------
        review_id : str
            The review identifier.
        """
        subs = self._store.get(review_id, [])
        return len(subs) >= 2


# ---------------------------------------------------------------------------
# ControlledProtocolTruthStore
# ---------------------------------------------------------------------------

class ControlledProtocolTruthStore:
    """Stores controlled ground truth indexed by review_id.

    Parameters
    ----------
    None
    """

    def __init__(self) -> None:
        self._store: Dict[str, ControlledProtocolTruth] = {}

    def set_truth(self, truth: ControlledProtocolTruth) -> None:
        """Store a controlled protocol truth.

        Parameters
        ----------
        truth : ControlledProtocolTruth
            The truth to store.

        Raises
        ------
        TypeError
            If *truth* is not a ControlledProtocolTruth.
        """
        if not isinstance(truth, ControlledProtocolTruth):
            raise TypeError(
                f"set_truth expects ControlledProtocolTruth, "
                f"got {type(truth).__name__}"
            )
        self._store[truth.review_id] = truth

    def get(self, review_id: str) -> Optional[ControlledProtocolTruth]:
        """Retrieve truth for a review.

        Parameters
        ----------
        review_id : str
            The review identifier.

        Returns
        -------
        ControlledProtocolTruth or None
        """
        return self._store.get(review_id)

    def get_target_value(self, review_id: str, label: str) -> Optional[int]:
        """Get the authoritative target value for a specific label.

        Parameters
        ----------
        review_id : str
            The review identifier.
        label : str
            Canonical label name.

        Returns
        -------
        int or None
            Authoritative value, or None if no truth exists.
        """
        truth = self._store.get(review_id)
        if truth is None:
            return None
        return truth.target_values.get(label)


# ---------------------------------------------------------------------------
# DispositionRegister
# ---------------------------------------------------------------------------

class DispositionRegister:
    """Stores record dispositions with reason codes.

    Parameters
    ----------
    None
    """

    def __init__(self) -> None:
        self._store: Dict[str, RecordDisposition] = {}

    def register(self, disp: RecordDisposition) -> None:
        """Record a disposition.

        Parameters
        ----------
        disp : RecordDisposition
            The disposition to register.

        Raises
        ------
        TypeError
            If *disp* is not a RecordDisposition.
        """
        if not isinstance(disp, RecordDisposition):
            raise TypeError(
                f"register expects RecordDisposition, "
                f"got {type(disp).__name__}"
            )
        self._store[disp.review_id] = disp

    def get(self, review_id: str) -> Optional[RecordDisposition]:
        """Retrieve disposition for a review.

        Parameters
        ----------
        review_id : str
            The review identifier.

        Returns
        -------
        RecordDisposition or None
        """
        return self._store.get(review_id)

    def is_excluded(self, review_id: str) -> bool:
        """Return True when the record is dispositioned as EXCLUDED.

        Parameters
        ----------
        review_id : str
            The review identifier.
        """
        disp = self._store.get(review_id)
        return disp is not None and disp.disposition == Disposition.EXCLUDED

    def is_held(self, review_id: str) -> bool:
        """Return True when the record is dispositioned as HOLD.

        Parameters
        ----------
        review_id : str
            The review identifier.
        """
        disp = self._store.get(review_id)
        return disp is not None and disp.disposition == Disposition.HOLD


# ---------------------------------------------------------------------------
# ReannotationRegister
# ---------------------------------------------------------------------------

class ReannotationRegister:
    """Stores reannotation requirements.

    Parameters
    ----------
    None
    """

    def __init__(self) -> None:
        self._store: Dict[str, ReannotationRequirement] = {}

    def require(self, req: ReannotationRequirement) -> None:
        """Record a reannotation requirement.

        Parameters
        ----------
        req : ReannotationRequirement
            The requirement to record.

        Raises
        ------
        TypeError
            If *req* is not a ReannotationRequirement.
        """
        if not isinstance(req, ReannotationRequirement):
            raise TypeError(
                f"require expects ReannotationRequirement, "
                f"got {type(req).__name__}"
            )
        self._store[req.review_id] = req

    def get_pending(self) -> Tuple[ReannotationRequirement, ...]:
        """Return all pending reannotation requirements.

        Returns
        -------
        tuple[ReannotationRequirement, ...]
        """
        return tuple(self._store.values())

    def has_active(self, review_id: str) -> bool:
        """Return True when the review has an unresolved reannotation requirement.

        Parameters
        ----------
        review_id : str
            The review identifier.

        Notes
        -----
        An unresolved requirement means the register has an entry for
        this review_id.  The entry is active until the workflow clears
        it after successful reannotation.
        """
        return review_id in self._store

    def get_for_review(
        self, review_id: str
    ) -> Optional[ReannotationRequirement]:
        """Return the reannotation requirement for a review.

        Parameters
        ----------
        review_id : str
            The review identifier.

        Returns
        -------
        ReannotationRequirement or None
        """
        return self._store.get(review_id)


# ---------------------------------------------------------------------------
# Operational context for Gate-D QC
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class GateCOperationalContext:
    """Internal operational context for Gate-D eligibility decisions.

    This object is NOT a CanonicalRecord field.  It is NOT portable.
    It is constructed fresh for each eligibility check from the current
    state of the operational stores.

    Parameters
    ----------
    submission_store : AnnotationSubmissionStore
        Active annotation submissions.
    controlled_truth_store : ControlledProtocolTruthStore
        Controlled protocol ground truths.
    disposition_register : DispositionRegister
        Record dispositions.
    reannotation_register : ReannotationRegister
        Reannotation requirements.
    known_review_ids : frozenset[str]
        All review IDs known to the corpus (for parent-link validation).
    inherited_target_evidence : dict[str, frozenset[str]]
        Mapping from review_id to the set of target names whose
        inheritance was explicitly reviewed and accepted as
        semantics-preserving.
    """

    submission_store: AnnotationSubmissionStore
    controlled_truth_store: ControlledProtocolTruthStore
    disposition_register: DispositionRegister
    reannotation_register: ReannotationRegister
    known_review_ids: frozenset
    inherited_target_evidence: Dict[str, frozenset]


# ---------------------------------------------------------------------------
# Eight-dimension disagreement
# ---------------------------------------------------------------------------

#: The eight annotation dimensions compared for disagreement.
_EIGHT_DIMENSIONS: Tuple[str, ...] = (
    "spam",
    "deception",
    "toxicity",
    "advertising",
    "off_topic",
    "pii",
    "language_mix",
    "college_category",
)


def compute_eight_dimension_disagreement(
    a: AnnotationSubmission,
    b: AnnotationSubmission,
) -> Dict[str, bool]:
    """Compare two submissions across the eight annotation dimensions.

    Parameters
    ----------
    a : AnnotationSubmission
        First annotator's submission.
    b : AnnotationSubmission
        Second annotator's submission.

    Returns
    -------
    dict[str, bool]
        Mapping of dimension name to True when annotators disagree on
        that dimension.

    Notes
    -----
    The eight dimensions are:
    spam, deception, toxicity, advertising, off_topic, pii,
    language_mix, college_category.
    """
    disagreement: Dict[str, bool] = {}

    # Numeric label dimensions
    for label in ("spam", "deception", "toxicity", "advertising", "off_topic", "pii"):
        a_val = getattr(a, label)
        b_val = getattr(b, label)
        disagreement[label] = a_val != b_val

    # Enum dimensions
    a_lang = a.language_mix.value if a.language_mix else None
    b_lang = b.language_mix.value if b.language_mix else None
    disagreement["language_mix"] = a_lang != b_lang

    a_cat = a.college_category.value if a.college_category else None
    b_cat = b.college_category.value if b.college_category else None
    disagreement["college_category"] = a_cat != b_cat

    return disagreement


def needs_adjudication(eight_dim: Dict[str, bool]) -> bool:
    """Return True if ANY of the eight dimensions disagrees.

    Parameters
    ----------
    eight_dim : dict[str, bool]
        Output from ``compute_eight_dimension_disagreement``.

    Returns
    -------
    bool
        True when adjudication is required.
    """
    return any(eight_dim.values())


# ---------------------------------------------------------------------------
# Adjudicator decision application
# ---------------------------------------------------------------------------

#: Dimensions that use int values (not enums) in adjudication.
_INT_DIMENSIONS: Tuple[str, ...] = (
    "spam",
    "deception",
    "toxicity",
    "advertising",
    "off_topic",
    "pii",
)

#: Dimensions that use enum values.
_ENUM_DIMENSIONS: Tuple[str, ...] = (
    "language_mix",
    "college_category",
)


def adjudicator_decision(
    a: AnnotationSubmission,
    b: AnnotationSubmission,
    adjudicator_choices: Dict[str, Union[int, str, None]],
) -> Dict[str, Union[int, str, None]]:
    """Apply adjudicator choices to resolve disagreements.

    Parameters
    ----------
    a : AnnotationSubmission
        First annotator's submission (used as fallback).
    b : AnnotationSubmission
        Second annotator's submission.
    adjudicator_choices : dict[str, int | str | None]
        Mapping of dimension name to adjudicator's choice value.
        Dimensions not present in this dict inherit from *a*.

    Returns
    -------
    dict[str, int | str | None]
        Resolved values for all eight dimensions.

    Raises
    ------
    TypeError
        If *adjudicator_choices* is not a dict.
    """
    if not isinstance(adjudicator_choices, dict):
        raise TypeError(
            f"adjudicator_choices must be a dict, "
            f"got {type(adjudicator_choices).__name__}"
        )

    resolved: Dict[str, Union[int, str, None]] = {}

    # Start with annotator A's values as defaults
    for dim in _EIGHT_DIMENSIONS:
        if dim in _INT_DIMENSIONS:
            resolved[dim] = getattr(a, dim)
        else:
            val = getattr(a, dim)
            resolved[dim] = val.value if val else None

    # Apply adjudicator choices
    for dim, choice in adjudicator_choices.items():
        if dim not in _EIGHT_DIMENSIONS:
            continue  # ignore unknown dimensions
        if choice is not None:
            resolved[dim] = choice

    return resolved
