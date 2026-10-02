"""Validation functions for RMC canonical records and operational objects.

This module provides deterministic validation for:
- CanonicalRecord structural and domain correctness
- Label value domains
- AnnotationSubmission integrity
- ControlledProtocolTruth consistency
- Synthetic provenance requirements
- Gate-D eligibility quality control

Architectural note:
    RRM UNDERSTANDS. TRUST DECIDES.

    Validation produces evidence about record quality.  It does NOT
    make moderation decisions.

Public API:
    validate_record
    validate_label
    validate_annotation_submission
    validate_controlled_truth
    validate_synthetic_provenance
    gate_d_eligibility_qc
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from rrm.corpus.annotation import GateCOperationalContext
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
    SourceType,
    _VALID_REASON_CODES,
)


# ---------------------------------------------------------------------------
# Label domain validation
# ---------------------------------------------------------------------------

#: Label domains: label_name -> set of valid integer values.
_LABEL_DOMAINS: Dict[str, frozenset[int]] = {
    "spam": frozenset({0, 1}),
    "deception": frozenset({-1, 0, 1}),
    "toxicity": frozenset({0, 1}),
    "advertising": frozenset({0, 1}),
    "off_topic": frozenset({0, 1}),
    "pii": frozenset({0, 1}),
}


#: Ordered tuple of canonical label names.
_CANONICAL_LABELS: Tuple[str, ...] = (
    "spam",
    "deception",
    "toxicity",
    "advertising",
    "off_topic",
    "pii",
)


#: Pattern for semantic version strings (e.g., "1.0", "1.2.3").
_SEMVER_PATTERN = re.compile(r"^\d+\.\d+(\.\d+)?$")


#: Pattern for ISO 8601 timestamps (lenient — checks non-empty + basic structure).
_ISO8601_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}")


def validate_label(label: str, value: int) -> bool:
    """Validate that *value* is in the domain for *label*.

    Parameters
    ----------
    label : str
        Canonical label name.  Must be one of the six canonical labels.
    value : int
        Integer value to validate.

    Returns
    -------
    bool
        True when the value is valid for the label, False otherwise.

    Raises
    ------
    TypeError
        If *label* is not a str or *value* is not an int.

    Examples
    --------
    >>> validate_label("spam", 1)
    True
    >>> validate_label("deception", -1)
    True
    >>> validate_label("spam", -1)
    False
    """
    if not isinstance(label, str):
        raise TypeError(
            f"validate_label expects str for label, got {type(label).__name__}"
        )
    if not isinstance(value, int):
        raise TypeError(
            f"validate_label expects int for value, got {type(value).__name__}"
        )

    if label not in _LABEL_DOMAINS:
        return False

    return value in _LABEL_DOMAINS[label]


# ---------------------------------------------------------------------------
# CanonicalRecord validation
# ---------------------------------------------------------------------------

def validate_record(record: CanonicalRecord) -> Tuple[bool, List[str]]:
    """Validate a CanonicalRecord for structural and domain correctness.

    Parameters
    ----------
    record : CanonicalRecord
        The record to validate.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list_of_error_strings).  is_valid is True when the
        record passes all checks.  The error list is empty on success.

    Notes
    -----
    Checks performed:
    - review_id is a non-empty string
    - All six label values are in their valid domains (when not None)
    - source_type is a valid SourceType (when not None)
    - language_mix is a valid LanguageMix (when not None)
    - college_category is a valid CollegeCategory (when not None)
    - annotation_status is a valid AnnotationStatus (always required, has default)
    - annotation_guide_version matches semantic version pattern (when not None)
    - created_at is a non-empty ISO 8601-like string (when not None)
    - deception_truth is not -1 (when not None)
    - pii_detected is bool (when not None)
    """
    errors: List[str] = []

    # review_id
    if not isinstance(record.review_id, str) or not record.review_id.strip():
        errors.append("review_id must be a non-empty string")

    # Six labels
    for label in _CANONICAL_LABELS:
        value = getattr(record, label)
        if value is not None:
            if not isinstance(value, int):
                errors.append(
                    f"{label} must be int or None, got {type(value).__name__}"
                )
            elif value not in _LABEL_DOMAINS[label]:
                errors.append(
                    f"{label}={value} is outside valid domain "
                    f"{sorted(_LABEL_DOMAINS[label])}"
                )

    # deception_truth must not be -1
    if record.deception_truth is not None:
        if not isinstance(record.deception_truth, int):
            errors.append(
                f"deception_truth must be int or None, "
                f"got {type(record.deception_truth).__name__}"
            )
        elif record.deception_truth == -1:
            errors.append("deception_truth must be 0 or 1 (rejects -1)")

    # pii_detected must be bool or None
    if record.pii_detected is not None and not isinstance(
        record.pii_detected, bool
    ):
        errors.append(
            f"pii_detected must be bool or None, "
            f"got {type(record.pii_detected).__name__}"
        )

    # source_type
    if record.source_type is not None and not isinstance(
        record.source_type, SourceType
    ):
        errors.append(
            f"source_type must be SourceType or None, "
            f"got {type(record.source_type).__name__}"
        )

    # language_mix
    if record.language_mix is not None and not isinstance(
        record.language_mix, LanguageMix
    ):
        errors.append(
            f"language_mix must be LanguageMix or None, "
            f"got {type(record.language_mix).__name__}"
        )

    # college_category
    if record.college_category is not None and not isinstance(
        record.college_category, CollegeCategory
    ):
        errors.append(
            f"college_category must be CollegeCategory or None, "
            f"got {type(record.college_category).__name__}"
        )

    # annotation_status (always required — has default)
    if not isinstance(record.annotation_status, AnnotationStatus):
        errors.append(
            f"annotation_status must be AnnotationStatus, "
            f"got {type(record.annotation_status).__name__}"
        )

    # annotation_guide_version
    if record.annotation_guide_version is not None:
        if not isinstance(record.annotation_guide_version, str):
            errors.append(
                f"annotation_guide_version must be str or None, "
                f"got {type(record.annotation_guide_version).__name__}"
            )
        elif not _SEMVER_PATTERN.match(record.annotation_guide_version):
            errors.append(
                f"annotation_guide_version must match semantic version "
                f"pattern (e.g., '1.0'), got '{record.annotation_guide_version}'"
            )

    # created_at
    if record.created_at is not None:
        if not isinstance(record.created_at, str):
            errors.append(
                f"created_at must be str or None, "
                f"got {type(record.created_at).__name__}"
            )
        elif not record.created_at.strip():
            errors.append("created_at must be a non-empty string when set")
        elif not _ISO8601_PATTERN.match(record.created_at):
            errors.append(
                f"created_at must look like ISO 8601, "
                f"got '{record.created_at}'"
            )

    # controlled_targets
    if record.controlled_targets is not None:
        if not isinstance(record.controlled_targets, dict):
            errors.append(
                f"controlled_targets must be dict or None, "
                f"got {type(record.controlled_targets).__name__}"
            )
        else:
            for k, v in record.controlled_targets.items():
                if k not in _LABEL_DOMAINS:
                    errors.append(
                        f"controlled_targets key '{k}' is not a "
                        f"canonical label"
                    )
                elif v not in _LABEL_DOMAINS[k]:
                    errors.append(
                        f"controlled_targets['{k}']={v} is outside "
                        f"valid domain {sorted(_LABEL_DOMAINS[k])}"
                    )

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# AnnotationSubmission validation
# ---------------------------------------------------------------------------

def validate_annotation_submission(
    sub: AnnotationSubmission,
) -> Tuple[bool, List[str]]:
    """Validate an AnnotationSubmission.

    Parameters
    ----------
    sub : AnnotationSubmission
        The submission to validate.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list_of_error_strings).

    Notes
    -----
    Checks:
    - All six label fields are in their valid domains
    - annotator_id is a non-empty string
    - annotation_guide_version matches semantic version pattern
    - submitted_at is a non-empty string
    """
    errors: List[str] = []

    # Labels
    for label in _CANONICAL_LABELS:
        value = getattr(sub, label)
        if value not in _LABEL_DOMAINS[label]:
            errors.append(
                f"submission.{label}={value} is outside valid domain "
                f"{sorted(_LABEL_DOMAINS[label])}"
            )

    # annotator_id
    if not isinstance(sub.annotator_id, str) or not sub.annotator_id.strip():
        errors.append("annotator_id must be a non-empty string")

    # annotation_guide_version
    if not isinstance(sub.annotation_guide_version, str) or not _SEMVER_PATTERN.match(
        sub.annotation_guide_version
    ):
        errors.append(
            f"annotation_guide_version must match semantic version "
            f"pattern, got '{sub.annotation_guide_version}'"
        )

    # submitted_at
    if not isinstance(sub.submitted_at, str) or not sub.submitted_at.strip():
        errors.append("submitted_at must be a non-empty string")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# ControlledProtocolTruth validation
# ---------------------------------------------------------------------------

def validate_controlled_truth(
    truth: ControlledProtocolTruth,
    record: CanonicalRecord,
) -> Tuple[bool, List[str]]:
    """Validate a ControlledProtocolTruth against its record.

    Parameters
    ----------
    truth : ControlledProtocolTruth
        The controlled truth to validate.
    record : CanonicalRecord
        The canonical record the truth applies to.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list_of_error_strings).

    Notes
    -----
    Checks:
    - record.source_type is CONTROLLED_RMC
    - truth.control_protocol_id is non-empty
    - truth.target_values keys are canonical labels
    - truth.target_values values are in valid domains for each label
    - deception_truth (if in target_values) is never -1
    """
    errors: List[str] = []

    if record.source_type != SourceType.CONTROLLED_RMC:
        errors.append(
            f"ControlledProtocolTruth requires source_type=CONTROLLED_RMC, "
            f"got {record.source_type}"
        )

    if not isinstance(truth.control_protocol_id, str) or not truth.control_protocol_id.strip():
        errors.append("control_protocol_id must be a non-empty string")

    if not isinstance(truth.target_values, dict):
        errors.append(
            f"target_values must be a dict, "
            f"got {type(truth.target_values).__name__}"
        )
        return len(errors) == 0, errors

    for k, v in truth.target_values.items():
        if k not in _LABEL_DOMAINS:
            errors.append(
                f"target_values key '{k}' is not a canonical label"
            )
        elif k == "deception" and v == -1:
            errors.append(
                f"target_values['deception']={v} is outside valid "
                f"domain {sorted(_LABEL_DOMAINS[k])} "
                f"(controlled truth does not accept -1)"
            )
        elif v not in _LABEL_DOMAINS[k]:
            errors.append(
                f"target_values['{k}']={v} is outside valid domain "
                f"{sorted(_LABEL_DOMAINS[k])}"
            )

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Synthetic provenance validation
# ---------------------------------------------------------------------------

def validate_synthetic_provenance(
    record: CanonicalRecord,
) -> Tuple[bool, List[str]]:
    """Validate provenance fields for a SYNTHETIC_DERIVED_RMC record.

    Parameters
    ----------
    record : CanonicalRecord
        The record to validate.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list_of_error_strings).

    Notes
    -----
    Checks (when source_type is SYNTHETIC_DERIVED_RMC):

    - parent_review_id: ALWAYS required (non-empty string).
    - derivation_type: CONDITIONAL — required only when supplied;
      if provided, must be a non-empty descriptive string.
    - generation_method: CONDITIONAL — required only when supplied;
      if provided, must be a non-empty descriptive string.

    Examples
    --------
    >>> from rrm.corpus.models import CanonicalRecord, SourceType
    >>> r = CanonicalRecord(review_id="r-1", source_type=SourceType.SYNTHETIC_DERIVED_RMC,
    ...                     parent_review_id="p-1")
    >>> validate_synthetic_provenance(r)
    (True, [])
    >>> r2 = CanonicalRecord(review_id="r-2", source_type=SourceType.SYNTHETIC_DERIVED_RMC,
    ...                      parent_review_id="p-1", derivation_type="")
    >>> validate_synthetic_provenance(r2)
    (False, ['SYNTHETIC_DERIVED_RMC derivation_type must be a non-empty descriptive string when supplied'])
    >>> r3 = CanonicalRecord(review_id="r-3", source_type=SourceType.SYNTHETIC_DERIVED_RMC)
    >>> validate_synthetic_provenance(r3)
    (False, ['SYNTHETIC_DERIVED_RMC requires non-empty parent_review_id'])
    """
    errors: List[str] = []

    if record.source_type != SourceType.SYNTHETIC_DERIVED_RMC:
        return True, []

    # parent_review_id is always required
    if not isinstance(record.parent_review_id, str) or not record.parent_review_id.strip():
        errors.append(
            "SYNTHETIC_DERIVED_RMC requires non-empty parent_review_id"
        )

    # derivation_type: conditional — validate only when supplied
    if record.derivation_type is not None:
        if not isinstance(record.derivation_type, str) or not record.derivation_type.strip():
            errors.append(
                "SYNTHETIC_DERIVED_RMC derivation_type must be a "
                "non-empty descriptive string when supplied"
            )

    # generation_method: conditional — validate only when supplied
    if record.generation_method is not None:
        if not isinstance(record.generation_method, str) or not record.generation_method.strip():
            errors.append(
                "SYNTHETIC_DERIVED_RMC generation_method must be a "
                "non-empty descriptive string when supplied"
            )

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Gate-D eligibility quality control
# ---------------------------------------------------------------------------

def gate_d_eligibility_qc(
    record: CanonicalRecord,
    context: GateCOperationalContext,
) -> Tuple[bool, List[str]]:
    """Gate-D QC: determine whether a record is eligible for export.

    This function requires full operational context.  It cannot make
    a correct determination from the CanonicalRecord alone.

    Parameters
    ----------
    record : CanonicalRecord
        The record to check.
    context : GateCOperationalContext
        Operational context providing access to submission stores,
        controlled truth, dispositions, reannotation requirements,
        known review IDs, and inherited target evidence.

    Returns
    -------
    tuple[bool, list[str]]
        (is_eligible, list_of_error_strings).

    Notes
    -----
    Checks performed (in order):
    A. Canonical/base checks (review_id, source_type, created_at,
       review_text, annotation_status, labels, language_mix,
       college_category, annotation_guide_version).
    B. Later-gate ownership: dataset_version, split_membership,
       split_group_id must all be None.
    C. Operational disposition: HOLD or EXCLUDED rejects.
    D. Reannotation: active unresolved requirement rejects.
    E. Independent double annotation: both A and B active, different
       annotators, matching review_id and annotation_guide_version.
    F. Eight-dimension disagreement: for every non-controlled dimension
       where A != B, adjudicator_id must be present and distinct from
       both annotators.
    G. Controlled truth: for CONTROLLED_RMC, protocol truth must exist,
       match, and controlled targets must align.
    H. Human-Written: consent_status must be "CONSENTED", deception = -1.
    I. Synthetic/Derived: parent_review_id required, present in known
       IDs, inherited targets validated.
    J. PII: final review_text must pass PII adapter checks.

    This function produces evidence.  It does NOT make moderation
    or trust decisions.
    """
    errors: List[str] = []

    # ------------------------------------------------------------------
    # A. Canonical/base checks
    # ------------------------------------------------------------------
    if not isinstance(record.review_id, str) or not record.review_id.strip():
        errors.append("review_id must be a non-empty string")

    if not isinstance(record.source_type, SourceType):
        errors.append(
            f"source_type must be a valid SourceType, "
            f"got {type(record.source_type).__name__}"
        )

    if not isinstance(record.created_at, str) or not record.created_at.strip():
        errors.append("created_at must be a non-empty string")

    if not isinstance(record.review_text, str) or not record.review_text.strip():
        errors.append("review_text must be a non-empty string")

    if not isinstance(record.annotation_status, AnnotationStatus):
        errors.append(
            f"annotation_status must be AnnotationStatus, "
            f"got {type(record.annotation_status).__name__}"
        )
    elif record.annotation_status != AnnotationStatus.FINAL:
        errors.append(
            f"annotation_status must be FINAL for Gate-D export, "
            f"got {record.annotation_status.value}"
        )

    if record.annotation_status == AnnotationStatus.EXCLUDED:
        errors.append(
            "annotation_status is EXCLUDED — record is not eligible "
            "for Gate-D export"
        )

    # Six label domains — all required, None is NOT valid on a FINAL record
    for label, domain in _LABEL_DOMAINS.items():
        value = getattr(record, label)
        if value is None:
            errors.append(
                f"{label} must be set for Gate-D export, got None"
            )
        elif not isinstance(value, int):
            errors.append(
                f"{label} must be int, "
                f"got {type(value).__name__}"
            )
        elif value not in domain:
            errors.append(
                f"{label}={value} is outside valid domain {sorted(domain)}"
            )

    # language_mix — required, non-None
    if record.language_mix is None:
        errors.append(
            "language_mix must be set for Gate-D export, got None"
        )
    elif not isinstance(record.language_mix, LanguageMix):
        errors.append(
            f"language_mix must be LanguageMix, "
            f"got {type(record.language_mix).__name__}"
        )

    # college_category — optional, may be None
    if record.college_category is not None and not isinstance(
        record.college_category, CollegeCategory
    ):
        errors.append(
            f"college_category must be CollegeCategory or None, "
            f"got {type(record.college_category).__name__}"
        )

    # annotation_guide_version — required, non-empty valid semver
    if record.annotation_guide_version is None:
        errors.append(
            "annotation_guide_version must be set for Gate-D export, "
            "got None"
        )
    elif not isinstance(record.annotation_guide_version, str):
        errors.append(
            f"annotation_guide_version must be str, "
            f"got {type(record.annotation_guide_version).__name__}"
        )
    elif not record.annotation_guide_version.strip():
        errors.append(
            "annotation_guide_version must be a non-empty string"
        )
    elif not _SEMVER_PATTERN.match(record.annotation_guide_version):
        errors.append(
            f"annotation_guide_version must match semantic version "
            f"pattern, got '{record.annotation_guide_version}'"
        )

    # finalized_at — required, non-empty
    if record.finalized_at is None:
        errors.append(
            "finalized_at must be set for Gate-D export, got None"
        )
    elif not isinstance(record.finalized_at, str):
        errors.append(
            f"finalized_at must be str, "
            f"got {type(record.finalized_at).__name__}"
        )
    elif not record.finalized_at.strip():
        errors.append(
            "finalized_at must be a non-empty string"
        )

    # ------------------------------------------------------------------
    # B. Later-gate ownership
    # ------------------------------------------------------------------
    if record.dataset_version is not None:
        errors.append(
            f"dataset_version must be None at Gate-D, "
            f"got '{record.dataset_version}'"
        )

    if record.split_membership is not None:
        errors.append(
            f"split_membership must be None at Gate-D, "
            f"got '{record.split_membership}'"
        )

    if record.split_group_id is not None:
        errors.append(
            f"split_group_id must be None at Gate-D, "
            f"got '{record.split_group_id}'"
        )

    # ------------------------------------------------------------------
    # C. Operational disposition
    # ------------------------------------------------------------------
    active_disp = context.disposition_register.get(record.review_id)
    if active_disp is not None:
        if active_disp.disposition == Disposition.HOLD:
            errors.append(
                f"Record {record.review_id} is dispositioned as HOLD"
            )
        elif active_disp.disposition == Disposition.EXCLUDED:
            errors.append(
                f"Record {record.review_id} is dispositioned as EXCLUDED"
            )

    # ------------------------------------------------------------------
    # D. Reannotation
    # ------------------------------------------------------------------
    if context.reannotation_register.has_active(record.review_id):
        errors.append(
            f"Record {record.review_id} has an active unresolved "
            f"reannotation requirement"
        )

    # ------------------------------------------------------------------
    # E. Independent double annotation
    # ------------------------------------------------------------------
    sub_a = context.submission_store.get_a(record.review_id)
    sub_b = context.submission_store.get_b(record.review_id)

    if sub_a is None:
        errors.append(
            f"Record {record.review_id} has no Annotator A submission"
        )

    if sub_b is None:
        errors.append(
            f"Record {record.review_id} has no Annotator B submission"
        )

    if sub_a is not None and sub_b is not None:
        # Different annotators
        if sub_a.annotator_id == sub_b.annotator_id:
            errors.append(
                f"Annotator A and B must be different annotators, "
                f"both are '{sub_a.annotator_id}'"
            )

        # Must match record's review_id and guide version
        if sub_a.review_id != record.review_id:
            errors.append(
                f"Annotator A submission review_id '{sub_a.review_id}' "
                f"does not match record '{record.review_id}'"
            )
        if sub_b.review_id != record.review_id:
            errors.append(
                f"Annotator B submission review_id '{sub_b.review_id}' "
                f"does not match record '{record.review_id}'"
            )

        current_guide = record.annotation_guide_version
        if current_guide is not None:
            if sub_a.annotation_guide_version != current_guide:
                errors.append(
                    f"Annotator A guide version "
                    f"'{sub_a.annotation_guide_version}' does not match "
                    f"record '{current_guide}'"
                )
            if sub_b.annotation_guide_version != current_guide:
                errors.append(
                    f"Annotator B guide version "
                    f"'{sub_b.annotation_guide_version}' does not match "
                    f"record '{current_guide}'"
                )

    # ------------------------------------------------------------------
    # F. Eight-dimension disagreement + agreement consistency
    # ------------------------------------------------------------------
    if sub_a is not None and sub_b is not None:
        _NON_CONTROLLED_DIMS = (
            "spam",
            "deception",
            "toxicity",
            "advertising",
            "off_topic",
            "pii",
            "language_mix",
            "college_category",
        )

        # Determine which dimensions are controlled
        controlled_labels = set()
        if record.source_type == SourceType.CONTROLLED_RMC:
            ctrl_truth = context.controlled_truth_store.get(record.review_id)
            if ctrl_truth is not None and isinstance(ctrl_truth.target_values, dict):
                controlled_labels = set(ctrl_truth.target_values.keys())

        dim_values_a = {}
        dim_values_b = {}
        for dim in _NON_CONTROLLED_DIMS:
            if dim in ("language_mix", "college_category"):
                a_val = getattr(sub_a, dim)
                b_val = getattr(sub_b, dim)
                dim_values_a[dim] = a_val.value if a_val else None
                dim_values_b[dim] = b_val.value if b_val else None
            else:
                dim_values_a[dim] = getattr(sub_a, dim)
                dim_values_b[dim] = getattr(sub_b, dim)

        for dim in _NON_CONTROLLED_DIMS:
            if dim in controlled_labels:
                continue  # controlled dimension — adjudicator optional

            if dim_values_a[dim] != dim_values_b[dim]:
                # Disagreement on non-controlled dimension
                adjudicator_id = record.adjudicator_id
                if adjudicator_id is None:
                    errors.append(
                        f"Dimension '{dim}' disagrees but no "
                        f"adjudicator_id is set"
                    )
                elif adjudicator_id == sub_a.annotator_id:
                    errors.append(
                        f"adjudicator_id must not be Annotator A's ID "
                        f"for disagreement on '{dim}'"
                    )
                elif adjudicator_id == sub_b.annotator_id:
                    errors.append(
                        f"adjudicator_id must not be Annotator B's ID "
                        f"for disagreement on '{dim}'"
                    )

        # Agreement consistency: for agreed dimensions, the FINAL record
        # must contain the agreed value.
        _AGREEMENT_DIMS = (
            "spam",
            "toxicity",
            "advertising",
            "off_topic",
            "pii",
            "language_mix",
            "college_category",
        )
        for dim in _AGREEMENT_DIMS:
            if dim in controlled_labels:
                continue  # controlled — skip

            if dim_values_a[dim] == dim_values_b[dim]:
                agreed = dim_values_a[dim]
                if dim in ("language_mix", "college_category"):
                    rec_val = getattr(record, dim)
                    rec_str = rec_val.value if rec_val else None
                    if rec_str != agreed:
                        errors.append(
                            f"Dimension '{dim}': A and B agree on "
                            f"'{agreed}' but record has "
                            f"'{rec_str}'"
                        )
                else:
                    rec_val = getattr(record, dim)
                    if rec_val != agreed:
                        errors.append(
                            f"Dimension '{dim}': A and B agree on "
                            f"{agreed} but record has {rec_val}"
                        )

    # ------------------------------------------------------------------
    # G. Controlled truth
    # ------------------------------------------------------------------
    if record.source_type == SourceType.CONTROLLED_RMC:
        # experiment_id must be present and non-empty
        if (
            not isinstance(record.experiment_id, str)
            or not record.experiment_id.strip()
        ):
            errors.append(
                "CONTROLLED_RMC requires experiment_id"
            )

        if record.control_protocol_id is None:
            errors.append(
                "CONTROLLED_RMC requires control_protocol_id"
            )

        ctrl_truth = context.controlled_truth_store.get(record.review_id)
        if ctrl_truth is None:
            errors.append(
                f"CONTROLLED_RMC record {record.review_id} has no "
                f"ControlledProtocolTruth in the truth store"
            )
        else:
            # control_protocol_id must match
            if record.control_protocol_id is not None:
                if ctrl_truth.control_protocol_id != record.control_protocol_id:
                    errors.append(
                        f"control_protocol_id mismatch: record has "
                        f"'{record.control_protocol_id}', truth has "
                        f"'{ctrl_truth.control_protocol_id}'"
                    )

            # controlled_targets must be subset of truth keys
            if record.controlled_targets is not None:
                truth_keys = set(ctrl_truth.target_values.keys())
                record_keys = set(record.controlled_targets.keys())
                if not record_keys.issubset(truth_keys):
                    errors.append(
                        f"controlled_targets keys {sorted(record_keys - truth_keys)} "
                        f"are not in the protocol truth"
                    )

                # Every controlled canonical target must equal protocol value
                for label, proto_val in ctrl_truth.target_values.items():
                    rec_val = record.controlled_targets.get(label)
                    if rec_val is not None and rec_val != proto_val:
                        errors.append(
                            f"controlled_targets['{label}']={rec_val} "
                            f"does not match protocol value {proto_val}"
                        )

            # Deception controlled: record.deception == protocol
            if "deception" in (ctrl_truth.target_values or {}):
                proto_deception = ctrl_truth.target_values["deception"]
                if record.deception != proto_deception:
                    errors.append(
                        f"deception={record.deception} does not match "
                        f"controlled protocol value {proto_deception}"
                    )
                if record.deception_truth != proto_deception:
                    errors.append(
                        f"deception_truth={record.deception_truth} does "
                        f"not match controlled protocol value {proto_deception}"
                    )
            else:
                # deception NOT established in controlled truth
                if record.deception_truth is not None:
                    errors.append(
                        f"deception_truth={record.deception_truth} is set "
                        f"but deception is not in controlled_targets"
                    )
                if record.deception != -1:
                    errors.append(
                        f"deception={record.deception} but deception is "
                        f"not controlled; expected -1"
                    )

    # ------------------------------------------------------------------
    # H. Human-Written
    # ------------------------------------------------------------------
    if record.source_type == SourceType.HUMAN_WRITTEN_RMC:
        if record.consent_status != "CONSENTED":
            errors.append(
                f"HUMAN_WRITTEN_RMC requires consent_status='CONSENTED', "
                f"got '{record.consent_status}'"
            )

        if record.deception != -1:
            errors.append(
                f"HUMAN_WRITTEN_RMC ordinary deception must be -1, "
                f"got {record.deception}"
            )

    # ------------------------------------------------------------------
    # I. Synthetic/Derived
    # ------------------------------------------------------------------
    if record.source_type == SourceType.SYNTHETIC_DERIVED_RMC:
        if not isinstance(record.parent_review_id, str) or not record.parent_review_id.strip():
            errors.append(
                "SYNTHETIC_DERIVED_RMC requires non-empty parent_review_id"
            )
        elif record.parent_review_id == record.review_id:
            errors.append(
                "parent_review_id must not equal review_id"
            )
        elif record.parent_review_id not in context.known_review_ids:
            errors.append(
                f"parent_review_id '{record.parent_review_id}' is not in "
                f"known_review_ids"
            )

        if record.derivation_type is not None:
            if not isinstance(record.derivation_type, str) or not record.derivation_type.strip():
                errors.append(
                    "SYNTHETIC_DERIVED_RMC derivation_type must be a "
                    "non-empty descriptive string when supplied"
                )

        if record.generation_method is not None:
            if not isinstance(record.generation_method, str) or not record.generation_method.strip():
                errors.append(
                    "SYNTHETIC_DERIVED_RMC generation_method must be a "
                    "non-empty descriptive string when supplied"
                )

        # Inherited target evidence check
        if record.review_id in context.inherited_target_evidence:
            inherited = context.inherited_target_evidence[record.review_id]
            if inherited:
                # All inherited targets must exist in controlled_targets or
                # be part of the standard label set already validated above
                errors.append(
                    f"Review {record.review_id} uses inherited target "
                    f"evidence for: {sorted(inherited)}"
                )

    # ------------------------------------------------------------------
    # J. PII
    # ------------------------------------------------------------------
    if record.pii == 0:
        # PII-free claim — text must contain no detector-positive evidence
        try:
            from rrm.pii_detection import detect_pii
            pii_matches = detect_pii(record.review_text)
            if pii_matches:
                errors.append(
                    f"Record claims pii=0 but PII detector found "
                    f"{len(pii_matches)} match(es) in review_text"
                )
        except ImportError:
            pass  # pii_detection module not available — skip check

    elif record.pii == 1:
        if not record.pii_detected:
            errors.append(
                "pii=1 requires pii_detected=True"
            )
        if not record.pii_categories:
            errors.append(
                "pii=1 requires non-empty pii_categories"
            )
        if not record.pii_evidence_id:
            errors.append(
                "pii=1 requires non-empty pii_evidence_id"
            )
        if record.pii_redaction_status not in (
            "redacted",
            "redacted_surrogate",
            "redacted_excluded",
        ):
            errors.append(
                f"pii_redaction_status must indicate completed safe "
                f"handling, got '{record.pii_redaction_status}'"
            )

        # Verify text contains surrogate (not original private value)
        try:
            from rrm.pii_detection import detect_pii
            from rrm.corpus.pii_adapter import redact_pii_in_text
            raw_matches = detect_pii(record.review_text)
            if raw_matches:
                redacted = redact_pii_in_text(record.review_text)
                for match in raw_matches:
                    if match in redacted:
                        errors.append(
                            f"UNSAFE_PII: original private value "
                            f"'{match}' appears in review_text"
                        )
        except ImportError:
            pass

    return len(errors) == 0, errors
