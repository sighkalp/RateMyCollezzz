"""Export functionality for the RMC corpus.

This module provides Gate-D QC-gated export of CanonicalRecord objects
to JSON and JSONL formats.

Architectural note:
    RRM UNDERSTANDS. TRUST DECIDES.

    Export is a data-formatting operation.  It does NOT make policy
    decisions about what to include or exclude — the whitelist defines
    what may be exported.

Public API:
    EXPORT_WHITELIST
    export_corpus
    export_to_jsonl
    export_metadata
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Export whitelist
# ---------------------------------------------------------------------------

#: Canonical field names eligible for portable export at Gate D.
#:
#: Exactly 32 fields.  Immutable tuple — do not mutate.
#:
#: Excluded from portable export (not in this list):
#: - source_text_raw: secure/restricted (governance required)
#: - contributor_pseudonym: privacy-sensitive
#: - annotator_A_id / annotator_B_id / adjudicator_id: annotator identity
#: - annotation_notes: unstructured annotator commentary
PORTABLE_CANDIDATE_FIELDS: Tuple[str, ...] = (
    "review_id",
    "review_text",
    "export_text_redacted",
    "source_type",
    "spam",
    "deception",
    "toxicity",
    "advertising",
    "off_topic",
    "pii",
    "language_mix",
    "college_category",
    "annotation_status",
    "annotation_guide_version",
    "finalized_at",
    "consent_status",
    "collection_method",
    "experiment_id",
    "controlled_targets",
    "deception_truth",
    "control_protocol_id",
    "parent_review_id",
    "derivation_type",
    "generation_method",
    "pii_detected",
    "pii_categories",
    "pii_redaction_status",
    "pii_evidence_id",
    "created_at",
    "dataset_version",
    "split_membership",
    "split_group_id",
)

#: Legacy alias for PORTABLE_CANDIDATE_FIELDS.
EXPORT_WHITELIST: Tuple[str, ...] = PORTABLE_CANDIDATE_FIELDS


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize_value(value: Any) -> Any:
    """Convert a CanonicalRecord field value to a JSON-serializable form."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    # Enum types — serialize to their string value
    if hasattr(value, "value"):
        return value.value
    # Fallback: use string representation
    return str(value)


def _record_to_dict(record: Any) -> Dict[str, Any]:
    """Serialize a CanonicalRecord to a dict using PORTABLE_CANDIDATE_FIELDS.

    All 32 whitelisted fields are included in the output, even when
    the value is None.  Fields not in PORTABLE_CANDIDATE_FIELDS are
    never included.
    """
    result: Dict[str, Any] = {}
    for field_name in PORTABLE_CANDIDATE_FIELDS:
        if hasattr(record, field_name):
            raw_value = getattr(record, field_name)
            result[field_name] = _serialize_value(raw_value)
        else:
            result[field_name] = None
    return result


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def _check_gate_d(
    record: Any,
    context: Any,
) -> None:
    """Run Gate-D QC and raise if the record is not eligible.

    Parameters
    ----------
    record : CanonicalRecord
        The record to check.
    context : GateCOperationalContext
        Operational context for Gate-D eligibility.

    Raises
    ------
    ValueError
        If the record fails any Gate-D eligibility check.
    """
    from rrm.corpus.validation import gate_d_eligibility_qc

    is_eligible, errors = gate_d_eligibility_qc(record, context)
    if not is_eligible:
        raise ValueError(
            f"Gate-D eligibility failure for {record.review_id}: "
            + "; ".join(errors)
        )


def to_portable_candidate(
    record: Any,
    context: Any,
) -> Dict[str, Any]:
    """Build a portable candidate dict for a Gate-D-eligible record.

    Parameters
    ----------
    record : CanonicalRecord
        The record to export.
    context : GateCOperationalContext
        Operational context for Gate-D eligibility.

    Returns
    -------
    dict[str, any]
        Serialized portable record dict.

    Raises
    ------
    ValueError
        If the record fails Gate-D eligibility.
    """
    _check_gate_d(record, context)
    return _record_to_dict(record)


def export_corpus(
    records: Iterable[Any],
    context: Any,
) -> List[Dict[str, Any]]:
    """Export records as a list of portable dicts.

    Portable export always runs Gate-D eligibility QC with full
    operational context.  If ANY record is invalid, a ValueError
    is raised identifying the review_id and all validation issues.

    Parameters
    ----------
    records : iterable
        Iterable of CanonicalRecord objects.
    context : GateCOperationalContext
        Operational context for Gate-D eligibility.

    Returns
    -------
    list[dict[str, any]]
        List of serialized portable record dicts.

    Raises
    ------
    ValueError
        If any record fails Gate-D eligibility.

    Notes
    -----
    There is no QC bypass parameter.  Portable export always requires
    FINAL annotation_status and passing Gate-D eligibility.

    Callers who need to know which records were excluded should call
    gate_d_eligibility_qc directly.
    """
    result: List[Dict[str, Any]] = []
    failures: List[str] = []

    for record in records:
        try:
            _check_gate_d(record, context)
            result.append(_record_to_dict(record))
        except ValueError as exc:
            failures.append(str(exc))

    if failures:
        raise ValueError(
            f"Gate-D export rejected {len(failures)} record(s):\n"
            + "\n".join(failures)
        )

    return result


def export_to_jsonl(
    records: Iterable[Any],
    filepath: str,
    context: Any,
) -> int:
    """Export records to a JSONL file.

    Parameters
    ----------
    records : iterable
        Iterable of CanonicalRecord objects.
    filepath : str
        Path to the output JSONL file.
    context : GateCOperationalContext
        Operational context for Gate-D eligibility.

    Returns
    -------
    int
        Number of records written.

    Raises
    ------
    TypeError
        If *filepath* is not a str.
    ValueError
        If any record fails Gate-D eligibility.
    """
    if not isinstance(filepath, str):
        raise TypeError(
            f"export_to_jsonl expects str for filepath, "
            f"got {type(filepath).__name__}"
        )

    exported = export_corpus(records, context)

    with open(filepath, "w", encoding="utf-8") as fh:
        for record_dict in exported:
            fh.write(json.dumps(record_dict, ensure_ascii=False) + "\n")

    return len(exported)




# ---------------------------------------------------------------------------
# Metadata export
# ---------------------------------------------------------------------------

def export_metadata(record: Any) -> Dict[str, Any]:
    """Export non-neural metadata fields only.

    Parameters
    ----------
    record : CanonicalRecord
        The record to export metadata from.

    Returns
    -------
    dict[str, any]
        Metadata-only dict.  Excludes review_text (neural input) and
        source_text_raw (restricted).

    Notes
    -----
    Intended for research and audit use cases where the full text
    fields are not needed or not permitted.
    """
    metadata_fields = tuple(
        f for f in PORTABLE_CANDIDATE_FIELDS
        if f not in ("review_text", "source_text_raw", "export_text_redacted")
    )

    result: Dict[str, Any] = {}
    for field_name in metadata_fields:
        if hasattr(record, field_name):
            raw_value = getattr(record, field_name)
            result[field_name] = _serialize_value(raw_value)
    return result
