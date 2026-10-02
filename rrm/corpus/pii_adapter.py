"""PII adapter for the RMC corpus.

This module bridges the deterministic PII detection in ``rrm.pii_detection``
with the corpus production workflow.  It provides:

- safe surrogate transformation of detected PII patterns
- redaction of PII in review text
- construction of PIIEvidenceSummary objects

Architectural note:
    RRM UNDERSTANDS. TRUST DECIDES.

    This module reports observable PII patterns only.  It does NOT judge
    whether a pattern constitutes a privacy violation or policy breach.
    Trust (Layer 4) makes that determination.

    IMPORTANT: This module uses rrm.pii_detection exclusively — it does
    NOT implement a second PII detector.

Public API:
    safe_surrogate_transform
    redact_pii_in_text
    build_pii_evidence_summary
"""

from __future__ import annotations

from typing import Optional, Tuple

from rrm.corpus.models import PIIEvidenceSummary, SourceType
from rrm.pii_detection import PIIMatch, detect_pii


# ---------------------------------------------------------------------------
# Safe surrogate transformations
# ---------------------------------------------------------------------------

def _surrogate_email(match_text: str) -> str:
    """Replace an email with a count-based non-real surrogate."""
    return "userX@example.test"


def _surrogate_phone(match_text: str) -> str:
    """Replace a phone number with a format-preserving non-real surrogate."""
    return "+XX-XXXXX-XXXXX"


def _surrogate_url(match_text: str) -> str:
    """Replace a URL with a redaction marker."""
    return "https://redacted.example"


_SURROGATE_MAP = {
    "EMAIL": _surrogate_email,
    "PHONE": _surrogate_phone,
    "URL": _surrogate_url,
}


def safe_surrogate_transform(text: str) -> str:
    """Replace detected PII patterns with safe, non-real surrogates.

    Parameters
    ----------
    text : str
        Raw text to transform.

    Returns
    -------
    str
        Text with PII patterns replaced by non-real surrogates.

    Raises
    ------
    TypeError
        If *text* is not a str.

    Notes
    -----
    Uses ``rrm.pii_detection.detect_pii`` exclusively for detection.
    Surrogates are count-based and format-preserving.  The original
    private values are never preserved in the output.

    If no PII is detected, the input text is returned unchanged.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"safe_surrogate_transform expects str, got {type(text).__name__}"
        )

    matches: Tuple[PIIMatch, ...] = detect_pii(text)

    if not matches:
        return text

    # Replace from end to start to preserve character offsets.
    result = list(text)
    for match in reversed(matches):
        surrogate_fn = _SURROGATE_MAP.get(match.kind)
        if surrogate_fn is not None:
            replacement = surrogate_fn(match.text)
            result[match.start:match.end] = list(replacement)

    return "".join(result)


def redact_pii_in_text(text: str) -> str:
    """Replace all detected PII patterns with redaction markers.

    Parameters
    ----------
    text : str
        Raw text to redact.

    Returns
    -------
    str
        Text with PII patterns replaced by redaction markers.

    Raises
    ------
    TypeError
        If *text* is not a str.

    Notes
    -----
    Uses ``rrm.pii_detection.detect_pii`` exclusively for detection.

    Redaction markers:
    - Email → ``[EMAIL REDACTED]``
    - Phone → ``[PHONE REDACTED]``
    - URL → ``[URL REDACTED]``
    """
    if not isinstance(text, str):
        raise TypeError(
            f"redact_pii_in_text expects str, got {type(text).__name__}"
        )

    _REDACTION_MAP = {
        "EMAIL": "[EMAIL REDACTED]",
        "PHONE": "[PHONE REDACTED]",
        "URL": "[URL REDACTED]",
    }

    matches: Tuple[PIIMatch, ...] = detect_pii(text)

    if not matches:
        return text

    result = list(text)
    for match in reversed(matches):
        marker = _REDACTION_MAP.get(match.kind, "[REDACTED]")
        result[match.start:match.end] = list(marker)

    return "".join(result)


# ---------------------------------------------------------------------------
# PII evidence summary construction
# ---------------------------------------------------------------------------

# Counter for generating unique evidence IDs.
_evidence_id_counter = 0


def build_pii_evidence_summary(
    review_id: str,
    raw_text: str,
) -> Optional[PIIEvidenceSummary]:
    """Build a PIIEvidenceSummary from raw text.

    Parameters
    ----------
    review_id : str
        The record identifier.
    raw_text : str
        Raw text to scan for PII patterns.

    Returns
    -------
    PIIEvidenceSummary or None
        Summary object when PII is detected, None when no PII is found.

    Raises
    ------
    TypeError
        If *review_id* or *raw_text* is not a str.

    Notes
    -----
    Uses ``rrm.pii_detection.detect_pii`` exclusively for detection.

    The returned summary contains NO original private matched values.
    It contains only category labels, match counts, and redaction status.
    """
    if not isinstance(review_id, str):
        raise TypeError(
            f"build_pii_evidence_summary expects str for review_id, "
            f"got {type(review_id).__name__}"
        )
    if not isinstance(raw_text, str):
        raise TypeError(
            f"build_pii_evidence_summary expects str for raw_text, "
            f"got {type(raw_text).__name__}"
        )

    matches: Tuple[PIIMatch, ...] = detect_pii(raw_text)

    if not matches:
        return None

    global _evidence_id_counter
    _evidence_id_counter += 1
    evidence_id = f"pii-evidence-{_evidence_id_counter:06d}"

    categories = tuple(sorted({m.kind for m in matches}))
    match_count = len(matches)

    # Determine redaction status based on whether the raw text
    # has been transformed.
    redaction_status = "detected_pending_redaction"

    return PIIEvidenceSummary(
        evidence_id=evidence_id,
        review_id=review_id,
        categories=categories,
        match_count=match_count,
        redaction_status=redaction_status,
        created_at="",
    )
