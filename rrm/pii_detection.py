"""Deterministic PII evidence extraction for the RRM.

This module provides conservative pattern-based detection of observable
text patterns that *may* indicate personally identifiable information
(PII).  It produces evidence signals only -- it does NOT make privacy
or moderation decisions.

Architectural note:
    RRM UNDERSTANDS. TRUST DECIDES.

    Detecting an email or phone pattern is NOT the same as confirming
    a privacy violation.  For example:

        admissions@college.example

    may match the EMAIL pattern, but whether it is personal/private PII
    is a later policy/context decision made by Trust (Layer 4).

    phone_count > 0 does NOT automatically mean:
        pii label = 1

    The deterministic component reports observable evidence only.
    Trust decides what that evidence means operationally.

Supported evidence kinds (initial set):
    EMAIL, PHONE, URL

Limitations:
    - Email detection uses a conservative regex and will miss some
      valid but unusual addresses.
    - Phone detection is intentionally conservative to avoid matching
      years, room numbers, and ordinary digit sequences.
    - URL detection supports http:// and https:// only.
    - No context-aware disambiguation is performed.
    - Detected patterns are not verified to belong to private persons
      vs. institutions.

Public API:
    PIIMatch (dataclass)
    detect_emails(text) -> tuple[PIIMatch, ...]
    detect_phones(text) -> tuple[PIIMatch, ...]
    detect_urls(text) -> tuple[PIIMatch, ...]
    detect_pii(text) -> tuple[PIIMatch, ...]
"""

from __future__ import annotations

import dataclasses
import re
from typing import Tuple

from rrm.text_normalization import normalize_for_similarity


@dataclasses.dataclass(frozen=True)
class PIIMatch:
    """Immutable evidence record for a detected PII pattern.

    Attributes
    ----------
    kind : str
        One of "EMAIL", "PHONE", "URL".
    start : int
        Character offset of the start of the match in the original text.
    end : int
        Character offset immediately past the end of the match
        (Python-style end index).
    text : str
        The exact substring from the original text that was matched.
    """

    kind: str
    start: int
    end: int
    text: str


# ---------------------------------------------------------------------------
# Email detection
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.ASCII,
)


def detect_emails(text: str) -> Tuple[PIIMatch, ...]:
    """Detect email address patterns in text.

    Parameters
    ----------
    text : str
        Raw text to scan.

    Returns
    -------
    tuple[PIIMatch, ...]
        Immutable matches sorted by (start, end, kind).
        Empty tuple when no emails are found.

    Raises
    ------
    TypeError
        If *text* is not a str.

    Notes
    -----
    The pattern is conservative and follows a standard local-part @
    domain TLD structure.  It may miss unusual but valid addresses.

    A detected email is a pattern match, NOT a confirmed privacy
    violation.  Contextual and policy decisions belong to Trust.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"detect_emails expects str, got {type(text).__name__}"
        )

    matches = []
    for m in _EMAIL_RE.finditer(text):
        matches.append(
            PIIMatch(kind="EMAIL", start=m.start(), end=m.end(), text=m.group())
        )
    return _sort_matches(matches)


# ---------------------------------------------------------------------------
# Phone detection
# ---------------------------------------------------------------------------

# Matches 10+ consecutive digits (e.g. "9876543210", "18001234567").
# Excludes short digit sequences like "2026", "123", "404".
_PHONE_10PLUS_RE = re.compile(r"\d{10,}")

# Matches digit groups separated by dashes or dots, total digits >= 10.
_PHONE_DASH_RE = re.compile(r"\d{2,}[\-.]\d{2,}[\-.]?\d*")

# Matches digit groups separated by spaces, total digits >= 10.
_PHONE_SPACE_RE = re.compile(r"\d{2,}\s+\d{2,}(?:\s+\d+)*")

# Matches parenthesized area code: (XXX) XXXXXXX or (XXX) XXXX-XXXX etc.
# Requires the entire digit group (with parens and separator stripped)
# to be >= 10 digits.
_PHONE_PAREN_RE = re.compile(r"\(\d{2,5}\)[\s\-.]?\d{4,}[\d\s\-.]*")

# Matches international prefix (+ or 00) followed by digit groups with
# flexible separators. E.g. "+91-98765-43210", "+1 (555) 123-4567",
# "00 91 98765 43210".
_PHONE_INTL_RE = re.compile(r"(?:\+|00)\s*\d{1,3}[\s\-.]?\d{4,}[\d\s\-.]*")


def _digits_only(s: str) -> str:
    """Strip non-digit characters from a string."""
    return "".join(c for c in s if c.isdigit())


def _phone_like(s: str) -> bool:
    """Return True when the digit content looks like a phone number.

    Conservative heuristic:
    - Must have at least 10 digits total after stripping separators.
    - Must not be all-identical digits (e.g. "1111111111").
    """
    digits = _digits_only(s)
    if len(digits) < 10:
        return False
    # Reject obvious all-identical runs (rarely real phone numbers).
    if len(set(digits)) == 1:
        return False
    return True


def detect_phones(text: str) -> Tuple[PIIMatch, ...]:
    """Detect phone-number-like patterns in text.

    Parameters
    ----------
    text : str
        Raw text to scan.

    Returns
    -------
    tuple[PIIMatch, ...]
        Immutable matches sorted by (start, end, kind).
        Empty tuple when no phone-like patterns are found.

    Raises
    ------
    TypeError
        If *text* is not a str.

    Notes
    -----
    This function intentionally does NOT match:
    - Short digit sequences like ``2026``, ``123``, ``404``
    - Room numbers (``room 404``)
    - Semester numbers (``semester 4``)
    - Pure years

    Phone detection is a conservative pattern match.  It is NOT a
    confirmation that the number belongs to a private person or that
    sharing it constitutes a policy violation.

    Indian phone numbers (10 digits starting with 6-9) and
    international-style numbers are the primary targets.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"detect_phones expects str, got {type(text).__name__}"
        )

    matches: list[PIIMatch] = []

    # Pattern 1: 10+ consecutive digits.
    for m in _PHONE_10PLUS_RE.finditer(text):
        digit_count = len(m.group())
        if digit_count >= 10:
            matches.append(
                PIIMatch(
                    kind="PHONE",
                    start=m.start(),
                    end=m.end(),
                    text=m.group(),
                )
            )

    # Pattern 2: Digit groups separated by dashes/dots (total digits >= 10).
    for m in _PHONE_DASH_RE.finditer(text):
        if _phone_like(m.group()):
            matches.append(
                PIIMatch(
                    kind="PHONE",
                    start=m.start(),
                    end=m.end(),
                    text=m.group(),
                )
            )

    # Pattern 3: Digit groups separated by spaces (total digits >= 10).
    for m in _PHONE_SPACE_RE.finditer(text):
        if _phone_like(m.group()):
            matches.append(
                PIIMatch(
                    kind="PHONE",
                    start=m.start(),
                    end=m.end(),
                    text=m.group(),
                )
            )

    # Pattern 4: Parenthesized area code.
    for m in _PHONE_PAREN_RE.finditer(text):
        if _phone_like(m.group()):
            matches.append(
                PIIMatch(
                    kind="PHONE",
                    start=m.start(),
                    end=m.end(),
                    text=m.group(),
                )
            )

    # Pattern 5: International prefix (+ or 00).
    for m in _PHONE_INTL_RE.finditer(text):
        if _phone_like(m.group()):
            matches.append(
                PIIMatch(
                    kind="PHONE",
                    start=m.start(),
                    end=m.end(),
                    text=m.group(),
                )
            )

    return _sort_matches(matches)


# ---------------------------------------------------------------------------
# URL detection
# ---------------------------------------------------------------------------

_URL_RE = re.compile(
    r"https?://[^\s)>\"']+",
    re.ASCII,
)


def detect_urls(text: str) -> Tuple[PIIMatch, ...]:
    """Detect http:// and https:// URLs in text.

    Parameters
    ----------
    text : str
        Raw text to scan.

    Returns
    -------
    tuple[PIIMatch, ...]
        Immutable matches sorted by (start, end, kind).
        Empty tuple when no URLs are found.

    Raises
    ------
    TypeError
        If *text* is not a str.

    Notes
    -----
    Only http:// and https:// URLs are supported initially.

    Not supported: ftp://, www without scheme, bare domains, IP
    addresses without scheme.  These may be added in later iterations
    if needed.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"detect_urls expects str, got {type(text).__name__}"
        )

    matches = []
    for m in _URL_RE.finditer(text):
        matches.append(
            PIIMatch(kind="URL", start=m.start(), end=m.end(), text=m.group())
        )
    return _sort_matches(matches)


# ---------------------------------------------------------------------------
# Unified detection
# ---------------------------------------------------------------------------


def detect_pii(text: str) -> Tuple[PIIMatch, ...]:
    """Detect all supported PII patterns in text.

    Parameters
    ----------
    text : str
        Raw text to scan.

    Returns
    -------
    tuple[PIIMatch, ...]
        Combined, deterministically ordered matches from all detectors.

    Raises
    ------
    TypeError
        If *text* is not a str.

    Notes
    -----
    Results are ordered by (start, end, kind) to ensure deterministic
    output across repeated calls.

    This function reports evidence only.  It does NOT classify the
    detected patterns as violations.  Policy decisions are made by
    Trust (Layer 4).
    """
    if not isinstance(text, str):
        raise TypeError(
            f"detect_pii expects str, got {type(text).__name__}"
        )

    matches: list[PIIMatch] = []
    matches.extend(detect_emails(text))
    matches.extend(detect_phones(text))
    matches.extend(detect_urls(text))
    return _sort_matches(matches)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sort_matches(
    matches: list[PIIMatch],
) -> Tuple[PIIMatch, ...]:
    """Sort matches deterministically and remove exact and overlapping duplicates.

    Exact duplicates (same start, end, kind) are collapsed.
    Overlapping matches of the same kind are deduplicated by keeping
    the one with the larger span (more specific match wins over a
    sub-match).
    """
    # Sort first by start, then by descending span length (longer first
    # at the same start), then by end, then by kind.
    sorted_matches = sorted(
        matches,
        key=lambda m: (m.start, -(m.end - m.start), m.end, m.kind),
    )

    unique: list[PIIMatch] = []
    for m in sorted_matches:
        # Check if this match is fully contained in an already-accepted match
        # of the same kind.
        contained = False
        for existing in unique:
            if m.kind == existing.kind and existing.start <= m.start and m.end <= existing.end:
                contained = True
                break
        if not contained:
            # Also check exact (start, end, kind) duplicates.
            if not any(
                (om.start, om.end, om.kind) == (m.start, m.end, m.kind)
                for om in unique
            ):
                unique.append(m)

    # Final sort by (start, end, kind) for deterministic ordering.
    unique.sort(key=lambda m: (m.start, m.end, m.kind))
    return tuple(unique)
