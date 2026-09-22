"""Descriptive text-evidence extraction for the RRM.

This module computes deterministic, descriptive statistics about review
text.  It produces evidence only -- it does NOT classify spam, toxicity,
advertising, or any other risk label.

Architectural note:
    RRM UNDERSTANDS. TRUST DECIDES.

    This module reports what is observable in the text.  It does NOT
    judge whether a particular pattern is harmful, spam, or a policy
    violation.  Trust (Layer 4) decides what action the evidence
    warrants.

Public API:
    TextEvidence (dataclass)
    extract_text_evidence(text) -> TextEvidence
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from rrm.pii_detection import detect_pii


@dataclasses.dataclass(frozen=True)
class TextEvidence:
    """Immutable collection of descriptive text statistics.

    All counts are computed deterministically from the raw input text.
    No classification, scoring, or thresholding is performed.

    Attributes
    ----------
    character_count : int
        Total character count of the original text (``len(text)``).
    non_whitespace_character_count : int
        Count of characters for which ``c.isspace()`` is False.
    word_count : int
        Count of whitespace-delimited, non-empty tokens.
    digit_count : int
        Count of characters where ``c.isdigit()`` is True.
    uppercase_character_count : int
        Count of characters where ``c.isupper()`` is True.
    exclamation_count : int
        Count of ``!`` characters.
    question_count : int
        Count of ``?`` characters.
    repeated_character_run_count : int
        Count of runs of the same character with length >= 3.
        Whitespace runs are excluded.
    max_character_run_length : int
        Maximum qualifying repeated-character run length, or 0 if none.
    email_count : int
        Number of email patterns detected (from pii_detection).
    phone_count : int
        Number of phone-like patterns detected (from pii_detection).
    url_count : int
        Number of URL patterns detected (from pii_detection).
    """

    character_count: int
    non_whitespace_character_count: int
    word_count: int
    digit_count: int
    uppercase_character_count: int
    exclamation_count: int
    question_count: int
    repeated_character_run_count: int
    max_character_run_length: int
    email_count: int
    phone_count: int
    url_count: int


def extract_text_evidence(text: str) -> TextEvidence:
    """Extract descriptive evidence from review text.

    Parameters
    ----------
    text : str
        Raw review text.

    Returns
    -------
    TextEvidence
        Immutable evidence record with descriptive counts.

    Raises
    ------
    TypeError
        If *text* is not a str.

    Notes
    -----
    This function is deterministic.  It does NOT:
    - classify the text as spam, toxic, or advertising
    - assign risk scores
    - apply thresholds
    - make moderation decisions

    It reports what is observable.  Trust decides what it means.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"extract_text_evidence expects str, got {type(text).__name__}"
        )

    character_count = len(text)

    non_whitespace_character_count = sum(
        1 for c in text if not c.isspace()
    )

    word_count = sum(1 for token in text.split() if token)

    digit_count = sum(1 for c in text if c.isdigit())

    uppercase_character_count = sum(1 for c in text if c.isupper())

    exclamation_count = text.count("!")

    question_count = text.count("?")

    # Repeated-character runs: runs of the same character with length >= 3.
    # Whitespace runs are excluded.
    repeated_character_run_count = 0
    max_character_run_length = 0

    i = 0
    while i < len(text):
        c = text[i]
        if c.isspace():
            i += 1
            continue

        run_start = i
        while i < len(text) and text[i] == c:
            i += 1
        run_length = i - run_start

        if run_length >= 3:
            repeated_character_run_count += 1
            if run_length > max_character_run_length:
                max_character_run_length = run_length

    # PII counts from deterministic detection
    pii_matches = detect_pii(text)
    email_count = sum(1 for m in pii_matches if m.kind == "EMAIL")
    phone_count = sum(1 for m in pii_matches if m.kind == "PHONE")
    url_count = sum(1 for m in pii_matches if m.kind == "URL")

    return TextEvidence(
        character_count=character_count,
        non_whitespace_character_count=non_whitespace_character_count,
        word_count=word_count,
        digit_count=digit_count,
        uppercase_character_count=uppercase_character_count,
        exclamation_count=exclamation_count,
        question_count=question_count,
        repeated_character_run_count=repeated_character_run_count,
        max_character_run_length=max_character_run_length,
        email_count=email_count,
        phone_count=phone_count,
        url_count=url_count,
    )
