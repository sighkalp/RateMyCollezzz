"""Unified deterministic precheck pipeline for the RRM.

This module aggregates all deterministic evidence-generation steps into a
single immutable result.  It orchestrates existing modules without
reimplementing their logic.

Architectural note:
    RRM UNDERSTANDS. TRUST DECIDES.

    This module is deterministic evidence aggregation.  It makes NO
    policy decisions.

    - exact duplicate != spam automatically
    - high similarity != malicious behavior automatically
    - PII pattern != confirmed privacy violation
    - phone_count > 0 does NOT automatically mean: pii label = 1

    Trust Layer decides actions based on this evidence.

Public API:
    DeterministicPrecheckResult (frozen dataclass)
    run_deterministic_prechecks(review_id, review_text, candidates, *, n)
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Tuple

from rrm.pii_detection import PIIMatch, detect_pii
from rrm.similarity import SimilarityMatch, find_best_match
from rrm.text_evidence import TextEvidence, extract_text_evidence
from rrm.text_normalization import exact_fingerprint


@dataclasses.dataclass(frozen=True)
class DeterministicPrecheckResult:
    """Immutable aggregation of all deterministic precheck evidence.

    This object contains evidence only.  It does NOT contain:
    - risk scores
    - moderation actions
    - spam / toxicity / advertising / off-topic labels
    - final PII policy labels
    - Trust decisions

    Fields
    ------
    review_id : str
        Identifier of the review being analyzed.
    fingerprint : str
        SHA-256 hex digest of the normalized text.
    text_evidence : TextEvidence
        Descriptive text statistics and PII counts.
    pii_matches : tuple[PIIMatch, ...]
        Raw PII pattern matches with original text offsets.
    best_similarity_match : SimilarityMatch or None
        Best similarity match against candidates, or None when no
        candidates are available.
    """

    review_id: str
    fingerprint: str
    text_evidence: TextEvidence
    pii_matches: Tuple[PIIMatch, ...]
    best_similarity_match: SimilarityMatch | None


def run_deterministic_prechecks(
    review_id: str,
    review_text: str,
    candidates: Iterable[tuple[str, str]] = (),
    *,
    n: int = 4,
) -> DeterministicPrecheckResult:
    """Run the full deterministic precheck pipeline on a review.

    Parameters
    ----------
    review_id : str
        Unique identifier for this review.  Must be non-empty after
        stripping whitespace.
    review_text : str
        Raw review text to analyze.
    candidates : iterable of (review_id, review_text), optional
        Existing reviews to compare against for similarity detection.
        Default is empty (no similarity check).
    n : int, optional
        N-gram size for similarity computation.  Must be >= 1.
        Default is 4.

    Returns
    -------
    DeterministicPrecheckResult
        Immutable evidence record aggregating all deterministic checks.

    Raises
    ------
    TypeError
        If *review_id* or *review_text* is not a str.
    ValueError
        If *review_id* is empty after stripping, or if *n* < 1.
        If any candidate is not a 2-item tuple/list with str elements.

    Notes
    -----
    Pipeline order:

    1. Compute exact fingerprint of the normalized text.
    2. Extract descriptive text evidence (character counts, word counts,
       punctuation counts, repeated-character runs, PII counts).
    3. Detect raw PII pattern matches with original text offsets.
    4. Find the best similarity match among candidates, excluding the
       current review_id.

    All sub-steps delegate to existing modules.  This function does not
    duplicate normalization, similarity, PII, or text-evidence logic.
    """
    # --- Input validation ---
    if not isinstance(review_id, str):
        raise TypeError(
            f"run_deterministic_prechecks expects str for review_id, "
            f"got {type(review_id).__name__}"
        )
    if not review_id.strip():
        raise ValueError(
            "review_id must not be empty or whitespace-only"
        )
    if not isinstance(review_text, str):
        raise TypeError(
            f"run_deterministic_prechecks expects str for review_text, "
            f"got {type(review_text).__name__}"
        )
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    # --- Materialize and validate candidates ---
    candidates_list: list[tuple[str, str]] = []
    for c in candidates:
        if not isinstance(c, (tuple, list)):
            raise ValueError(
                f"Each candidate must be a 2-item tuple or list, "
                f"got {type(c).__name__}: {c!r}"
            )
        if len(c) != 2:
            raise ValueError(
                f"Each candidate must have exactly 2 items, got {len(c)}: {c!r}"
            )
        c_id, c_text = c
        if not isinstance(c_id, str):
            raise TypeError(
                f"Candidate review_id must be str, got {type(c_id).__name__}: {c_id!r}"
            )
        if not isinstance(c_text, str):
            raise TypeError(
                f"Candidate review_text must be str, got {type(c_text).__name__}: {c_text!r}"
            )
        candidates_list.append((c_id, c_text))

    # --- Step 1: Fingerprint ---
    fingerprint = exact_fingerprint(review_text)

    # --- Step 2: Text evidence (includes PII counts via detect_pii) ---
    text_evidence = extract_text_evidence(review_text)

    # --- Step 3: Raw PII matches (for offsets and kind-level detail) ---
    pii_matches = detect_pii(review_text)

    # --- Step 4: Best similarity match ---
    best_match = find_best_match(
        review_text,
        candidates_list,
        n=n,
        exclude_review_id=review_id,
    )

    return DeterministicPrecheckResult(
        review_id=review_id,
        fingerprint=fingerprint,
        text_evidence=text_evidence,
        pii_matches=pii_matches,
        best_similarity_match=best_match,
    )
