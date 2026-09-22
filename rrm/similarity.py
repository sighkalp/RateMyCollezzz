"""Deterministic near-duplicate similarity for the RRM.

This module implements character n-gram Jaccard similarity for detecting
near-duplicate reviews.  It is part of the RRM deterministic pre-checks
pipeline and produces evidence signals only -- it does NOT make moderation
decisions.

Architectural note:
    RRM UNDERSTANDS. TRUST DECIDES.

    This module emits similarity evidence.  Trust (Layer 4) decides what
    action to take based on that evidence.  A high similarity score does
    NOT automatically mean:
    - the reviews are spam
    - the reviews are malicious
    - the reviews are part of a coordinated campaign

    The caller decides what threshold or policy applies.

Similarity mathematics:
    Character n-gram Jaccard similarity between two texts A and B:

        J(A, B) = |ngrams(A) ∩ ngrams(B)| / |ngrams(A) ∪ ngrams(B)|

    where ngrams(text) is the set of all contiguous character sequences
    of length n in the normalized text.

    - n=4 (default) is chosen because it is small enough to capture
      character-level patterns in short reviews while being robust to
      minor edits (insertion, deletion, substitution of a few characters).
    - Character n-grams are language-agnostic deterministic features that
      can capture local character overlap and spelling variation in
      English, Hinglish, and Roman-Hindi text. Their empirical
      effectiveness for the RMC domain must be measured during later
      experiments.
    - For Devanagari text (not yet a primary target), character n-grams
      still work but may need n adjustment for syllable-aware behavior.

Public API:
    character_ngrams(text, n) -> set[str]
    jaccard_similarity(text_a, text_b, n) -> float
    SimilarityMatch (dataclass)
    find_best_match(query_text, candidates, *, n, exclude_review_id) -> SimilarityMatch | None
"""

from __future__ import annotations

import dataclasses
from typing import Iterable

from rrm.text_normalization import normalize_for_similarity


@dataclasses.dataclass(frozen=True)
class SimilarityMatch:
    """Immutable result of a similarity comparison.

    Attributes
    ----------
    review_id : str
        Identifier of the matched review.
    score : float
        Jaccard similarity score in [0.0, 1.0].
    exact_duplicate : bool
        True when the candidate is an exact (fingerprint) match
        to the query, not merely a high-similarity near-duplicate.
    """

    review_id: str
    score: float
    exact_duplicate: bool


def character_ngrams(text: str, n: int = 4) -> set[str]:
    """Extract character n-grams from normalized text.

    Parameters
    ----------
    text : str
        Raw text to n-gram-ize.  Will be normalized before extraction.
    n : int, optional
        Length of each n-gram.  Must be >= 1.  Default is 4.

    Returns
    -------
    set[str]
        Set of unique character n-grams.  Order is not preserved
        (set semantics).

    Raises
    ------
    TypeError
        If *text* is not a str.
    ValueError
        If *n* < 1.

    Notes
    -----
    The text is normalized using :func:`rrm.text_normalization.normalize_for_similarity`
    before n-gram extraction.

    For strings shorter than n, a single n-gram containing the entire
    string is returned (after normalization).  This ensures short texts
    can still participate in similarity computation without errors.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"character_ngrams expects str, got {type(text).__name__}"
        )
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    normalized = normalize_for_similarity(text)
    length = len(normalized)

    if length == 0:
        return set()

    if length < n:
        return {normalized}

    return set(normalized[i : i + n] for i in range(length - n + 1))


def jaccard_similarity(
    text_a: str,
    text_b: str,
    n: int = 4,
) -> float:
    """Compute character n-gram Jaccard similarity between two texts.

    Parameters
    ----------
    text_a : str
        First text.
    text_b : str
        Second text.
    n : int, optional
        N-gram size.  Must be >= 1.  Default is 4.

    Returns
    -------
    float
        Jaccard similarity score in [0.0, 1.0].

        - 1.0 when both texts normalize to identical content
        - 1.0 when both texts normalize to empty strings
        - 0.0 when one text is empty and the other is non-empty
        - Otherwise: |A ∩ B| / |A ∪ B|

    Raises
    ------
    TypeError
        If either text is not a str.
    ValueError
        If n < 1.

    Notes
    -----
    This function is deterministic and does not depend on any
    external state.
    """
    if not isinstance(text_a, str):
        raise TypeError(
            f"jaccard_similarity expects str for text_a, "
            f"got {type(text_a).__name__}"
        )
    if not isinstance(text_b, str):
        raise TypeError(
            f"jaccard_similarity expects str for text_b, "
            f"got {type(text_b).__name__}"
        )
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    ngrams_a = character_ngrams(text_a, n)
    ngrams_b = character_ngrams(text_b, n)

    # Both empty after normalization
    if not ngrams_a and not ngrams_b:
        return 1.0

    # One empty, one non-empty
    if not ngrams_a or not ngrams_b:
        return 0.0

    intersection = ngrams_a & ngrams_b
    union = ngrams_a | ngrams_b

    return len(intersection) / len(union)


def find_best_match(
    query_text: str,
    candidates: Iterable[tuple[str, str]],
    *,
    n: int = 4,
    exclude_review_id: str | None = None,
) -> SimilarityMatch | None:
    """Find the candidate review with the highest similarity to the query.

    Parameters
    ----------
    query_text : str
        The review text to compare against candidates.
    candidates : iterable of (review_id, review_text)
        Existing reviews to check against.  Each entry is a 2-tuple
        of (unique identifier, raw text).
    n : int, optional
        N-gram size for similarity computation.  Must be >= 1.
        Default is 4.
    exclude_review_id : str or None, optional
        If provided, the candidate with this review_id is skipped.
        Useful for excluding the query review itself from self-matching.

    Returns
    -------
    SimilarityMatch or None
        The best match found, or None if no candidates are available
        after filtering.

        When multiple candidates have the same highest score, the one
        with the lexicographically smallest review_id is chosen
        (deterministic tie-breaking).

    Notes
    -----
    This function does NOT impose any global near-duplicate threshold.
    The caller is responsible for deciding what score is operationally
    meaningful.  This design keeps rule-derived evidence distinguishable
    from Trust-layer policy decisions.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    candidates_list = list(candidates)

    if not candidates_list:
        return None

    query_fingerprint: str | None = None

    best_match: SimilarityMatch | None = None

    for review_id, review_text in candidates_list:
        if review_id == exclude_review_id:
            continue

        score = jaccard_similarity(query_text, review_text, n)

        if query_fingerprint is None:
            from rrm.text_normalization import exact_fingerprint
            query_fingerprint = exact_fingerprint(query_text)

        candidate_fingerprint = exact_fingerprint(review_text)
        exact_duplicate = query_fingerprint == candidate_fingerprint

        candidate_match = SimilarityMatch(
            review_id=review_id,
            score=score,
            exact_duplicate=exact_duplicate,
        )

        if best_match is None:
            best_match = candidate_match
            continue

        if score > best_match.score:
            best_match = candidate_match
        elif score == best_match.score:
            # Deterministic tie-breaking: lexicographically smallest review_id
            if review_id < best_match.review_id:
                best_match = candidate_match

    return best_match
