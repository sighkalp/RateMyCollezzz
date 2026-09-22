"""Conservative text normalization for RRM deterministic similarity analysis.

This module provides text normalization intended ONLY for use in deterministic
duplicate and near-duplicate detection within the Review Risk Model (RRM).

Normalization is deliberately conservative:
- it does NOT remove stopwords
- it does NOT stem or lemmatize
- it does NOT transliterate Hindi text
- it does NOT perform semantic rewriting

These choices preserve evidence that other RRM components (neural or
rule-based) may need at later stages of the pipeline.

Architectural note:
    The RRM produces evidence signals. Trust (Layer 4) makes final decisions.
    Normalization output is evidence, not a moderation decision.

    High similarity does NOT automatically mean:
    - spam
    - malicious behaviour
    - coordinated manipulation

    The caller (higher-level RRM pipeline or Trust engine) decides what
    score is operationally meaningful.

Public API:
    normalize_for_similarity(text) -> str
    exact_fingerprint(text) -> str
"""

from __future__ import annotations

import hashlib
import re
import unicodedata


def normalize_for_similarity(text: str) -> str:
    """Normalize text for deterministic similarity and duplicate analysis.

    Parameters
    ----------
    text : str
        Raw review text to normalize.

    Returns
    -------
    str
        Normalized text ready for similarity comparison.

    Raises
    ------
    TypeError
        If *text* is not a str.

    Notes
    -----
    The normalization pipeline:

    1. **Type check** - reject non-string input immediately.
    2. **Unicode NFKC normalization** - canonical decomposition followed
       by canonical composition.  This resolves compatibility characters
       (e.g. full-width Latin, superscript digits) while preserving
       semantic content.
    3. **Unicode-aware case folding** - ``casefold()`` is more aggressive
       than ``lower()`` and handles special cases such as the German
       eszett (ß → ss).  This is safe for English, Roman Hindi, and
       Hinglish because those scripts use Latin characters where
       case-fold has predictable behaviour.
    4. **Whitespace collapse** - any sequence of one or more whitespace
       characters is collapsed to a single ASCII space.
    5. **Trim** - leading and trailing whitespace is removed.

    What is preserved intentionally:
    - Punctuation (``!``, ``?``, ``.``, etc.)
    - Digits
    - URLs (punctuation is preserved, so ``https://...`` survives)
    - Email addresses
    - Normal word content including Hindi words written in Roman script

    What is NOT done (by design):
    - No transliteration of Hindi words written in Devanagari
    - No stemming or lemmatization
    - No stopword removal
    - No semantic rewriting
    - No HTML stripping (caller should strip HTML before calling)
    """
    if not isinstance(text, str):
        raise TypeError(
            f"normalize_for_similarity expects str, got {type(text).__name__}"
        )

    # Step 1: Unicode NFKC normalization
    normalized = unicodedata.normalize("NFKC", text)

    # Step 2: Unicode-aware case folding
    normalized = normalized.casefold()

    # Step 3: Collapse consecutive whitespace to single ASCII space
    normalized = re.sub(r"\s+", " ", normalized)

    # Step 4: Trim leading/trailing whitespace
    normalized = normalized.strip()

    return normalized


def exact_fingerprint(text: str) -> str:
    """Compute a deterministic SHA-256 fingerprint of normalized text.

    Parameters
    ----------
    text : str
        Raw review text to fingerprint.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest of the UTF-8 encoded
        normalized text.

    Notes
    -----
    The fingerprint is deterministic across runs and platforms because:
    - normalization is deterministic
    - SHA-256 is a standard hash function
    - the normalized text is encoded as UTF-8 before hashing

    Fingerprints enable exact-duplicate detection without storing or
    comparing raw text.  Two texts with the same fingerprint are
    byte-for-byte identical after normalization.
    """
    normalized = normalize_for_similarity(text)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return digest
