"""Character n-gram TF-IDF + Logistic Regression baseline for the RRM.

This module implements a character-level TF-IDF + Logistic Regression
baseline as required by the RRM research plan (RRM 3.4B).  It serves
as a comparator against the word-level baseline (RRM 3.4A).

The training and evaluation machinery is shared with the word baseline
via the private helper ``_fit_tfidf_logreg_with_analyzer`` in
``baseline_tfidf_lr.py``.  Evaluation is delegated to
``evaluate_tfidf_logreg`` because the fitted object carries its
vectorizer and the evaluation pipeline is analyzer-independent.

Architectural note:
    RRM UNDERSTANDS. TRUST DECIDES.

    This baseline produces risk-probability evidence only.  It makes
    NO moderation decisions.  No Trust-layer action may depend on
    this component.

Research provenance:
    - Character n-gram TF-IDF + Logistic Regression: REQUIRED BY PROJECT
      PLAN (IMPLEMENTATION_PLAN.md section 3.4)
    - analyzer="char", ngram_range=(3,5): RRM EXPERIMENTAL DESIGN CHOICE
    - Reuse of shared training/evaluation pipeline: STANDARD ENGINEERING
      PRACTICE

Scientific purpose:
    Character n-grams test sensitivity to:
    - spelling variation (e.g. "goooood", "achiiii")
    - elongated characters (e.g. "facultyyyy")
    - noisy typing (e.g. "colleege")
    - simple obfuscation (e.g. "b3st", "fr@ud")
    - Roman-Hindi variation (e.g. "achha" vs "accha")
    - partial subword overlap

    These are hypotheses / representation capabilities.
    Character n-grams are NOT empirically claimed superior for RMC.
    That must be measured in evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from rrm.labels import PRIMARY_LABELS
from rrm.baseline_tfidf_lr import (
    BaselineEvaluation,
    FittedTfidfLogRegBaseline,
    TfidfLogRegConfig,
    _fit_tfidf_logreg_with_analyzer,
    evaluate_tfidf_logreg,
)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def fit_char_ngram_logreg(
    train_records: Iterable[Mapping[str, Any]],
    *,
    labels: Tuple[str, ...] = PRIMARY_LABELS,
    config: Optional[TfidfLogRegConfig] = None,
) -> FittedTfidfLogRegBaseline:
    """Fit a character n-gram TF-IDF + Logistic Regression baseline.

    Uses analyzer="char" with ngram_range=(3, 5) by default.

    Parameters
    ----------
    train_records : iterable of mapping-like
        Training records.  Each must contain 'review_id', 'review_text',
        and label fields.
    labels : tuple[str, ...]
        Label names to train models for.  Default is PRIMARY_LABELS.
    config : TfidfLogRegConfig or None
        Configuration.  If None, uses ``TfidfLogRegConfig(ngram_range=(3, 5))``.
        The analyzer is fixed to "char" regardless of config content.

    Returns
    -------
    FittedTfidfLogRegBaseline
        Fitted baseline with character TF-IDF vectorizer and per-label models.

    Raises
    ------
    TypeError
        On type violations in records or config.
    ValueError
        On value violations (empty review_id, invalid labels,
        duplicate review_id).
    """
    if config is None:
        config = TfidfLogRegConfig(ngram_range=(3, 5))

    return _fit_tfidf_logreg_with_analyzer(
        train_records,
        labels=labels,
        config=config,
        analyzer="char",
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_char_ngram_logreg(
    fitted: FittedTfidfLogRegBaseline,
    train_records: Iterable[Mapping[str, Any]],
    eval_records: Iterable[Mapping[str, Any]],
    *,
    labels: Tuple[str, ...] = PRIMARY_LABELS,
) -> BaselineEvaluation:
    """Evaluate a fitted character n-gram baseline.

    Delegates to :func:`evaluate_tfidf_logreg` — the evaluation pipeline
    is analyzer-independent because the fitted object carries its
    vectorizer.

    Parameters
    ----------
    fitted : FittedTfidfLogRegBaseline
        Fitted baseline from :func:`fit_char_ngram_logreg`.
    train_records : iterable of mapping-like
        Training records (used for leakage validation).
    eval_records : iterable of mapping-like
        Evaluation records.
    labels : tuple[str, ...]
        Label names to evaluate.  Default is PRIMARY_LABELS.

    Returns
    -------
    BaselineEvaluation
        Evaluation metrics per label, plus macro-F1.

    Raises
    ------
    ValueError
        If leakage is detected between train and eval.
    TypeError
        On type violations.
    """
    return evaluate_tfidf_logreg(
        fitted, train_records, eval_records, labels=labels
    )
