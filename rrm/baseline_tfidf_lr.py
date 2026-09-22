"""Word TF-IDF + Logistic Regression baseline for the RRM.

This module implements a reproducible word-level TF-IDF + Logistic
Regression baseline as required by the RRM research plan.  It serves
as a comparator against which the custom RRM must later earn its
complexity.

Architectural note:
    RRM UNDERSTANDS. TRUST DECIDES.

    This baseline produces risk-probability evidence only.  It makes
    NO moderation decisions.  No Trust-layer action may depend on
    this component.

Research provenance:
    - TF-IDF + Logistic Regression baseline: REQUIRED BY PROJECT PLAN
      (IMPLEMENTATION_PLAN.md section 3.4)
    - fit vectorizer on train only, leakage prevention, fixed random
      seed, masked unknown labels: STANDARD ENGINEERING PRACTICE
    - word ngram_range=(1,2), C=1.0, solver=liblinear, max_iter=1000,
      no class weighting: RRM EXPERIMENTAL DESIGN CHOICE

Mathematical background:
    TF-IDF (sklearn implementation with smooth_idf=True):
    - TF(t,d) = raw term frequency of term t in document d
    - idf(t) = log((1 + N) / (1 + df(t))) + 1
      where N = number of training documents
        df(t) = number of training documents containing term t
    - TF-IDF(t,d) = TF(t,d) * idf(t)
    - Resulting vectors are L2-normalized.
    - Frequent corpus-wide words receive lower idf weight.
    - The smoothed idf prevents division by zero for terms that appear
      in every document.

    Logistic Regression:
    p(y=1|x) = sigmoid(w . x + b)
    - w = learned feature weights (one per TF-IDF feature)
    - x = TF-IDF sparse vector
    - b = learned bias
    - sigmoid(z) = 1 / (1 + exp(-z)) maps to [0, 1]

Public API:
    TfidfLogRegConfig (frozen dataclass)
    FittedTfidfLogRegBaseline (frozen dataclass)
    LabelMetrics (frozen dataclass)
    BaselineEvaluation (frozen dataclass)
    PRIMARY_LABELS (tuple)
    UNKNOWN_LABEL (int)
    fit_tfidf_logreg(train_records, *, labels, config) -> FittedTfidfLogRegBaseline
    evaluate_tfidf_logreg(fitted, train_records, eval_records, *, labels) -> BaselineEvaluation
    validate_no_exact_leakage(train_records, eval_records) -> None
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from rrm.text_normalization import exact_fingerprint

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRIMARY_LABELS: Tuple[str, ...] = (
    "spam",
    "deception",
    "toxicity",
    "advertising",
    "off_topic",
    "pii",
)

UNKNOWN_LABEL: int = -1

_VALID_LABEL_VALUES = (0, 1, UNKNOWN_LABEL)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TfidfLogRegConfig:
    """Immutable configuration for the TF-IDF + Logistic Regression baseline.

    These are initial engineering defaults.  They are NOT claimed to be
    optimal hyperparameters.

    Attributes
    ----------
    random_state : int
        Random seed for reproducibility.
    ngram_range : tuple[int, int]
        Lower and upper n-gram bounds for the word analyzer.
    min_df : int
        Minimum document frequency for a term to be included.
    max_df : float
        Maximum document frequency ratio (0, 1.0] for a term to be
        included.
    max_features : int or None
        Maximum number of features (vocabulary size), or None for
        unlimited.
    sublinear_tf : bool
        If True, use log(1 + tf) instead of raw tf.
    C : float
        Inverse regularization strength for LogisticRegression.
    max_iter : int
        Maximum iterations for the logistic regression solver.
    solver : str
        Solver algorithm for LogisticRegression.
    """

    random_state: int = 42
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 1
    max_df: float = 1.0
    max_features: Optional[int] = None
    sublinear_tf: bool = False
    C: float = 1.0
    max_iter: int = 1000
    solver: str = "liblinear"

    def __post_init__(self) -> None:
        if not isinstance(self.random_state, int):
            raise TypeError(
                f"random_state must be int, got {type(self.random_state).__name__}"
            )
        if not isinstance(self.ngram_range, (tuple, list)) or len(self.ngram_range) != 2:
            raise TypeError(
                f"ngram_range must be a 2-item tuple, got {self.ngram_range!r}"
            )
        lo, hi = self.ngram_range
        if not isinstance(lo, int) or not isinstance(hi, int):
            raise TypeError(
                f"ngram_range bounds must be int, got ({type(lo).__name__}, {type(hi).__name__})"
            )
        if lo < 1 or hi < 1:
            raise ValueError(
                f"ngram_range bounds must be >= 1, got ({lo}, {hi})"
            )
        if lo > hi:
            raise ValueError(
                f"lower ngram ({lo}) must be <= upper ngram ({hi})"
            )
        if self.min_df < 1:
            raise ValueError(
                f"min_df must be >= 1, got {self.min_df}"
            )
        if not (0 < self.max_df <= 1.0):
            raise ValueError(
                f"max_df must be in (0, 1.0], got {self.max_df}"
            )
        if self.max_features is not None and self.max_features < 1:
            raise ValueError(
                f"max_features must be None or >= 1, got {self.max_features}"
            )
        if self.C <= 0:
            raise ValueError(
                f"C must be > 0, got {self.C}"
            )
        if self.max_iter < 1:
            raise ValueError(
                f"max_iter must be >= 1, got {self.max_iter}"
            )


# ---------------------------------------------------------------------------
# Fitted model container
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FittedTfidfLogRegBaseline:
    """Immutable container for a fitted TF-IDF + Logistic Regression baseline.

    Attributes
    ----------
    config : TfidfLogRegConfig
        Configuration used for training.
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer (fitted on train text only).
    models : dict[str, LogisticRegression]
        Mapping from label name to fitted LogisticRegression model.
    trained_labels : tuple[str, ...]
        Labels for which models were successfully trained.
    skipped_labels : dict[str, str]
        Mapping from label name to reason it was skipped.
    """

    config: TfidfLogRegConfig
    vectorizer: TfidfVectorizer
    models: Dict[str, LogisticRegression]
    trained_labels: Tuple[str, ...]
    skipped_labels: Dict[str, str]


# ---------------------------------------------------------------------------
# Metrics types
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LabelMetrics:
    """Immutable per-label evaluation metrics.

    Attributes
    ----------
    label : str
        Label name.
    support : int
        Number of evaluation samples for this label (after masking
        unknown).
    negative_count : int
        Number of negative (0) samples.
    positive_count : int
        Number of positive (1) samples.
    precision : float
        Precision score.
    recall : float
        Recall score.
    f1 : float
        F1 score.
    auprc : float
        Area Under the Precision-Recall Curve.
    """

    label: str
    support: int
    negative_count: int
    positive_count: int
    precision: float
    recall: float
    f1: float
    auprc: float


@dataclasses.dataclass(frozen=True)
class BaselineEvaluation:
    """Immutable evaluation result for the TF-IDF + Logistic Regression baseline.

    Attributes
    ----------
    per_label : tuple[LabelMetrics, ...]
        Per-label metrics for successfully evaluated labels.
    macro_f1 : float or None
        Mean F1 across evaluated labels.  None if no label was
        evaluated.
    evaluated_labels : tuple[str, ...]
        Labels that were successfully evaluated.
    skipped_labels : dict[str, str]
        Mapping from label name to reason it was skipped.
    """

    per_label: Tuple[LabelMetrics, ...]
    macro_f1: Optional[float]
    evaluated_labels: Tuple[str, ...]
    skipped_labels: Dict[str, str]


# ---------------------------------------------------------------------------
# Record validation
# ---------------------------------------------------------------------------


def _validate_records(
    records: Iterable[Mapping[str, Any]],
    split_name: str,
    labels: Tuple[str, ...] = PRIMARY_LABELS,
) -> Tuple[list, Dict[str, str]]:
    """Validate and return records with duplicate detection.

    Parameters
    ----------
    records : iterable of mapping-like
        Input records.
    split_name : str
        Name of the split (for error messages).
    labels : tuple[str, ...]
        Expected label field names.

    Returns
    -------
    list of dict
        Validated records as dicts.

    Raises
    ------
    TypeError
        On type violations.
    ValueError
        On value violations (empty review_id, invalid label values,
        duplicate review_id).
    """
    validated: list = []
    seen_ids: set = set()

    for i, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(
                f"Record {i} in {split_name} is not a mapping: {type(record).__name__}"
            )

        record = dict(record)

        # Validate review_id
        review_id = record.get("review_id")
        if not isinstance(review_id, str) or not review_id.strip():
            raise ValueError(
                f"Record {i} in {split_name}: review_id must be a non-empty str, "
                f"got {review_id!r}"
            )

        # Check duplicates
        if review_id in seen_ids:
            raise ValueError(
                f"Duplicate review_id '{review_id}' in {split_name} at record {i}"
            )
        seen_ids.add(review_id)

        # Validate review_text
        review_text = record.get("review_text")
        if not isinstance(review_text, str):
            raise TypeError(
                f"Record {review_id} in {split_name}: review_text must be str, "
                f"got {type(review_text).__name__}"
            )

        # Validate label values — key MUST exist; -1 means explicitly unknown
        for label in labels:
            if label not in record:
                raise ValueError(
                    f"Record {review_id} in {split_name}: missing required "
                    f"label field '{label}'"
                )
            value = record[label]
            if value not in _VALID_LABEL_VALUES:
                raise ValueError(
                    f"Record {review_id} in {split_name}: label '{label}' must be "
                    f"one of {_VALID_LABEL_VALUES}, got {value!r}"
                )

        validated.append(record)

    return validated, {}


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------


def validate_no_exact_leakage(
    train_records: Iterable[Mapping[str, Any]],
    eval_records: Iterable[Mapping[str, Any]],
) -> None:
    """Validate that no exact leakage exists between train and eval splits.

    Checks:
    1. Same review_id appearing in both splits.
    2. Exact normalized duplicate text (via exact_fingerprint) crossing
       splits.

    Parameters
    ----------
    train_records : iterable of mapping-like
        Training records.
    eval_records : iterable of mapping-like
        Evaluation records.

    Raises
    ------
    ValueError
        If leakage is detected.
    """
    train_ids = set()
    train_fingerprints = set()

    for record in train_records:
        rid = record.get("review_id") if isinstance(record, Mapping) else None
        text = record.get("review_text") if isinstance(record, Mapping) else None
        if rid is not None:
            train_ids.add(rid)
        if text is not None:
            train_fingerprints.add(exact_fingerprint(text))

    for record in eval_records:
        rid = record.get("review_id") if isinstance(record, Mapping) else None
        text = record.get("review_text") if isinstance(record, Mapping) else None

        if rid is not None and rid in train_ids:
            raise ValueError(
                f"Leakage detected: review_id '{rid}' appears in both train and eval splits."
            )

        if text is not None:
            fp = exact_fingerprint(text)
            if fp in train_fingerprints:
                raise ValueError(
                    f"Leakage detected: normalized text in eval matches a train "
                    f"record (review_id: {rid!r})."
                )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _fit_tfidf_logreg_with_analyzer(
    train_records: Iterable[Mapping[str, Any]],
    *,
    labels: Tuple[str, ...],
    config: TfidfLogRegConfig,
    analyzer: str,
) -> FittedTfidfLogRegBaseline:
    """Private shared trainer for TF-IDF + Logistic Regression baselines.

    Parameters
    ----------
    train_records : iterable of mapping-like
        Training records.  Each must contain 'review_id', 'review_text',
        and label fields.
    labels : tuple[str, ...]
        Label names to train models for.
    config : TfidfLogRegConfig
        Configuration.
    analyzer : str
        Analyzer for TfidfVectorizer.  ``"word"`` for the word baseline,
        ``"char"`` for the character n-gram baseline.

    Returns
    -------
    FittedTfidfLogRegBaseline
        Fitted baseline with vectorizer and per-label models.

    Raises
    ------
    TypeError
        On type violations in records or config.
    ValueError
        On value violations (empty review_id, invalid labels,
        duplicate review_id).
    """
    # Validate records
    validated, _ = _validate_records(train_records, "train", labels=labels)

    # Extract texts
    train_texts = [r["review_text"] for r in validated]

    # Fit ONE shared TF-IDF vectorizer on train text only
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
        max_features=config.max_features,
        sublinear_tf=config.sublinear_tf,
        lowercase=True,
        use_idf=True,
        smooth_idf=True,
        norm="l2",
    )
    X_train = vectorizer.fit_transform(train_texts)

    # Train one LogisticRegression per usable label
    models: Dict[str, LogisticRegression] = {}
    skipped_labels: Dict[str, str] = {}

    for label in labels:
        # Extract label values for this label (key guaranteed by validation)
        y = numpy.array([r[label] for r in validated])

        # Mask unknown rows
        known_mask = y != UNKNOWN_LABEL
        y_known = y[known_mask]
        X_label = X_train[known_mask]

        # Require both classes
        unique_classes = numpy.unique(y_known)
        if len(unique_classes) < 2:
            skipped_labels[label] = (
                "requires both supervised classes 0 and 1"
            )
            continue

        model = LogisticRegression(
            C=config.C,
            solver=config.solver,
            max_iter=config.max_iter,
            random_state=config.random_state,
        )
        model.fit(X_label, y_known)
        models[label] = model

    trained_labels = tuple(sorted(models.keys()))

    return FittedTfidfLogRegBaseline(
        config=config,
        vectorizer=vectorizer,
        models=models,
        trained_labels=trained_labels,
        skipped_labels=skipped_labels,
    )


def fit_tfidf_logreg(
    train_records: Iterable[Mapping[str, Any]],
    *,
    labels: Tuple[str, ...] = PRIMARY_LABELS,
    config: Optional[TfidfLogRegConfig] = None,
) -> FittedTfidfLogRegBaseline:
    """Fit a word-level TF-IDF + Logistic Regression baseline on training data.

    Parameters
    ----------
    train_records : iterable of mapping-like
        Training records.  Each must contain 'review_id', 'review_text',
        and label fields.
    labels : tuple[str, ...]
        Label names to train models for.  Default is PRIMARY_LABELS.
    config : TfidfLogRegConfig or None
        Configuration.  Uses defaults if None.

    Returns
    -------
    FittedTfidfLogRegBaseline
        Fitted baseline with vectorizer and per-label models.

    Raises
    ------
    TypeError
        On type violations in records or config.
    ValueError
        On value violations (empty review_id, invalid labels,
        duplicate review_id).

    Notes
    -----
    This function implements the word-analyzer TF-IDF baseline
    (analyzer="word").  For the character n-gram baseline, use
    :func:`rrm.baseline_char_ngram_lr.fit_char_ngram_logreg`.
    """
    if config is None:
        config = TfidfLogRegConfig()

    return _fit_tfidf_logreg_with_analyzer(
        train_records,
        labels=labels,
        config=config,
        analyzer="word",
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_tfidf_logreg(
    fitted: FittedTfidfLogRegBaseline,
    train_records: Iterable[Mapping[str, Any]],
    eval_records: Iterable[Mapping[str, Any]],
    *,
    labels: Tuple[str, ...] = PRIMARY_LABELS,
) -> BaselineEvaluation:
    """Evaluate a fitted TF-IDF + Logistic Regression baseline.

    Parameters
    ----------
    fitted : FittedTfidfLogRegBaseline
        Fitted baseline from fit_tfidf_logreg().
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
    # Materialize iterables ONCE so generators are not exhausted
    train_list = list(train_records)
    eval_list = list(eval_records)

    # Validate both splits (also catches missing label fields)
    train_validated, _ = _validate_records(train_list, "train", labels=labels)
    eval_validated, _ = _validate_records(eval_list, "eval", labels=labels)

    # Extract texts after validation
    eval_texts = [r["review_text"] for r in eval_validated]

    # First: validate no exact leakage (operates on validated lists)
    validate_no_exact_leakage(train_validated, eval_validated)

    # Handle empty eval
    if not eval_texts:
        skipped_labels: Dict[str, str] = dict(fitted.skipped_labels)
        for label in labels:
            if label not in fitted.models:
                skipped_labels[label] = "no fitted model"
            else:
                skipped_labels[label] = "no supervised evaluation data"
        return BaselineEvaluation(
            per_label=(),
            macro_f1=None,
            evaluated_labels=(),
            skipped_labels=skipped_labels,
        )

    # Transform eval text using the FITTED vectorizer (never refit)
    X_eval = fitted.vectorizer.transform(eval_texts)

    per_label: list = []
    evaluated_labels: list = []
    skipped_labels: Dict[str, str] = dict(fitted.skipped_labels)

    for label in labels:
        # Skip if no fitted model
        if label not in fitted.models:
            skipped_labels[label] = "no fitted model"
            continue

        y_true = numpy.array([r[label] for r in eval_validated])

        # Mask unknown rows
        known_mask = y_true != UNKNOWN_LABEL
        y_known = y_true[known_mask]
        X_eval_label = X_eval[known_mask]

        # Require supervised evaluation data
        if len(y_known) == 0:
            skipped_labels[label] = "no supervised evaluation data"
            continue

        unique_classes = numpy.unique(y_known)
        if len(unique_classes) < 2:
            skipped_labels[label] = (
                "evaluation target has only one class"
            )
            continue

        # Predict
        model = fitted.models[label]
        y_pred = model.predict(X_eval_label)
        y_proba = model.predict_proba(X_eval_label)[:, 1]

        # Compute metrics
        precision = precision_score(y_known, y_pred, zero_division=0)
        recall = recall_score(y_known, y_pred, zero_division=0)
        f1 = f1_score(y_known, y_pred, zero_division=0)
        auprc = average_precision_score(y_known, y_proba)

        neg_count = int(numpy.sum(y_known == 0))
        pos_count = int(numpy.sum(y_known == 1))

        per_label.append(
            LabelMetrics(
                label=label,
                support=len(y_known),
                negative_count=neg_count,
                positive_count=pos_count,
                precision=precision,
                recall=recall,
                f1=f1,
                auprc=auprc,
            )
        )
        evaluated_labels.append(label)

    # Macro-F1
    if per_label:
        macro_f1 = float(numpy.mean([m.f1 for m in per_label]))
    else:
        macro_f1 = None

    return BaselineEvaluation(
        per_label=tuple(per_label),
        macro_f1=macro_f1,
        evaluated_labels=tuple(evaluated_labels),
        skipped_labels=skipped_labels,
    )
