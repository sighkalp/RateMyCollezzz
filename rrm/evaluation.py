"""Evaluation utilities for the RRM (RRM 3.9).

Implements per-task metric computation with UNKNOWN masking, support
reporting, and macro metric aggregation following the RRM 3.9 scientific
reporting contract.

Research provenance:
    - Precision/Recall/F1: STANDARD ENGINEERING PRACTICE (scikit-learn)
    - AUPRC: STANDARD ENGINEERING PRACTICE (scikit-learn)
    - UNKNOWN exclusion from metrics: RRM EXPERIMENTAL DESIGN CHOICE
    - One-class task exclusion from headline macro metrics: RRM 3.9
      scientific reporting contract
    - Macro averages over valid tasks only (no zero-fill): RRM 3.9
      scientific reporting contract

Architecture rule:
    RRM UNDERSTANDS. TRUST DECIDES.

This module produces evaluation metrics only.
It does NOT make moderation decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import numpy
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from rrm.labels import NUM_PRIMARY_LABELS, PRIMARY_LABELS, UNKNOWN_LABEL


# ---------------------------------------------------------------------------
# Per-task metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskMetrics:
    """Per-task evaluation metrics.

    Attributes
    ----------
    label : str
        Label name from PRIMARY_LABELS.
    known_support : int
        Number of evaluation samples with a known (non-UNKNOWN) label.
    positive_support : int
        Number of known positive (1) samples.
    negative_support : int
        Number of known negative (0) samples.
    precision : float or None
        Precision.  None if unavailable (no known targets or only
        one ground-truth class).
    recall : float or None
        Recall.  None if unavailable.
    f1 : float or None
        F1 score.  None if unavailable.
    auprc : float or None
        Area under the precision-recall curve.  None if unavailable
        (no known positives).
    """

    label: str
    known_support: int
    positive_support: int
    negative_support: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    auprc: Optional[float]


# ---------------------------------------------------------------------------
# Macro aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MacroMetrics:
    """Aggregated macro metrics across valid tasks.

    Attributes
    ----------
    macro_f1 : float or None
        Macro-averaged F1 across tasks with valid F1.  None if no
        task has valid F1.
    macro_f1_task_count : int
        Number of tasks contributing to macro_f1.
    macro_auprc : float or None
        Macro-averaged AUPRC across tasks with valid AUPRC.  None if
        no task has valid AUPRC.
    macro_auprc_task_count : int
        Number of tasks contributing to macro_auprc.
    """

    macro_f1: Optional[float]
    macro_f1_task_count: int
    macro_auprc: Optional[float]
    macro_auprc_task_count: int


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_task_metrics(
    y_true: numpy.ndarray,
    y_pred: numpy.ndarray,
    y_proba: numpy.ndarray,
    label_name: str,
    *,
    threshold: float = 0.5,
) -> TaskMetrics:
    """Compute per-task metrics with UNKNOWN exclusion.

    UNKNOWN labels (-1) are excluded before metric computation.

    A task contributes headline discriminative metrics (precision, recall,
    F1) only when BOTH ground-truth classes are present among known
    targets.  If only one class is present, metrics are reported as None.

    AUPRC is unavailable when there are no known positives.

    Parameters
    ----------
    y_true : numpy.ndarray
        True labels.  Shape (n,).  Values: -1, 0, 1.
    y_pred : numpy.ndarray
        Predicted binary labels (after threshold).  Shape (n,).
    y_proba : numpy.ndarray
        Predicted probabilities (sigmoid scores).  Shape (n,).
    label_name : str
        Label name for the metrics container.
    threshold : float
        Decision threshold for converting probabilities to predictions.
        Default 0.5.

    Returns
    -------
    TaskMetrics
    """
    known_mask = y_true != UNKNOWN_LABEL
    known_support = int(numpy.sum(known_mask))

    if known_support == 0:
        return TaskMetrics(
            label=label_name,
            known_support=0,
            positive_support=0,
            negative_support=0,
            precision=None,
            recall=None,
            f1=None,
            auprc=None,
        )

    y_true_known = y_true[known_mask]
    y_pred_known = y_pred[known_mask]
    y_proba_known = y_proba[known_mask]

    positive_support = int(numpy.sum(y_true_known == 1))
    negative_support = int(numpy.sum(y_true_known == 0))

    # AUPRC requires at least one positive
    if positive_support == 0:
        auprc = None
    else:
        auprc = float(average_precision_score(y_true_known, y_proba_known))

    # Discriminative metrics require both classes
    if positive_support == 0 or negative_support == 0:
        return TaskMetrics(
            label=label_name,
            known_support=known_support,
            positive_support=positive_support,
            negative_support=negative_support,
            precision=None,
            recall=None,
            f1=None,
            auprc=auprc,
        )

    precision = float(precision_score(y_true_known, y_pred_known, zero_division=0))
    recall = float(recall_score(y_true_known, y_pred_known, zero_division=0))
    f1 = float(f1_score(y_true_known, y_pred_known, zero_division=0))

    return TaskMetrics(
        label=label_name,
        known_support=known_support,
        positive_support=positive_support,
        negative_support=negative_support,
        precision=precision,
        recall=recall,
        f1=f1,
        auprc=auprc,
    )


def aggregate_macro_metrics(
    task_metrics: List[TaskMetrics],
) -> MacroMetrics:
    """Aggregate macro metrics from per-task results.

    Only tasks with valid (non-None) metric values contribute to the
    macro average.  Unavailable tasks are never treated as zero.

    Parameters
    ----------
    task_metrics : list of TaskMetrics
        Per-task metrics.

    Returns
    -------
    MacroMetrics
    """
    valid_f1 = [tm.f1 for tm in task_metrics if tm.f1 is not None]
    valid_auprc = [tm.auprc for tm in task_metrics if tm.auprc is not None]

    macro_f1 = (
        float(numpy.mean(valid_f1)) if valid_f1 else None
    )
    macro_f1_task_count = len(valid_f1)

    macro_auprc = (
        float(numpy.mean(valid_auprc)) if valid_auprc else None
    )
    macro_auprc_task_count = len(valid_auprc)

    return MacroMetrics(
        macro_f1=macro_f1,
        macro_f1_task_count=macro_f1_task_count,
        macro_auprc=macro_auprc,
        macro_auprc_task_count=macro_auprc_task_count,
    )


def format_inference_signals(
    logits: torch.Tensor,
    *,
    label_names: Optional[Tuple[str, ...]] = None,
    threshold: float = 0.5,
    thresholds: Optional[Tuple[float, ...]] = None,
) -> List[dict]:
    """Format model outputs as human-readable signal dictionaries.

    For each canonical label, exposes:

    - ``label``: label name
    - ``logit``: raw logit value
    - ``sigmoid_score``: sigmoid(logit), NOT a calibrated probability

    If ``thresholds`` is provided, also exposes:

    - ``threshold``: per-task threshold
    - ``is_above_threshold``: model signal only, NOT a moderation decision

    Parameters
    ----------
    logits : torch.Tensor
        Model logits.  Shape [B, 6], dtype floating.
    label_names : tuple of str or None
        Label names.  Default PRIMARY_LABELS.
    threshold : float
        Default threshold used when ``thresholds`` is None.
    thresholds : tuple of float or None
        Per-task thresholds.  Length must match num_labels.

    Returns
    -------
    list of dict
        One dict per label.
    """
    if label_names is None:
        label_names = PRIMARY_LABELS

    if logits.dim() != 2 or logits.shape[1] != NUM_PRIMARY_LABELS:
        raise ValueError(
            f"logits must be [B, {NUM_PRIMARY_LABELS}], got shape {tuple(logits.shape)}"
        )

    if thresholds is not None:
        if len(thresholds) != NUM_PRIMARY_LABELS:
            raise ValueError(
                f"thresholds length must be {NUM_PRIMARY_LABELS}, got {len(thresholds)}"
            )
        threshold_list = list(thresholds)
    else:
        threshold_list = [threshold] * NUM_PRIMARY_LABELS

    results: List[dict] = []
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        logits_cpu = logits.detach().cpu()
        probs_cpu = probs.detach().cpu()

    for i, label in enumerate(label_names):
        entry = {
            "label": label,
            "logit": float(logits_cpu[0, i]),
            "sigmoid_score": float(probs_cpu[0, i]),
        }
        entry["threshold"] = threshold_list[i]
        # Locked comparison: >= so a score exactly equal to threshold
        # is considered at/above.  This is a model signal, NOT a
        # moderation decision.
        entry["is_above_threshold"] = bool(probs_cpu[0, i] >= threshold_list[i])
        results.append(entry)

    return results
