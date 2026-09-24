"""Tests for rrm/evaluation.py — per-task metrics and macro aggregation."""

from __future__ import annotations

import dataclasses
from dataclasses import FrozenInstanceError

import numpy
import pytest
import torch

from rrm.evaluation import (
    MacroMetrics,
    TaskMetrics,
    aggregate_macro_metrics,
    compute_task_metrics,
    format_inference_signals,
)
from rrm.labels import NUM_PRIMARY_LABELS, PRIMARY_LABELS, UNKNOWN_LABEL


# ---------------------------------------------------------------------------
# compute_task_metrics tests
# ---------------------------------------------------------------------------


class TestComputeTaskMetrics:
    """Tests for per-task metric computation."""

    def test_all_known_perfect_predictions(self):
        y_true = numpy.array([1, 1, 0, 0, 1, 0])
        y_pred = numpy.array([1, 1, 0, 0, 1, 0])
        y_proba = numpy.array([0.9, 0.8, 0.1, 0.2, 0.95, 0.05])
        tm = compute_task_metrics(y_true, y_pred, y_proba, "spam")
        assert tm.known_support == 6
        assert tm.positive_support == 3
        assert tm.negative_support == 3
        assert tm.precision == 1.0
        assert tm.recall == 1.0
        assert tm.f1 == 1.0
        assert tm.auprc == 1.0

    def test_all_unknown(self):
        y_true = numpy.array([-1, -1, -1])
        y_pred = numpy.array([0, 0, 0])
        y_proba = numpy.array([0.5, 0.5, 0.5])
        tm = compute_task_metrics(y_true, y_pred, y_proba, "deception")
        assert tm.known_support == 0
        assert tm.precision is None
        assert tm.recall is None
        assert tm.f1 is None
        assert tm.auprc is None

    def test_mixed_known_unknown(self):
        y_true = numpy.array([1, -1, 0, 1, -1, 0])
        y_pred = numpy.array([1, 0, 0, 1, 0, 0])
        y_proba = numpy.array([0.9, 0.5, 0.1, 0.8, 0.5, 0.2])
        tm = compute_task_metrics(y_true, y_pred, y_proba, "toxicity")
        assert tm.known_support == 4
        assert tm.positive_support == 2
        assert tm.negative_support == 2
        assert tm.auprc is not None

    def test_no_known_positives_auprc_none(self):
        """AUPRC unavailable when no known positives."""
        y_true = numpy.array([0, 0, 0, 0])
        y_pred = numpy.array([0, 0, 0, 0])
        y_proba = numpy.array([0.1, 0.2, 0.3, 0.4])
        tm = compute_task_metrics(y_true, y_pred, y_proba, "pii")
        assert tm.positive_support == 0
        assert tm.auprc is None
        # Discriminative metrics also unavailable with one class
        assert tm.f1 is None

    def test_one_class_only_f1_none(self):
        """F1 unavailable when only one ground-truth class present."""
        y_true = numpy.array([1, 1, 1, 1])
        y_pred = numpy.array([1, 1, 1, 1])
        y_proba = numpy.array([0.9, 0.8, 0.7, 0.6])
        tm = compute_task_metrics(y_true, y_pred, y_proba, "advertising")
        assert tm.known_support == 4
        assert tm.positive_support == 4
        assert tm.negative_support == 0
        assert tm.f1 is None  # only one class
        assert tm.precision is None
        assert tm.recall is None
        # AUPRC is available (has positives)
        assert tm.auprc is not None

    def test_label_name_preserved(self):
        y_true = numpy.array([1, 0])
        y_pred = numpy.array([1, 0])
        y_proba = numpy.array([0.9, 0.1])
        tm = compute_task_metrics(y_true, y_pred, y_proba, "off_topic")
        assert tm.label == "off_topic"

    def test_supports_reported(self):
        y_true = numpy.array([1, 1, 0, -1, 0])
        y_pred = numpy.array([1, 0, 0, 0, 1])
        y_proba = numpy.array([0.9, 0.6, 0.1, 0.5, 0.3])
        tm = compute_task_metrics(y_true, y_pred, y_proba, "spam")
        assert tm.known_support == 4
        assert tm.positive_support == 2
        assert tm.negative_support == 2

    def test_custom_threshold(self):
        """Custom threshold affects predictions."""
        y_true = numpy.array([1, 0, 1, 0])
        y_pred = numpy.array([1, 0, 1, 0])  # perfect predictions
        y_proba = numpy.array([0.7, 0.4, 0.8, 0.3])
        tm = compute_task_metrics(y_true, y_pred, y_proba, "spam", threshold=0.6)
        assert tm.precision is not None
        assert tm.f1 == 1.0

    def test_frozen_dataclass(self):
        y_true = numpy.array([1, 0])
        y_pred = numpy.array([1, 0])
        y_proba = numpy.array([0.9, 0.1])
        tm = compute_task_metrics(y_true, y_pred, y_proba, "spam")
        with pytest.raises(FrozenInstanceError):
            tm.f1 = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# aggregate_macro_metrics tests
# ---------------------------------------------------------------------------


class TestAggregateMacroMetrics:
    def test_all_valid(self):
        tasks = [
            _make_task_metrics("spam", f1=0.8, auprc=0.85),
            _make_task_metrics("deception", f1=0.6, auprc=0.70),
            _make_task_metrics("toxicity", f1=0.9, auprc=0.92),
        ]
        macro = aggregate_macro_metrics(tasks)
        assert macro.macro_f1 == pytest.approx(0.7667, abs=1e-4)
        assert macro.macro_f1_task_count == 3
        assert macro.macro_auprc == pytest.approx(0.8233, abs=1e-4)
        assert macro.macro_auprc_task_count == 3

    def test_some_none_f1(self):
        """None F1 values are excluded, not treated as zero."""
        tasks = [
            _make_task_metrics("spam", f1=0.8, auprc=0.85),
            _make_task_metrics("deception", f1=None, auprc=None),
            _make_task_metrics("toxicity", f1=0.9, auprc=0.92),
        ]
        macro = aggregate_macro_metrics(tasks)
        assert macro.macro_f1 == pytest.approx(0.85)
        assert macro.macro_f1_task_count == 2
        assert macro.macro_auprc == pytest.approx(0.885)
        assert macro.macro_auprc_task_count == 2

    def test_all_none_f1(self):
        tasks = [
            _make_task_metrics("spam", f1=None, auprc=None),
            _make_task_metrics("deception", f1=None, auprc=None),
        ]
        macro = aggregate_macro_metrics(tasks)
        assert macro.macro_f1 is None
        assert macro.macro_f1_task_count == 0
        assert macro.macro_auprc is None
        assert macro.macro_auprc_task_count == 0

    def test_partial_auprc(self):
        """Some tasks have AUPRC, others don't."""
        tasks = [
            _make_task_metrics("spam", f1=0.8, auprc=0.85),
            _make_task_metrics("deception", f1=0.6, auprc=None),
            _make_task_metrics("toxicity", f1=0.9, auprc=0.92),
        ]
        macro = aggregate_macro_metrics(tasks)
        assert macro.macro_auprc == pytest.approx((0.85 + 0.92) / 2)
        assert macro.macro_auprc_task_count == 2

    def test_frozen(self):
        tasks = [_make_task_metrics("spam", f1=0.8, auprc=0.85)]
        macro = aggregate_macro_metrics(tasks)
        with pytest.raises(FrozenInstanceError):
            macro.macro_f1 = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# format_inference_signals tests
# ---------------------------------------------------------------------------


class TestFormatInferenceSignals:
    def test_basic_format(self):
        logits = torch.zeros(1, 6)
        results = format_inference_signals(logits)
        assert len(results) == 6
        for i, entry in enumerate(results):
            assert entry["label"] == PRIMARY_LABELS[i]
            assert "logit" in entry
            assert "sigmoid_score" in entry
            assert "threshold" in entry
            assert "is_above_threshold" in entry

    def test_sigmoid_score_range(self):
        logits = torch.zeros(1, 6)
        results = format_inference_signals(logits)
        for entry in results:
            assert 0.0 <= entry["sigmoid_score"] <= 1.0

    def test_default_threshold(self):
        logits = torch.zeros(1, 6)
        results = format_inference_signals(logits, threshold=0.5)
        for entry in results:
            assert entry["threshold"] == 0.5
            # sigmoid(0) = 0.5, >= 0.5 is True (locked comparison)
            assert entry["is_above_threshold"] is True

    def test_per_task_thresholds(self):
        logits = torch.zeros(1, 6)
        thresholds = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
        results = format_inference_signals(logits, thresholds=thresholds)
        for i, entry in enumerate(results):
            assert entry["threshold"] == thresholds[i]

    def test_per_task_thresholds_wrong_length(self):
        logits = torch.zeros(1, 6)
        with pytest.raises(ValueError, match="length must be 6"):
            format_inference_signals(logits, thresholds=(0.5, 0.5))

    def test_wrong_shape(self):
        logits = torch.zeros(1, 3)
        with pytest.raises(ValueError, match="\[B, 6\]"):
            format_inference_signals(logits)

    def test_batch_size_greater_than_one(self):
        logits = torch.zeros(2, 6)
        results = format_inference_signals(logits)
        assert len(results) == 6  # Still 6 labels
        for entry in results:
            assert isinstance(entry["logit"], float)

    def test_negative_logit(self):
        """Negative logit → sigmoid_score < 0.5 → not above default threshold."""
        logits = torch.full((1, 6), -10.0)
        results = format_inference_signals(logits)
        for entry in results:
            assert entry["sigmoid_score"] < 0.5
            assert entry["is_above_threshold"] is False

    def test_positive_logit(self):
        """Positive logit → sigmoid_score > 0.5 → above default threshold."""
        logits = torch.full((1, 6), 10.0)
        results = format_inference_signals(logits)
        for entry in results:
            assert entry["sigmoid_score"] > 0.5
            assert entry["is_above_threshold"] is True

    def test_threshold_comparison_locked_greater_equal(self):
        """Locked comparison: sigmoid_score >= threshold (not >).
        A score exactly equal to threshold must be considered at/above."""
        # sigmoid(logit) = 0.5 when logit = 0.0
        logits = torch.zeros(1, 6)
        results = format_inference_signals(logits, threshold=0.5)
        for entry in results:
            # sigmoid(0) = 0.5 exactly, and 0.5 >= 0.5 is True
            assert entry["sigmoid_score"] == 0.5
            assert entry["is_above_threshold"] is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task_metrics(
    label: str,
    known_support: int = 10,
    positive_support: int = 5,
    negative_support: int = 5,
    precision: float | None = 0.8,
    recall: float | None = 0.8,
    f1: float | None = 0.8,
    auprc: float | None = 0.8,
) -> TaskMetrics:
    return TaskMetrics(
        label=label,
        known_support=known_support,
        positive_support=positive_support,
        negative_support=negative_support,
        precision=precision,
        recall=recall,
        f1=f1,
        auprc=auprc,
    )
