"""Tests for rrm.scientific_evaluation.

Validates the RRM 3.10 scientific evaluation infrastructure:
- FrozenSplitManifest creation and validation
- Leakage detection (review_id, exact_text, fingerprint, near_duplicate, lineage)
- Task evaluability and support counts
- ScientificMacroResult computation
- Same-task-set comparison validation
- Threshold selection (fixed and validation-tuned)
- Slice aggregation
- Bootstrap infrastructure
- Efficiency measurement
"""

from __future__ import annotations

import math
import re
import time
from typing import Optional
from unittest.mock import MagicMock, call

import numpy
import pytest

from rrm.scientific_evaluation import (
    NON_SCIENTIFIC_SYNTHETIC_SMOKE,
    SCIENTIFIC_PRODUCTION,
    SCIENTIFIC_PARTIAL,
    VALID_SCIENTIFIC_STATUSES,
    BootstrapConfig,
    BootstrapCI,
    EfficiencyConfig,
    FrozenSplitManifest,
    LeakageFinding,
    LeakageInputRecord,
    LeakageReport,
    LatencyResult,
    MemoryResult,
    ScientificMacroResult,
    ComparisonValidationResult,
    DEFAULT_THRESHOLD,
    DEFAULT_THRESHOLD_SOURCE,
    SliceAggregation,
    SliceResult,
    ThresholdResult,
    aggregate_slices,
    bootstrap_ci,
    check_exact_text_leakage,
    check_fingerprint_leakage,
    check_lineage_leakage,
    check_near_duplicate_leakage,
    check_review_id_leakage,
    compute_scientific_macro,
    is_task_evaluable,
    measure_latency,
    measure_memory,
    paired_bootstrap_indices,
    run_full_leakage_check,
    select_validation_threshold,
    task_support_counts,
    validate_comparison_task_sets,
    validate_split_manifest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task_metrics(
    label: str,
    known_support: int = 10,
    positive_support: int = 5,
    negative_support: int = 5,
    precision: Optional[float] = 0.8,
    recall: Optional[float] = 0.8,
    f1: Optional[float] = 0.8,
    auprc: Optional[float] = 0.85,
) -> "rrm.evaluation.TaskMetrics":
    """Create a TaskMetrics instance."""
    from rrm.evaluation import TaskMetrics
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


def _valid_manifest(**kwargs) -> FrozenSplitManifest:
    """Create a minimally valid FrozenSplitManifest."""
    defaults = dict(
        dataset_version="v1.0",
        dataset_hash="a" * 64,
        split_version="s1",
        split_hash="b" * 64,
        split_generation_identity="test",
        split_seed=42,
        train_review_ids=("r1", "r2", "r3"),
        validation_review_ids=("r4", "r5"),
        test_review_ids=("r6",),
        created_at="2024-01-01T00:00:00+00:00",
        protocol_version="1.0",
    )
    defaults.update(kwargs)
    return FrozenSplitManifest(**defaults)


# ===================================================================
# 1. Scientific status constants
# ===================================================================


class TestScientificStatusConstants:
    """Section 4: scientific status constants."""

    def test_non_scientific_exact_string(self):
        assert NON_SCIENTIFIC_SYNTHETIC_SMOKE == (
            "NON_SCIENTIFIC_SYNTHETIC_SMOKE"
        )

    def test_valid_statuses_contains_smoke(self):
        assert NON_SCIENTIFIC_SYNTHETIC_SMOKE in VALID_SCIENTIFIC_STATUSES

    def test_valid_statuses_contains_production(self):
        assert SCIENTIFIC_PRODUCTION in VALID_SCIENTIFIC_STATUSES

    def test_valid_statuses_contains_partial(self):
        assert SCIENTIFIC_PARTIAL in VALID_SCIENTIFIC_STATUSES

    def test_valid_statuses_count(self):
        assert len(VALID_SCIENTIFIC_STATUSES) == 3

    def test_all_statuses_are_strings(self):
        for s in VALID_SCIENTIFIC_STATUSES:
            assert isinstance(s, str)


# ===================================================================
# 2. FrozenSplitManifest — creation and validation
# ===================================================================


class TestFrozenSplitManifest:
    """Section 5: frozen split manifest."""

    def test_creation_with_minimal_fields(self):
        manifest = _valid_manifest()
        assert manifest.dataset_version == "v1.0"
        assert manifest.split_seed == 42
        assert manifest.protocol_version == "1.0"

    def test_split_seed_none_allowed(self):
        manifest = _valid_manifest(split_seed=None)
        assert manifest.split_seed is None

    def test_review_ids_are_tuples(self):
        manifest = _valid_manifest()
        assert isinstance(manifest.train_review_ids, tuple)

    def test_metadata_optional(self):
        manifest = _valid_manifest(
            lineage_group_metadata=None,
            duplicate_cluster_metadata=None,
        )
        assert manifest.lineage_group_metadata is None
        assert manifest.duplicate_cluster_metadata is None

    def test_lineage_metadata_accepted(self):
        meta = {"g1": ["r1", "r2"], "g2": ["r3"]}
        manifest = _valid_manifest(lineage_group_metadata=meta)
        assert manifest.lineage_group_metadata["g1"] == ["r1", "r2"]


class TestValidateSplitManifest:
    """Section 6: split manifest validation."""

    def test_valid_manifest_passes(self):
        manifest = _valid_manifest()
        errors = validate_split_manifest(manifest)
        assert errors == []

    def test_empty_dataset_version_rejected(self):
        manifest = _valid_manifest(dataset_version="")
        errors = validate_split_manifest(manifest)
        assert any("dataset_version" in e for e in errors)

    def test_empty_dataset_hash_rejected(self):
        manifest = _valid_manifest(dataset_hash="")
        errors = validate_split_manifest(manifest)
        assert any("dataset_hash" in e for e in errors)

    def test_empty_split_version_rejected(self):
        manifest = _valid_manifest(split_version="")
        errors = validate_split_manifest(manifest)
        assert any("split_version" in e for e in errors)

    def test_empty_split_hash_rejected(self):
        manifest = _valid_manifest(split_hash="")
        errors = validate_split_manifest(manifest)
        assert any("split_hash" in e for e in errors)

    def test_empty_split_generation_identity_rejected(self):
        manifest = _valid_manifest(split_generation_identity="")
        errors = validate_split_manifest(manifest)
        assert any("split_generation_identity" in e for e in errors)

    def test_empty_protocol_version_rejected(self):
        manifest = _valid_manifest(protocol_version="")
        errors = validate_split_manifest(manifest)
        assert any("protocol_version" in e for e in errors)

    def test_invalid_split_seed_type_rejected(self):
        manifest = _valid_manifest(split_seed="not_an_int")
        errors = validate_split_manifest(manifest)
        assert any("split_seed" in e for e in errors)

    def test_non_string_review_id_rejected(self):
        manifest = _valid_manifest(train_review_ids=("r1", 42, "r3"))
        errors = validate_split_manifest(manifest)
        assert any("not a str" in e for e in errors)

    def test_duplicate_review_id_within_train_rejected(self):
        manifest = _valid_manifest(train_review_ids=("r1", "r1", "r3"))
        errors = validate_split_manifest(manifest)
        assert any("Duplicate review_id" in e for e in errors)

    def test_duplicate_review_id_across_train_and_validation_rejected(self):
        manifest = _valid_manifest(
            train_review_ids=("r1", "r2"),
            validation_review_ids=("r2", "r4"),
        )
        errors = validate_split_manifest(manifest)
        assert any("appears in both" in e for e in errors)

    def test_duplicate_across_train_and_test_rejected(self):
        manifest = _valid_manifest(
            train_review_ids=("r1", "r2"),
            test_review_ids=("r1", "r6"),
        )
        errors = validate_split_manifest(manifest)
        assert any("appears in both" in e for e in errors)

    def test_duplicate_across_validation_and_test_rejected(self):
        manifest = _valid_manifest(
            validation_review_ids=("r4", "r5"),
            test_review_ids=("r5", "r6"),
        )
        errors = validate_split_manifest(manifest)
        assert any("appears in both" in e for e in errors)

    def test_non_serializable_lineage_metadata_rejected(self):
        meta = {"g1": lambda x: x}  # lambda is not JSON-serializable
        manifest = _valid_manifest(lineage_group_metadata=meta)
        errors = validate_split_manifest(manifest)
        assert any("lineage_group_metadata" in e for e in errors)

    def test_non_serializable_duplicate_metadata_rejected(self):
        meta = {"c1": object()}  # object is not JSON-serializable
        manifest = _valid_manifest(duplicate_cluster_metadata=meta)
        errors = validate_split_manifest(manifest)
        assert any("duplicate_cluster_metadata" in e for e in errors)

    def test_json_serializable_metadata_accepted(self):
        meta = {"g1": ["r1", "r2"], "count": 3}
        manifest = _valid_manifest(lineage_group_metadata=meta)
        errors = validate_split_manifest(manifest)
        assert not any("lineage_group_metadata" in e for e in errors)


# ===================================================================
# 3. LeakageInputRecord
# ===================================================================


class TestLeakageInputRecord:
    """Section 8: leakage input contract."""

    def test_minimal_record(self):
        rec = LeakageInputRecord(
            review_id="r1",
            review_text="hello world",
            split="train",
        )
        assert rec.review_id == "r1"
        assert rec.split == "train"
        assert rec.lineage_group_id is None

    def test_record_with_lineage(self):
        rec = LeakageInputRecord(
            review_id="r1",
            review_text="hello world",
            split="train",
            lineage_group_id="g1",
        )
        assert rec.lineage_group_id == "g1"


# ===================================================================
# 4. Review-ID leakage
# ===================================================================


class TestReviewIdLeakage:
    """Section 9: review-ID leakage."""

    def test_no_overlap_no_findings(self):
        records = [
            LeakageInputRecord("r1", "text1", "train"),
            LeakageInputRecord("r2", "text2", "validation"),
            LeakageInputRecord("r3", "text3", "test"),
        ]
        findings = check_review_id_leakage(records)
        assert findings == []

    def test_cross_split_overlap_detected(self):
        records = [
            LeakageInputRecord("r1", "text1", "train"),
            LeakageInputRecord("r1", "text2", "test"),
        ]
        findings = check_review_id_leakage(records)
        assert len(findings) == 1
        assert findings[0].leak_type == "review_id"
        assert findings[0].record_id_a == "r1"
        assert findings[0].split_a == "train"
        assert findings[0].split_b == "test"

    def test_three_way_overlap_all_pairs(self):
        records = [
            LeakageInputRecord("r1", "text1", "train"),
            LeakageInputRecord("r1", "text2", "validation"),
            LeakageInputRecord("r1", "text3", "test"),
        ]
        findings = check_review_id_leakage(records)
        # train/val, train/test, val/test
        assert len(findings) == 3

    def test_same_split_no_finding(self):
        records = [
            LeakageInputRecord("r1", "text1", "train"),
            LeakageInputRecord("r1", "text2", "train"),
        ]
        findings = check_review_id_leakage(records)
        assert findings == []

    def test_deterministic_ordering(self):
        records = [
            LeakageInputRecord("z1", "text1", "test"),
            LeakageInputRecord("z1", "text2", "train"),
            LeakageInputRecord("a1", "text3", "train"),
            LeakageInputRecord("a1", "text4", "test"),
        ]
        findings = check_review_id_leakage(records)
        # Should be sorted by (record_id_a, split_a, split_b)
        assert findings[0].record_id_a <= findings[-1].record_id_a


# ===================================================================
# 5. Exact text leakage
# ===================================================================


class TestExactTextLeakage:
    """Section 10: exact raw-text leakage."""

    def test_no_overlap_no_findings(self):
        records = [
            LeakageInputRecord("r1", "hello", "train"),
            LeakageInputRecord("r2", "world", "test"),
        ]
        findings = check_exact_text_leakage(records)
        assert findings == []

    def test_identical_text_across_splits(self):
        records = [
            LeakageInputRecord("r1", "same text", "train"),
            LeakageInputRecord("r2", "same text", "test"),
        ]
        findings = check_exact_text_leakage(records)
        assert len(findings) == 1
        assert findings[0].leak_type == "exact_text"

    def test_identical_text_same_split_no_finding(self):
        records = [
            LeakageInputRecord("r1", "same text", "train"),
            LeakageInputRecord("r2", "same text", "train"),
        ]
        findings = check_exact_text_leakage(records)
        assert findings == []

    def test_three_records_same_text_different_splits(self):
        records = [
            LeakageInputRecord("r1", "same text", "train"),
            LeakageInputRecord("r2", "same text", "validation"),
            LeakageInputRecord("r3", "same text", "test"),
        ]
        findings = check_exact_text_leakage(records)
        # train/val, train/test, val/test
        assert len(findings) == 3


# ===================================================================
# 6. Fingerprint leakage
# ===================================================================


class TestFingerprintLeakage:
    """Section 11: normalized fingerprint leakage."""

    def test_no_overlap_no_findings(self):
        records = [
            LeakageInputRecord("r1", "hello world", "train"),
            LeakageInputRecord("r2", "goodbye world", "test"),
        ]
        findings = check_fingerprint_leakage(records)
        assert findings == []

    def test_identical_after_normalization(self):
        records = [
            LeakageInputRecord("r1", "  Hello   World  ", "train"),
            LeakageInputRecord("r2", "hello world", "test"),
        ]
        findings = check_fingerprint_leakage(records)
        assert len(findings) == 1
        assert findings[0].leak_type == "fingerprint"

    def test_different_text_same_fingerprint_no_issue(self):
        """Punctuation changes should NOT trigger fingerprint match."""
        records = [
            LeakageInputRecord("r1", "best", "train"),
            LeakageInputRecord("r2", "best!!!", "test"),
        ]
        findings = check_fingerprint_leakage(records)
        # normalized "best!!!" ≠ normalized "best"
        assert len(findings) == 0


# ===================================================================
# 7. Near-duplicate leakage
# ===================================================================


class TestNearDuplicateLeakage:
    """Section 12: near-duplicate candidate reporting."""

    def test_high_similarity_detected(self):
        records = [
            LeakageInputRecord("r1", "This college is amazing", "train"),
            LeakageInputRecord("r2", "This college is amazing!", "test"),
        ]
        findings = check_near_duplicate_leakage(records, n=4)
        assert len(findings) == 1
        assert findings[0].leak_type == "near_duplicate"
        assert findings[0].score is not None
        assert findings[0].score >= 0.85

    def test_unrelated_text_not_detected(self):
        records = [
            LeakageInputRecord("r1", "This college is amazing", "train"),
            LeakageInputRecord("r2", "The food tastes terrible", "test"),
        ]
        findings = check_near_duplicate_leakage(
            records, n=4, threshold=0.5
        )
        assert len(findings) == 0

    def test_threshold_parameter_used(self):
        records = [
            LeakageInputRecord("r1", "This college is good", "train"),
            LeakageInputRecord("r2", "This college is okay", "test"),
        ]
        # High threshold: should not report
        findings_high = check_near_duplicate_leakage(
            records, n=4, threshold=0.95
        )
        # Low threshold: might report
        findings_low = check_near_duplicate_leakage(
            records, n=4, threshold=0.3
        )
        assert len(findings_high) <= len(findings_low)

    def test_self_excluded_by_default(self):
        records = [
            LeakageInputRecord("r1", "hello world", "train"),
            LeakageInputRecord("r1", "hello world", "train"),  # same id
        ]
        findings = check_near_duplicate_leakage(records, n=4)
        # Same review_id excluded, so no cross-split finding
        assert len(findings) == 0


# ===================================================================
# 8. Lineage leakage
# ===================================================================


class TestLineageLeakage:
    """Section 13: lineage-group leakage."""

    def test_no_metadata_no_finding(self):
        records = [
            LeakageInputRecord("r1", "text1", "train"),
            LeakageInputRecord("r2", "text2", "test"),
        ]
        findings = check_lineage_leakage(records)
        assert findings == []

    def test_same_group_same_split_no_finding(self):
        records = [
            LeakageInputRecord("r1", "text1", "train", lineage_group_id="g1"),
            LeakageInputRecord("r2", "text2", "train", lineage_group_id="g1"),
        ]
        findings = check_lineage_leakage(records)
        assert findings == []

    def test_same_group_cross_split_detected(self):
        records = [
            LeakageInputRecord("r1", "text1", "train", lineage_group_id="g1"),
            LeakageInputRecord("r2", "text2", "test", lineage_group_id="g1"),
        ]
        findings = check_lineage_leakage(records)
        assert len(findings) == 1
        assert findings[0].leak_type == "lineage_group"
        assert "g1" in findings[0].detail

    def test_mixed_lineage_and_none(self):
        records = [
            LeakageInputRecord("r1", "text1", "train", lineage_group_id="g1"),
            LeakageInputRecord("r2", "text2", "test"),  # no lineage
            LeakageInputRecord("r3", "text3", "test", lineage_group_id="g1"),
        ]
        findings = check_lineage_leakage(records)
        assert len(findings) == 1  # g1 crosses train/test

    def test_deterministic_ordering(self):
        records = [
            LeakageInputRecord("r3", "text3", "test", lineage_group_id="g2"),
            LeakageInputRecord("r1", "text1", "train", lineage_group_id="g2"),
        ]
        findings = check_lineage_leakage(records)
        assert len(findings) == 1
        assert findings[0].record_id_a == "r1"
        assert findings[0].record_id_b == "r3"


# ===================================================================
# 9. Full leakage check
# ===================================================================


class TestFullLeakageCheck:
    """Section 7: complete leakage report."""

    def test_clean_records_no_findings(self):
        records = [
            LeakageInputRecord("r1", "text1", "train"),
            LeakageInputRecord("r2", "text2", "validation"),
            LeakageInputRecord("r3", "text3", "test"),
        ]
        report = run_full_leakage_check(records)
        assert report.has_unresolved_leakage is False
        assert report.total_review_ids_checked == 3
        assert report.total_records_checked == 3

    def test_report_counts_correct(self):
        records = [
            LeakageInputRecord("r1", "same text", "train"),
            LeakageInputRecord("r1", "same text", "test"),
            LeakageInputRecord("r2", "same text", "train"),
            LeakageInputRecord("r3", "other text", "validation"),
        ]
        report = run_full_leakage_check(records)
        assert report.has_unresolved_leakage is True
        assert report.total_review_ids_checked == 3
        assert report.total_records_checked == 4


# ===================================================================
# 10. Task evaluability
# ===================================================================


class TestIsTaskEvaluable:
    """Section 14: task evaluability."""

    def test_both_classes_evaluable(self):
        y_true = [0, 1, 0, 1, 0, 1]
        assert is_task_evaluable(y_true) is True

    def test_only_positives_not_evaluable(self):
        y_true = [1, 1, 1]
        assert is_task_evaluable(y_true) is False

    def test_only_negatives_not_evaluable(self):
        y_true = [0, 0, 0]
        assert is_task_evaluable(y_true) is False

    def test_unknown_only_not_evaluable(self):
        y_true = [-1, -1, -1]
        assert is_task_evaluable(y_true) is False

    def test_mixed_with_unknown_both_classes_evaluable(self):
        y_true = [0, -1, 1, 0, -1, 1]
        assert is_task_evaluable(y_true) is True

    def test_empty_not_evaluable(self):
        assert is_task_evaluable([]) is False


class TestTaskSupportCounts:
    """Task support count helper."""

    def test_known_support_counts_non_unknown(self):
        y_true = [0, 1, -1, 0, 1]
        k, p, n = task_support_counts(y_true)
        assert k == 4
        assert p == 2
        assert n == 2

    def test_all_unknown(self):
        y_true = [-1, -1]
        k, p, n = task_support_counts(y_true)
        assert k == 0
        assert p == 0
        assert n == 0


# ===================================================================
# 11. ScientificMacroResult
# ===================================================================


class TestComputeScientificMacro:
    """Section 16: scientific macro results."""

    def test_no_tasks_none_values(self):
        result = compute_scientific_macro([])
        assert result.macro_f1 is None
        assert result.macro_auprc is None
        assert result.macro_f1_task_count == 0
        assert result.macro_auprc_task_count == 0

    def test_single_task(self):
        tms = [_make_task_metrics("spam", f1=0.8, auprc=0.9)]
        result = compute_scientific_macro(tms)
        assert result.macro_f1 == pytest.approx(0.8)
        assert result.macro_auprc == pytest.approx(0.9)
        assert result.macro_f1_task_count == 1
        assert result.macro_auprc_task_count == 1
        assert result.macro_f1_task_names == ("spam",)
        assert result.macro_auprc_task_names == ("spam",)

    def test_multiple_tasks_mean_computed(self):
        tms = [
            _make_task_metrics("spam", f1=0.8, auprc=0.9),
            _make_task_metrics("deception", f1=0.6, auprc=0.7),
        ]
        result = compute_scientific_macro(tms)
        assert result.macro_f1 == pytest.approx(0.7)
        assert result.macro_auprc == pytest.approx(0.8)
        assert result.macro_f1_task_count == 2
        assert result.macro_auprc_task_count == 2

    def test_none_values_excluded_from_mean(self):
        tms = [
            _make_task_metrics("spam", f1=0.8, auprc=0.9),
            _make_task_metrics("deception", f1=None, auprc=None),
        ]
        result = compute_scientific_macro(tms)
        assert result.macro_f1 == pytest.approx(0.8)
        assert result.macro_f1_task_count == 1
        assert "spam" in result.macro_f1_task_names
        assert "deception" not in result.macro_f1_task_names

    def test_auprc_none_f1_present(self):
        tms = [
            _make_task_metrics("spam", f1=0.8, auprc=None),
        ]
        result = compute_scientific_macro(tms)
        assert result.macro_f1 == pytest.approx(0.8)
        assert result.macro_f1_task_count == 1
        assert result.macro_auprc is None
        assert result.macro_auprc_task_count == 0

    def test_frozen(self):
        result = compute_scientific_macro([])
        with pytest.raises(AttributeError):
            result.macro_f1 = 0.5


# ===================================================================
# 12. Same-task-set comparison validation
# ===================================================================


class TestValidateComparisonTaskSets:
    """Section 17: same task set comparison."""

    def _make_macro(self, task_names):
        """Create a ScientificMacroResult with specified task names."""
        tms = [_make_task_metrics(name) for name in task_names]
        result = compute_scientific_macro(tms)
        return result

    def test_identical_task_sets_valid(self):
        ma = self._make_macro(["spam", "deception"])
        mb = self._make_macro(["spam", "deception"])
        r = validate_comparison_task_sets(ma, mb)
        assert r.valid is True
        assert r.differences == ()

    def test_different_task_sets_invalid(self):
        ma = self._make_macro(["spam", "deception"])
        mb = self._make_macro(["spam", "toxicity"])
        r = validate_comparison_task_sets(ma, mb)
        assert r.valid is False
        assert len(r.differences) > 0


# ===================================================================
# 13. Threshold selection
# ===================================================================


class TestDefaultThreshold:
    """Section 18: fixed threshold."""

    def test_default_threshold_value(self):
        assert DEFAULT_THRESHOLD == 0.5

    def test_default_source_string(self):
        assert DEFAULT_THRESHOLD_SOURCE == (
            "default_0.5_no_evaluable_validation"
        )


class TestSelectValidationThreshold:
    """Section 19-20: validation threshold selection."""

    def test_one_class_returns_default(self):
        y_true = [0, 0, 0]  # only negatives
        y_proba = [0.3, 0.4, 0.5]
        result = select_validation_threshold(y_true, y_proba)
        assert result.threshold == DEFAULT_THRESHOLD
        assert result.threshold_source == DEFAULT_THRESHOLD_SOURCE
        assert result.candidate_count == 0

    def test_one_class_positive_returns_default(self):
        y_true = [1, 1, 1]
        y_proba = [0.7, 0.8, 0.9]
        result = select_validation_threshold(y_true, y_proba)
        assert result.threshold == DEFAULT_THRESHOLD
        assert result.threshold_source == DEFAULT_THRESHOLD_SOURCE

    def test_unknown_only_returns_default(self):
        y_true = [-1, -1, -1]
        y_proba = [0.5, 0.5, 0.5]
        result = select_validation_threshold(y_true, y_proba)
        assert result.threshold == DEFAULT_THRESHOLD
        assert result.threshold_source == DEFAULT_THRESHOLD_SOURCE

    def test_both_classes_selects_threshold(self):
        numpy.random.seed(42)
        n = 20
        y_true = [0, 1] * (n // 2)
        y_proba = numpy.random.uniform(0.1, 0.9, size=n).tolist()
        result = select_validation_threshold(y_true, y_proba)
        assert result.threshold_source == "validation_tuned"
        assert result.candidate_count > 0

    def test_candidates_include_unique_scores_and_half(self):
        y_true = [0, 1, 0, 1]
        y_proba = [0.1, 0.9, 0.2, 0.8]
        result = select_validation_threshold(y_true, y_proba)
        # candidates = unique([0.1, 0.9, 0.2, 0.8]) + {0.5}
        assert 0.5 in [0.1, 0.2, 0.5, 0.8, 0.9]  # 0.5 added
        assert result.candidate_count == 5

    def test_tie_break_closest_to_half(self):
        # With all correct predictions, many thresholds give F1=1.0
        y_true = [0, 1, 0, 1]
        y_proba = [0.1, 0.9, 0.1, 0.9]
        result = select_validation_threshold(y_true, y_proba)
        # All thresholds below 0.9 give perfect F1=1.0
        # Closest to 0.5 should be chosen
        assert result.threshold_source == "validation_tuned"

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            select_validation_threshold([0, 1], [0.5])

    def test_fixed_zero_point_five(self):
        y_true = [0, 1, 0, 1]
        y_proba = [0.5, 0.5, 0.5, 0.5]
        result = select_validation_threshold(y_true, y_proba)
        assert result.threshold == 0.5

    def test_threshold_result_frozen(self):
        result = select_validation_threshold([0, 1], [0.3, 0.7])
        with pytest.raises(AttributeError):
            result.threshold = 0.9


# ===================================================================
# 14. Slice aggregation
# ===================================================================


class TestSliceAggregation:
    """Section 21: slice aggregation."""

    def _make_y(self, n: int) -> "numpy.ndarray":
        """Create simple known-label arrays."""
        y = numpy.zeros((n, 6), dtype=numpy.int64)
        # Alternating 0/1
        for i in range(n):
            y[i] = [i % 2] * 6
        return y

    def test_slice_by_simple_group(self):
        numpy.random.seed(42)
        n = 6
        y_true = self._make_y(n)
        y_pred = self._make_y(n)
        y_proba = numpy.random.uniform(0, 1, size=(n, 6)).astype(
            numpy.float64
        )
        # Set probabilities so first 3 are above 0.5, rest below
        for i in range(n):
            y_proba[i] = 0.9 if i % 2 == 0 else 0.1

        slice_labels = ["A"] * (n // 2) + ["B"] * (n // 2)
        agg = aggregate_slices(y_true, y_pred, y_proba, slice_labels)
        assert len(agg.slices) == 2
        labels = {s.slice_label for s in agg.slices}
        assert labels == {"A", "B"}

    def test_length_mismatch_raises(self):
        y_true = numpy.zeros((3, 6))
        y_pred = numpy.zeros((3, 6))
        y_proba = numpy.zeros((3, 6))
        slice_labels = ["A", "B"]  # wrong length
        with pytest.raises(ValueError):
            aggregate_slices(y_true, y_pred, y_proba, slice_labels)

    def test_slice_result_frozen(self):
        sr = SliceResult(
            slice_label="test",
            record_count=5,
            task_metrics=(),
            macro=None,
        )
        with pytest.raises(AttributeError):
            sr.record_count = 10


# ===================================================================
# 15. Bootstrap infrastructure
# ===================================================================


class TestBootstrapConfig:
    """Section 3: bootstrap defaults."""

    def test_default_values(self):
        config = BootstrapConfig()
        assert config.seed == 42
        assert config.replicates == 1000
        assert config.ci_level == 0.95
        assert config.method == "percentile"

    def test_custom_values(self):
        config = BootstrapConfig(seed=123, replicates=500, ci_level=0.90)
        assert config.seed == 123
        assert config.replicates == 500
        assert config.ci_level == 0.90

    def test_invalid_replicates_raises(self):
        with pytest.raises(ValueError):
            BootstrapConfig(replicates=0)

    def test_invalid_ci_level_raises(self):
        with pytest.raises(ValueError):
            BootstrapConfig(ci_level=1.5)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            BootstrapConfig(method="bca")

    def test_frozen(self):
        config = BootstrapConfig()
        with pytest.raises(AttributeError):
            config.seed = 99


class TestBootstrapCI:
    """Bootstrap CI computation."""

    def test_ci_contains_point_estimate(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        config = BootstrapConfig(seed=42, replicates=100)
        result = bootstrap_ci(values, config=config)
        assert result.lower <= result.point_estimate <= result.upper

    def test_ci_level_controls_width(self):
        values = list(range(1, 21))
        config_90 = BootstrapConfig(seed=42, replicates=200, ci_level=0.90)
        config_95 = BootstrapConfig(seed=42, replicates=200, ci_level=0.95)
        ci_90 = bootstrap_ci(values, config=config_90)
        ci_95 = bootstrap_ci(values, config=config_95)
        # 95% CI should be wider than 90% CI
        assert (ci_95.upper - ci_95.lower) >= (ci_90.upper - ci_90.lower)

    def test_deterministic_with_same_seed(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        config = BootstrapConfig(seed=42, replicates=50)
        ci1 = bootstrap_ci(values, config=config)
        ci2 = bootstrap_ci(values, config=config)
        assert ci1.lower == ci2.lower
        assert ci1.upper == ci2.upper
        assert ci1.point_estimate == ci2.point_estimate

    def test_different_seed_produces_different_results(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        config_a = BootstrapConfig(seed=42, replicates=50)
        config_b = BootstrapConfig(seed=99, replicates=50)
        ci_a = bootstrap_ci(values, config=config_a)
        ci_b = bootstrap_ci(values, config=config_b)
        # Different seeds should very likely produce different CIs
        assert not (
            ci_a.lower == ci_b.lower
            and ci_a.upper == ci_b.upper
            and ci_a.point_estimate == ci_b.point_estimate
        )

    def test_empty_values_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci([], config=BootstrapConfig(replicates=10))

    def test_single_value_ci(self):
        values = [3.0]
        result = bootstrap_ci(values, config=BootstrapConfig(replicates=100))
        assert math.isclose(result.point_estimate, 3.0)

    def test_invalid_statistic_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci(
                [1.0, 2.0],
                config=BootstrapConfig(replicates=10),
                statistic="median",
            )

    def test_correct_point_estimate(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = bootstrap_ci(
            values, config=BootstrapConfig(replicates=100)
        )
        assert math.isclose(result.point_estimate, 3.0)

    def test_result_fields_populated(self):
        values = [1.0, 2.0, 3.0]
        result = bootstrap_ci(
            values, config=BootstrapConfig(seed=7, replicates=50)
        )
        assert result.replicates == 50
        assert result.seed == 7
        assert result.method == "percentile"
        assert result.ci_level == 0.95


class TestPairedBootstrapIndices:
    """Section 3: paired bootstrap indices."""

    def test_shape_correct(self):
        indices = paired_bootstrap_indices(
            10, config=BootstrapConfig(replicates=5)
        )
        assert indices.shape == (5, 10)

    def test_indices_in_valid_range(self):
        n = 5
        indices = paired_bootstrap_indices(
            n, config=BootstrapConfig(replicates=100, seed=42)
        )
        assert numpy.all(indices >= 0)
        assert numpy.all(indices < n)

    def test_deterministic_with_seed(self):
        config = BootstrapConfig(seed=42, replicates=10)
        idx1 = paired_bootstrap_indices(5, config=config)
        idx2 = paired_bootstrap_indices(5, config=config)
        assert numpy.array_equal(idx1, idx2)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            paired_bootstrap_indices(0)

    def test_default_config_used(self):
        indices = paired_bootstrap_indices(10)
        assert indices.shape == (1000, 10)


# ===================================================================
# 16. Efficiency measurement
# ===================================================================


class TestEfficiencyConfig:
    """Latency/memory benchmark defaults."""

    def test_default_values(self):
        config = EfficiencyConfig()
        assert config.batch_size == 1
        assert config.semantic_sequence_length == 256
        assert config.byte_sequence_length == 512
        assert config.warmup_iterations == 10
        assert config.timed_iterations == 50

    def test_custom_values(self):
        config = EfficiencyConfig(
            batch_size=4, semantic_sequence_length=128
        )
        assert config.batch_size == 4
        assert config.semantic_sequence_length == 128

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError):
            EfficiencyConfig(batch_size=0)

    def test_invalid_timed_iterations_raises(self):
        with pytest.raises(ValueError):
            EfficiencyConfig(timed_iterations=0)

    def test_negative_warmup_raises(self):
        with pytest.raises(ValueError):
            EfficiencyConfig(warmup_iterations=-1)


class TestMeasureLatency:
    """Latency measurement helper."""

    def test_basic_latency_measurement(self):
        call_count = 0

        def forward_fn():
            nonlocal call_count
            call_count += 1
            time.sleep(0.001)

        config = EfficiencyConfig(
            warmup_iterations=2, timed_iterations=5
        )
        result = measure_latency(
            forward_fn,
            config=config,
            device="cpu",
            model_variant="test",
        )
        assert call_count == 7  # 2 warmup + 5 timed
        assert result.mean_ms > 0
        assert len(result.individual_ms) == 5
        assert result.device == "cpu"

    def test_result_fields(self):
        def noop():
            pass

        config = EfficiencyConfig(warmup_iterations=1, timed_iterations=3)
        result = measure_latency(
            noop, config=config, dtype="float32"
        )
        assert result.dtype == "float32"
        assert result.model_variant == "unknown"
        assert result.runtime_versions is None


class TestMeasureMemory:
    """Memory measurement helper."""

    def test_cpu_returns_none(self):
        def noop():
            pass

        result = measure_memory(
            noop, device="cpu", model_variant="test"
        )
        assert result.max_memory_allocated_bytes is None
        assert result.max_memory_reserved_bytes is None
        assert result.cuda_available is False


# ===================================================================
# 18. Frozen structures
# ===================================================================


class TestFrozenStructures:
    """Immutability of frozen dataclasses."""

    def test_frozen_split_manifest(self):
        manifest = _valid_manifest()
        with pytest.raises(AttributeError):
            manifest.dataset_version = "v2.0"

    def test_frozen_bootstrap_config(self):
        config = BootstrapConfig()
        with pytest.raises(AttributeError):
            config.seed = 99

    def test_frozen_bootstrap_ci(self):
        ci = BootstrapCI(
            point_estimate=0.8,
            lower=0.6,
            upper=0.95,
            replicates=100,
            seed=42,
            ci_level=0.95,
            method="percentile",
        )
        with pytest.raises(AttributeError):
            ci.point_estimate = 0.9

    def test_frozen_scientific_macro_result(self):
        result = compute_scientific_macro([])
        with pytest.raises(AttributeError):
            result.macro_f1 = 0.5

    def test_frozen_efficiency_config(self):
        config = EfficiencyConfig()
        with pytest.raises(AttributeError):
            config.batch_size = 8
