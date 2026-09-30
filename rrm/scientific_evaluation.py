"""Scientific evaluation infrastructure for the RRM (RRM 3.10).

Implements the frozen protocol for scientific model comparison:

- FrozenSplitManifest: immutable split manifest with validation
- LeakageReport / LeakageFinding: cross-split leakage detection
- Task evaluability: ground-truth support based evaluability
- ScientificTaskResult / ScientificMacroResult: per-task and macro metrics
- Same-task-set comparison validation
- Threshold selection: fixed 0.5 and validation-tuned
- Slice aggregation: generic slice evaluation infrastructure
- BootstrapConfig / bootstrap_ci: NumPy percentile bootstrap CIs
- paired_bootstrap_indices: identical sampling for paired comparison
- EfficiencyConfig / latency / memory measurement helpers

Research provenance:
    - Task evaluability contract: RRM 3.10 SCIENTIFIC PROTOCOL (LOCKED)
    - One-class → all discriminative metrics = None: RRM 3.10 protocol
    - Validation threshold selection: RRM 3.10 protocol
    - Bootstrap percentile CI: STANDARD ENGINEERING PRACTICE (Efron)
    - Paired bootstrap indices: STANDARD ENGINEERING PRACTICE
    - All defaults: RRM 3.10 protocol

Architecture rule:
    RRM UNDERSTANDS. TRUST DECIDES.

This module produces evaluation evidence and infrastructure only.
It does NOT make moderation decisions.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from rrm.evaluation import (
    MacroMetrics,
    TaskMetrics,
    aggregate_macro_metrics,
    compute_task_metrics,
)
from rrm.labels import PRIMARY_LABELS, NUM_PRIMARY_LABELS, UNKNOWN_LABEL
from rrm.similarity import SimilarityMatch, find_best_match


# ---------------------------------------------------------------------------
# Scientific status constants
# ---------------------------------------------------------------------------

#: Status for results produced from synthetic pilot data.
#: Such results may only support infrastructure/smoke tests.
NON_SCIENTIFIC_SYNTHETIC_SMOKE: str = "NON_SCIENTIFIC_SYNTHETIC_SMOKE"

#: Status for results that have passed all production evaluation gates.
SCIENTIFIC_PRODUCTION: str = "SCIENTIFIC_PRODUCTION"

#: Status for results from a real dataset that has not yet completed
#: all leakage/split gates.
SCIENTIFIC_PARTIAL: str = "SCIENTIFIC_PARTIAL"

#: All valid scientific status values.
VALID_SCIENTIFIC_STATUSES: Tuple[str, ...] = (
    NON_SCIENTIFIC_SYNTHETIC_SMOKE,
    SCIENTIFIC_PRODUCTION,
    SCIENTIFIC_PARTIAL,
)


# ---------------------------------------------------------------------------
# Frozen split manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenSplitManifest:
    """Immutable manifest describing a frozen train/validation/test split.

    A frozen manifest is consumed by scientific evaluation.  Evaluation
    must NOT reshuffle records immediately before evaluation.

    Attributes
    ----------
    dataset_version : str
        Human-readable version identifier for the dataset.
    dataset_hash : str
        SHA-256 hex digest of the frozen dataset artifact.
    split_version : str
        Human-readable version identifier for this split.
    split_hash : str
        SHA-256 hex digest of the split assignment artifact.
    split_generation_identity : str
        Identity of the process/person that generated the split.
    split_seed : int or None
        Random seed used for split generation, if any.
    train_review_ids : tuple[str, ...]
        Review IDs assigned to the training split.
    validation_review_ids : tuple[str, ...]
        Review IDs assigned to the validation split.
    test_review_ids : tuple[str, ...]
        Review IDs assigned to the test split.
    lineage_group_metadata : dict or None
        Mapping from lineage_group_id to review_ids, where available.
    duplicate_cluster_metadata : dict or None
        Duplicate/near-duplicate cluster information, where available.
    created_at : str
        ISO-8601 timestamp of manifest creation.
    protocol_version : str
        Protocol version this manifest conforms to.
    """

    dataset_version: str
    dataset_hash: str
    split_version: str
    split_hash: str
    split_generation_identity: str
    split_seed: Optional[int]
    train_review_ids: Tuple[str, ...]
    validation_review_ids: Tuple[str, ...]
    test_review_ids: Tuple[str, ...]
    lineage_group_metadata: Optional[Dict[str, Any]] = None
    duplicate_cluster_metadata: Optional[Dict[str, Any]] = None
    created_at: str = ""
    protocol_version: str = "1.0"


def validate_split_manifest(manifest: FrozenSplitManifest) -> List[str]:
    """Validate a frozen split manifest.

    Returns a list of error messages.  An empty list means the manifest
    is valid.

    Parameters
    ----------
    manifest : FrozenSplitManifest
        Manifest to validate.

    Returns
    -------
    list of str
        Error messages, empty if valid.
    """
    errors: List[str] = []

    # Required string fields
    if not manifest.dataset_version:
        errors.append("dataset_version must be non-empty")
    if not manifest.dataset_hash:
        errors.append("dataset_hash must be non-empty")
    if not manifest.split_version:
        errors.append("split_version must be non-empty")
    if not manifest.split_hash:
        errors.append("split_hash must be non-empty")
    if not manifest.split_generation_identity:
        errors.append("split_generation_identity must be non-empty")
    if not manifest.protocol_version:
        errors.append("protocol_version must be non-empty")

    # split_seed must be None or int
    if manifest.split_seed is not None and not isinstance(
        manifest.split_seed, int
    ):
        errors.append(
            f"split_seed must be None or int, got {type(manifest.split_seed).__name__}"
        )

    # Review IDs must be strings
    for split_name, ids in [
        ("train", manifest.train_review_ids),
        ("validation", manifest.validation_review_ids),
        ("test", manifest.test_review_ids),
    ]:
        for rid in ids:
            if not isinstance(rid, str):
                errors.append(
                    f"review_id {rid!r} in {split_name} is not a str"
                )

    # No duplicate IDs within a single split
    for split_name, ids in [
        ("train", manifest.train_review_ids),
        ("validation", manifest.validation_review_ids),
        ("test", manifest.test_review_ids),
    ]:
        seen: set = set()
        for rid in ids:
            if rid in seen:
                errors.append(
                    f"Duplicate review_id '{rid}' within {split_name} split"
                )
            seen.add(rid)

    # No review ID across splits
    all_ids: Dict[str, str] = {}
    for split_name, ids in [
        ("train", manifest.train_review_ids),
        ("validation", manifest.validation_review_ids),
        ("test", manifest.test_review_ids),
    ]:
        for rid in ids:
            if rid in all_ids:
                errors.append(
                    f"review_id '{rid}' appears in both "
                    f"{split_name} and {all_ids[rid]}"
                )
            all_ids[rid] = split_name

    # Metadata must be serializable if provided
    if manifest.lineage_group_metadata is not None:
        try:
            json.dumps(manifest.lineage_group_metadata)
        except (TypeError, ValueError) as exc:
            errors.append(
                f"lineage_group_metadata is not JSON-serializable: {exc}"
            )

    if manifest.duplicate_cluster_metadata is not None:
        try:
            json.dumps(manifest.duplicate_cluster_metadata)
        except (TypeError, ValueError) as exc:
            errors.append(
                f"duplicate_cluster_metadata is not JSON-serializable: {exc}"
            )

    return errors


# ---------------------------------------------------------------------------
# Leakage structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeakageFinding:
    """One detected cross-split leakage instance.

    Attributes
    ----------
    leak_type : str
        Type of leakage: "review_id", "exact_text", "fingerprint",
        "near_duplicate", or "lineage_group".
    record_id_a : str
        First record identifier.
    split_a : str
        Split containing record_a.
    record_id_b : str
        Second record identifier.
    split_b : str
        Split containing record_b.
    detail : str
        Human-readable detail about the leakage.
    score : float or None
        Similarity score for near-duplicate findings, None otherwise.
    """

    leak_type: str
    record_id_a: str
    split_a: str
    record_id_b: str
    split_b: str
    detail: str
    score: Optional[float] = None


@dataclass(frozen=True)
class LeakageReport:
    """Complete leakage audit report for a frozen split.

    Attributes
    ----------
    findings : tuple[LeakageFinding, ...]
        All detected leakage instances.
    has_unresolved_leakage : bool
        True if any finding requires resolution before evaluation.
    total_review_ids_checked : int
        Total number of unique review IDs audited.
    total_records_checked : int
        Total number of records audited.
    """

    findings: Tuple[LeakageFinding, ...]
    has_unresolved_leakage: bool
    total_review_ids_checked: int
    total_records_checked: int


@dataclass(frozen=True)
class LeakageInputRecord:
    """Minimal record structure for leakage analysis.

    Attributes
    ----------
    review_id : str
        Unique identifier for the review.
    review_text : str
        Raw review text.
    split : str
        Split identity: "train", "validation", or "test".
    lineage_group_id : str or None
        Optional lineage/group identifier.
    """

    review_id: str
    review_text: str
    split: str
    lineage_group_id: Optional[str] = None


def check_review_id_leakage(
    records: Sequence[LeakageInputRecord],
) -> List[LeakageFinding]:
    """Detect review_ids appearing in more than one split.

    Parameters
    ----------
    records : sequence of LeakageInputRecord
        Records to audit.

    Returns
    -------
    list of LeakageFinding
        Deterministically sorted findings.
    """
    id_to_splits: Dict[str, List[str]] = {}
    for rec in records:
        if rec.review_id not in id_to_splits:
            id_to_splits[rec.review_id] = []
        if rec.split not in id_to_splits[rec.review_id]:
            id_to_splits[rec.review_id].append(rec.split)

    findings: List[LeakageFinding] = []
    for rid, splits in sorted(id_to_splits.items()):
        if len(splits) > 1:
            for i in range(len(splits)):
                for j in range(i + 1, len(splits)):
                    findings.append(
                        LeakageFinding(
                            leak_type="review_id",
                            record_id_a=rid,
                            split_a=splits[i],
                            record_id_b=rid,
                            split_b=splits[j],
                            detail=f"review_id '{rid}' in both "
                                   f"{splits[i]} and {splits[j]}",
                        )
                    )

    return sorted(findings, key=lambda f: (f.record_id_a, f.split_a, f.split_b))


def check_exact_text_leakage(
    records: Sequence[LeakageInputRecord],
) -> List[LeakageFinding]:
    """Detect identical raw review text crossing splits.

    Parameters
    ----------
    records : sequence of LeakageInputRecord
        Records to audit.

    Returns
    -------
    list of LeakageFinding
        Deterministically sorted findings.
    """
    text_to_ids: Dict[str, List[Tuple[str, str]]] = {}
    for rec in records:
        text_to_ids.setdefault(rec.review_text, []).append(
            (rec.review_id, rec.split)
        )

    findings: List[LeakageFinding] = []
    seen_pairs: set = set()

    for _text, id_splits in sorted(text_to_ids.items()):
        if len(id_splits) > 1:
            for i in range(len(id_splits)):
                for j in range(i + 1, len(id_splits)):
                    rid_a, split_a = id_splits[i]
                    rid_b, split_b = id_splits[j]
                    if split_a != split_b:
                        pair = (rid_a, split_a, rid_b, split_b)
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            findings.append(
                                LeakageFinding(
                                    leak_type="exact_text",
                                    record_id_a=rid_a,
                                    split_a=split_a,
                                    record_id_b=rid_b,
                                    split_b=split_b,
                                    detail=f"identical raw text in "
                                           f"{split_a} ({rid_a}) and "
                                           f"{split_b} ({rid_b})",
                                )
                            )

    return sorted(findings, key=lambda f: (f.record_id_a, f.split_a))


def check_fingerprint_leakage(
    records: Sequence[LeakageInputRecord],
) -> List[LeakageFinding]:
    """Detect same normalized fingerprint crossing splits.

    Uses the public ``exact_fingerprint`` function from
    ``rrm.text_normalization``.

    Parameters
    ----------
    records : sequence of LeakageInputRecord
        Records to audit.

    Returns
    -------
    list of LeakageFinding
        Deterministically sorted findings.
    """
    from rrm.text_normalization import exact_fingerprint

    fp_to_ids: Dict[str, List[Tuple[str, str]]] = {}
    for rec in records:
        fp = exact_fingerprint(rec.review_text)
        fp_to_ids.setdefault(fp, []).append((rec.review_id, rec.split))

    findings: List[LeakageFinding] = []
    seen_pairs: set = set()

    for _fp, id_splits in sorted(fp_to_ids.items()):
        if len(id_splits) > 1:
            for i in range(len(id_splits)):
                for j in range(i + 1, len(id_splits)):
                    rid_a, split_a = id_splits[i]
                    rid_b, split_b = id_splits[j]
                    if split_a != split_b:
                        pair = (rid_a, split_a, rid_b, split_b)
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            findings.append(
                                LeakageFinding(
                                    leak_type="fingerprint",
                                    record_id_a=rid_a,
                                    split_a=split_a,
                                    record_id_b=rid_b,
                                    split_b=split_b,
                                    detail=f"same normalized fingerprint in "
                                           f"{split_a} ({rid_a}) and "
                                           f"{split_b} ({rid_b})",
                                )
                            )

    return sorted(findings, key=lambda f: (f.record_id_a, f.split_a))


def check_near_duplicate_leakage(
    records: Sequence[LeakageInputRecord],
    *,
    n: int = 4,
    threshold: float = 0.85,
    exclude_self: bool = True,
) -> List[LeakageFinding]:
    """Report near-duplicate candidates crossing splits.

    Uses the public ``find_best_match`` function from ``rrm.similarity``.
    Results are CANDIDATES only.  No record is automatically invalidated.

    Parameters
    ----------
    records : sequence of LeakageInputRecord
        Records to audit.
    n : int
        N-gram size for similarity computation.  Default 4.
    threshold : float
        Minimum similarity score to report as a candidate.
        Default 0.85.  This is an engineering smoke value, not a
        scientifically final threshold.
    exclude_self : bool
        If True, skip candidates from the same review_id.

    Returns
    -------
    list of LeakageFinding
        Deterministically sorted findings.
    """
    # Group by split for efficient cross-split comparison
    by_split: Dict[str, List[Tuple[str, str]]] = {}
    for rec in records:
        by_split.setdefault(rec.split, []).append(
            (rec.review_id, rec.review_text)
        )

    split_names = sorted(by_split.keys())
    findings: List[LeakageFinding] = []

    for idx_a in range(len(split_names)):
        for idx_b in range(idx_a + 1, len(split_names)):
            split_a = split_names[idx_a]
            split_b = split_names[idx_b]
            candidates = by_split[split_b]

            for query_id, query_text in by_split[split_a]:
                if exclude_self:
                    cands = [
                        (cid, ctext)
                        for cid, ctext in candidates
                        if cid != query_id
                    ]
                else:
                    cands = candidates

                if not cands:
                    continue

                match = find_best_match(
                    query_text, cands, n=n, exclude_review_id=None
                )
                if match is not None and match.score >= threshold:
                    findings.append(
                        LeakageFinding(
                            leak_type="near_duplicate",
                            record_id_a=query_id,
                            split_a=split_a,
                            record_id_b=match.review_id,
                            split_b=split_b,
                            detail=f"near-duplicate candidate "
                                   f"(score={match.score:.4f}, "
                                   f"n={n}, threshold={threshold})",
                            score=float(match.score),
                        )
                    )

    return sorted(findings, key=lambda f: (
        -f.score if f.score is not None else 0.0,
        f.record_id_a,
        f.split_a,
    ))


_CANONICAL_SPLIT_ORDER: Tuple[str, ...] = (
    "train",
    "validation",
    "test",
)
_SPLIT_RANK: Dict[str, int] = {s: i for i, s in enumerate(_CANONICAL_SPLIT_ORDER)}


def _split_sort_key(split: str) -> int:
    """Return canonical rank for a split name; unknown splits sort last."""
    return _SPLIT_RANK.get(split, 999)


def check_lineage_leakage(
    records: Sequence[LeakageInputRecord],
) -> List[LeakageFinding]:
    """Detect lineage_group_ids crossing splits.

    If lineage metadata is absent for a record, that record is skipped
    (no fabricated violation).

    Parameters
    ----------
    records : sequence of LeakageInputRecord
        Records to audit.

    Returns
    -------
    list of LeakageFinding
        Deterministically sorted findings.
    """
    group_to_splits: Dict[str, List[Tuple[str, str]]] = {}
    for rec in records:
        if rec.lineage_group_id is None:
            continue
        if rec.lineage_group_id not in group_to_splits:
            group_to_splits[rec.lineage_group_id] = []
        group_to_splits[rec.lineage_group_id].append(
            (rec.review_id, rec.split)
        )

    findings: List[LeakageFinding] = []
    for group_id in sorted(group_to_splits.keys()):
        id_splits = group_to_splits[group_id]
        splits_seen_set: set = set()
        for _, split in id_splits:
            splits_seen_set.add(split)
        splits_seen: List[str] = sorted(
            splits_seen_set, key=_split_sort_key
        )

        if len(splits_seen) > 1:
            for i in range(len(splits_seen)):
                for j in range(i + 1, len(splits_seen)):
                    ids_in_a = sorted([
                        rid for rid, s in id_splits if s == splits_seen[i]
                    ])
                    ids_in_b = sorted([
                        rid for rid, s in id_splits if s == splits_seen[j]
                    ])
                    findings.append(
                        LeakageFinding(
                            leak_type="lineage_group",
                            record_id_a=ids_in_a[0],
                            split_a=splits_seen[i],
                            record_id_b=ids_in_b[0],
                            split_b=splits_seen[j],
                            detail=f"lineage_group '{group_id}' spans "
                                   f"{splits_seen[i]} and {splits_seen[j]}",
                        )
                    )

    return sorted(findings, key=lambda f: (f.record_id_a, f.split_a))


def run_full_leakage_check(
    records: Sequence[LeakageInputRecord],
    *,
    n: int = 4,
    near_duplicate_threshold: float = 0.85,
) -> LeakageReport:
    """Run all leakage checks and produce a complete report.

    Parameters
    ----------
    records : sequence of LeakageInputRecord
        Records to audit.
    n : int
        N-gram size for near-duplicate similarity.
    near_duplicate_threshold : float
        Minimum similarity for near-duplicate candidates.

    Returns
    -------
    LeakageReport
        Complete leakage audit report.
    """
    all_findings: List[LeakageFinding] = []

    all_findings.extend(check_review_id_leakage(records))
    all_findings.extend(check_exact_text_leakage(records))
    all_findings.extend(check_fingerprint_leakage(records))
    all_findings.extend(
        check_near_duplicate_leakage(
            records,
            n=n,
            threshold=near_duplicate_threshold,
        )
    )
    all_findings.extend(check_lineage_leakage(records))

    return LeakageReport(
        findings=tuple(all_findings),
        has_unresolved_leakage=len(all_findings) > 0,
        total_review_ids_checked=len(
            {rec.review_id for rec in records}
        ),
        total_records_checked=len(records),
    )


# ---------------------------------------------------------------------------
# Task evaluability
# ---------------------------------------------------------------------------


def is_task_evaluable(
    y_true: Sequence[int],
) -> bool:
    """Determine if a task is scientifically evaluable from ground truth.

    A task is evaluable if and only if BOTH ground-truth classes are
    present among known (non-UNKNOWN) labels.

    Parameters
    ----------
    y_true : sequence of int
        True label values.  Valid: -1 (UNKNOWN), 0 (negative), 1 (positive).

    Returns
    -------
    bool
        True if both classes 0 and 1 are present, False otherwise.
    """
    known = [y for y in y_true if y != UNKNOWN_LABEL]
    if not known:
        return False
    unique = set(known)
    return 0 in unique and 1 in unique


def task_support_counts(
    y_true: Sequence[int],
) -> Tuple[int, int, int]:
    """Compute known, positive, and negative support counts.

    Parameters
    ----------
    y_true : sequence of int
        True label values.

    Returns
    -------
    (known_support, positive_support, negative_support)
    """
    known = [y for y in y_true if y != UNKNOWN_LABEL]
    known_support = len(known)
    positive_support = sum(1 for y in known if y == 1)
    negative_support = sum(1 for y in known if y == 0)
    return known_support, positive_support, negative_support


# ---------------------------------------------------------------------------
# Scientific macro results with task names
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScientificMacroResult:
    """Macro metrics with contributing task names.

    Attributes
    ----------
    macro_f1 : float or None
        Mean F1 across tasks with valid F1.
    macro_f1_task_count : int
        Number of tasks contributing to macro_f1.
    macro_f1_task_names : tuple[str, ...]
        Canonical PRIMARY_LABELS names of contributing tasks.
    macro_auprc : float or None
        Mean AUPRC across tasks with valid AUPRC.
    macro_auprc_task_count : int
        Number of tasks contributing to macro_auprc.
    macro_auprc_task_names : tuple[str, ...]
        Canonical PRIMARY_LABELS names of contributing tasks.
    """

    macro_f1: Optional[float]
    macro_f1_task_count: int
    macro_f1_task_names: Tuple[str, ...]
    macro_auprc: Optional[float]
    macro_auprc_task_count: int
    macro_auprc_task_names: Tuple[str, ...]


def compute_scientific_macro(
    task_metrics: Sequence[TaskMetrics],
) -> ScientificMacroResult:
    """Compute macro metrics with task-name tracking.

    Only tasks with non-None metric values contribute.
    Unavailable tasks are never replaced with zero.

    Parameters
    ----------
    task_metrics : sequence of TaskMetrics
        Per-task metrics in canonical PRIMARY_LABELS order.

    Returns
    -------
    ScientificMacroResult
    """
    valid_f1 = [
        tm for tm in task_metrics if tm.f1 is not None
    ]
    valid_auprc = [
        tm for tm in task_metrics if tm.auprc is not None
    ]

    macro_f1 = float(numpy.mean([tm.f1 for tm in valid_f1])) if valid_f1 else None
    macro_f1_task_names = tuple(tm.label for tm in valid_f1)

    macro_auprc = (
        float(numpy.mean([tm.auprc for tm in valid_auprc]))
        if valid_auprc
        else None
    )
    macro_auprc_task_names = tuple(tm.label for tm in valid_auprc)

    return ScientificMacroResult(
        macro_f1=macro_f1,
        macro_f1_task_count=len(valid_f1),
        macro_f1_task_names=macro_f1_task_names,
        macro_auprc=macro_auprc,
        macro_auprc_task_count=len(valid_auprc),
        macro_auprc_task_names=macro_auprc_task_names,
    )


# ---------------------------------------------------------------------------
# Same task set comparison validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonValidationResult:
    """Result of comparing two models' scientific task sets.

    Attributes
    ----------
    valid : bool
        True if both models have the same evaluable task set.
    model_a_task_names : tuple[str, ...]
        Contributing task names for model A.
    model_b_task_names : tuple[str, ...]
        Contributing task names for model B.
    differences : tuple[str, ...]
        Description of differences, empty if valid.
    """

    valid: bool
    model_a_task_names: Tuple[str, ...]
    model_b_task_names: Tuple[str, ...]
    differences: Tuple[str, ...]


def validate_comparison_task_sets(
    macro_a: ScientificMacroResult,
    macro_b: ScientificMacroResult,
) -> ComparisonValidationResult:
    """Validate that two models share the same evaluable task set.

    For a fixed evaluation dataset, the evaluable task set is determined
    by ground truth.  It must NOT differ depending on model predictions.

    Parameters
    ----------
    macro_a : ScientificMacroResult
        Macro result for model A.
    macro_b : ScientificMacroResult
        Macro result for model B.

    Returns
    -------
    ComparisonValidationResult
    """
    differences: List[str] = []

    if macro_a.macro_auprc_task_names != macro_b.macro_auprc_task_names:
        differences.append(
            f"macro_auprc task sets differ: "
            f"model_a={macro_a.macro_auprc_task_names}, "
            f"model_b={macro_b.macro_auprc_task_names}"
        )

    if macro_a.macro_f1_task_names != macro_b.macro_f1_task_names:
        differences.append(
            f"macro_f1 task sets differ: "
            f"model_a={macro_a.macro_f1_task_names}, "
            f"model_b={macro_b.macro_f1_task_names}"
        )

    return ComparisonValidationResult(
        valid=len(differences) == 0,
        model_a_task_names=macro_a.macro_auprc_task_names,
        model_b_task_names=macro_b.macro_auprc_task_names,
        differences=tuple(differences),
    )


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdResult:
    """Result of validation threshold selection.

    Attributes
    ----------
    threshold : float
        Selected threshold value.
    threshold_source : str
        Origin of the threshold.
        "validation_tuned" or "default_0.5_no_evaluable_validation".
    validation_f1 : float or None
        F1 achieved at the selected threshold on validation data.
        None if default 0.5 was used.
    candidate_count : int
        Number of candidate thresholds evaluated.
    """

    threshold: float
    threshold_source: str
    validation_f1: Optional[float]
    candidate_count: int


DEFAULT_THRESHOLD: float = 0.5
DEFAULT_THRESHOLD_SOURCE: str = "default_0.5_no_evaluable_validation"


def select_validation_threshold(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    *,
    label_name: str = "",
) -> ThresholdResult:
    """Select per-task threshold using validation data only.

    Candidate thresholds: unique validation sigmoid scores plus 0.5.
    Optimize: validation F1 on known labels only.
    Tie-breaking:
        1. threshold closest to 0.5
        2. smaller numeric threshold

    If the validation task lacks both classes: return 0.5 with
    ``threshold_source = "default_0.5_no_evaluable_validation"``.

    Parameters
    ----------
    y_true : sequence of int
        True labels.  Values: -1, 0, 1.
    y_proba : sequence of float
        Predicted sigmoid probabilities.
    label_name : str
        Label name for logging context.

    Returns
    -------
    ThresholdResult
    """
    if len(y_true) != len(y_proba):
        raise ValueError(
            f"y_true and y_proba must have the same length, "
            f"got {len(y_true)} and {len(y_proba)}"
        )

    # Mask known labels
    known_mask = [y != UNKNOWN_LABEL for y in y_true]
    y_true_known = [y for y, m in zip(y_true, known_mask) if m]
    y_proba_known = [p for p, m in zip(y_proba, known_mask) if m]

    # Check evaluability
    if not is_task_evaluable(y_true_known):
        return ThresholdResult(
            threshold=DEFAULT_THRESHOLD,
            threshold_source=DEFAULT_THRESHOLD_SOURCE,
            validation_f1=None,
            candidate_count=0,
        )

    # Build candidate thresholds: unique sigmoid scores + 0.5
    unique_scores = sorted(set(y_proba_known))
    candidates = list(unique_scores)
    if DEFAULT_THRESHOLD not in candidates:
        candidates.append(DEFAULT_THRESHOLD)
    candidates.sort()

    # Evaluate each candidate
    best_f1 = -1.0
    best_threshold = DEFAULT_THRESHOLD

    for thresh in candidates:
        y_pred = [1 if p >= thresh else 0 for p in y_proba_known]
        f1 = float(f1_score(y_true_known, y_pred, zero_division=0.0))

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
        elif f1 == best_f1:
            # Tie-breaking: closest to 0.5, then smaller
            if abs(thresh - DEFAULT_THRESHOLD) < abs(
                best_threshold - DEFAULT_THRESHOLD
            ):
                best_threshold = thresh
            elif abs(thresh - DEFAULT_THRESHOLD) == abs(
                best_threshold - DEFAULT_THRESHOLD
            ) and thresh < best_threshold:
                best_threshold = thresh

    return ThresholdResult(
        threshold=best_threshold,
        threshold_source="validation_tuned",
        validation_f1=best_f1 if best_f1 >= 0 else None,
        candidate_count=len(candidates),
    )


# ---------------------------------------------------------------------------
# Slice aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SliceResult:
    """Evaluation result for one slice.

    Attributes
    ----------
    slice_label : str
        Identifier for this slice.
    record_count : int
        Number of records in this slice.
    task_metrics : tuple[TaskMetrics, ...]
        Per-task metrics in canonical PRIMARY_LABELS order.
    macro : ScientificMacroResult or None
        Macro metrics, or None if no task is evaluable.
    """

    slice_label: str
    record_count: int
    task_metrics: Tuple[TaskMetrics, ...]
    macro: Optional[ScientificMacroResult]


@dataclass(frozen=True)
class SliceAggregation:
    """Complete slice aggregation for a model.

    Attributes
    ----------
    slices : tuple[SliceResult, ...]
        Per-slice results in deterministic order.
    """

    slices: Tuple[SliceResult, ...]


def aggregate_slices(
    y_true: numpy.ndarray,
    y_pred: numpy.ndarray,
    y_proba: numpy.ndarray,
    slice_labels: Sequence[str],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> SliceAggregation:
    """Aggregate evaluation metrics by slice.

    Caller supplies slice labels for each evaluation record.

    Parameters
    ----------
    y_true : numpy.ndarray
        True labels.  Shape (n, 6).
    y_pred : numpy.ndarray
        Predicted binary labels.  Shape (n, 6).
    y_proba : numpy.ndarray
        Predicted probabilities.  Shape (n, 6).
    slice_labels : sequence of str
        Slice label for each record.  Length must equal n.
    threshold : float
        Decision threshold for computing thresholded metrics.

    Returns
    -------
    SliceAggregation
    """
    if len(y_true) != len(slice_labels):
        raise ValueError(
            f"y_true ({len(y_true)}) and slice_labels ({len(slice_labels)}) "
            f"must have the same length"
        )

    # Group indices by slice label
    slice_indices: Dict[str, List[int]] = {}
    for i, label in enumerate(slice_labels):
        slice_indices.setdefault(label, []).append(i)

    slice_results: List[SliceResult] = []
    for slice_label in sorted(slice_indices.keys()):
        indices = slice_indices[slice_label]
        task_metrics: List[TaskMetrics] = []

        for task_idx, label_name in enumerate(PRIMARY_LABELS):
            st = y_true[indices, task_idx]
            sp = y_pred[indices, task_idx]
            sb = y_proba[indices, task_idx]

            tm = compute_task_metrics(
                st, sp, sb, label_name, threshold=threshold
            )
            task_metrics.append(tm)

        macro = compute_scientific_macro(task_metrics)
        slice_results.append(
            SliceResult(
                slice_label=slice_label,
                record_count=len(indices),
                task_metrics=tuple(task_metrics),
                macro=macro,
            )
        )

    return SliceAggregation(slices=tuple(slice_results))


# ---------------------------------------------------------------------------
# Bootstrap infrastructure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapConfig:
    """Immutable bootstrap configuration.

    Attributes
    ----------
    seed : int
        Random seed for reproducibility.
    replicates : int
        Number of bootstrap replicates.
    ci_level : float
        Confidence interval level (e.g., 0.95 for 95% CI).
    method : str
        CI method.  Currently only "percentile" is supported.
    """

    seed: int = 42
    replicates: int = 1000
    ci_level: float = 0.95
    method: str = "percentile"

    def __post_init__(self) -> None:
        if self.replicates <= 0:
            raise ValueError(
                f"replicates must be > 0, got {self.replicates}"
            )
        if not (0.0 < self.ci_level < 1.0):
            raise ValueError(
                f"ci_level must be in (0, 1), got {self.ci_level}"
            )
        if self.method != "percentile":
            raise ValueError(
                f"method must be 'percentile', got {self.method!r}"
            )


@dataclass(frozen=True)
class BootstrapCI:
    """Bootstrap confidence interval result.

    Attributes
    ----------
    point_estimate : float
        Point estimate from the full sample.
    lower : float
        Lower bound of the confidence interval.
    upper : float
        Upper bound of the confidence interval.
    replicates : int
        Number of bootstrap replicates used.
    seed : int
        Random seed used.
    ci_level : float
        Confidence interval level.
    method : str
        CI method used.
    """

    point_estimate: float
    lower: float
    upper: float
    replicates: int
    seed: int
    ci_level: float
    method: str


def bootstrap_ci(
    values: Sequence[float],
    *,
    config: Optional[BootstrapConfig] = None,
    statistic: Optional[str] = "mean",
) -> BootstrapCI:
    """Compute bootstrap confidence interval using NumPy percentile method.

    Parameters
    ----------
    values : sequence of float
        Point estimates from evaluation records.
    config : BootstrapConfig or None
        Bootstrap configuration.  Uses defaults if None.
    statistic : str
        Statistic to compute.  Currently only "mean" is supported.

    Returns
    -------
    BootstrapCI
    """
    if config is None:
        config = BootstrapConfig()

    if statistic != "mean":
        raise ValueError(
            f"statistic must be 'mean', got {statistic!r}"
        )

    arr = numpy.asarray(values, dtype=numpy.float64)
    n = len(arr)
    if n == 0:
        raise ValueError("values must not be empty")

    point_estimate = float(numpy.mean(arr))

    rng = numpy.random.Generator(numpy.random.Philox(seed=config.seed))
    bootstrap_means = numpy.empty(config.replicates, dtype=numpy.float64)

    for i in range(config.replicates):
        sample = rng.choice(arr, size=n, replace=True)
        bootstrap_means[i] = float(numpy.mean(sample))

    alpha = 1.0 - config.ci_level
    lower = float(numpy.percentile(bootstrap_means, 100.0 * alpha / 2.0))
    upper = float(numpy.percentile(bootstrap_means, 100.0 * (1.0 - alpha / 2.0)))

    return BootstrapCI(
        point_estimate=point_estimate,
        lower=lower,
        upper=upper,
        replicates=config.replicates,
        seed=config.seed,
        ci_level=config.ci_level,
        method=config.method,
    )


def paired_bootstrap_indices(
    n: int,
    *,
    config: Optional[BootstrapConfig] = None,
) -> numpy.ndarray:
    """Generate bootstrap sample indices for paired model comparison.

    For model A and model B evaluated on the same N records, use the
    SAME bootstrap sample indices for both.

    Parameters
    ----------
    n : int
        Number of evaluation records.
    config : BootstrapConfig or None
        Bootstrap configuration.  Uses defaults if None.

    Returns
    -------
    numpy.ndarray
        Shape (replicates, n).  Each row is one bootstrap sample's
        indices into the original evaluation arrays.
    """
    if config is None:
        config = BootstrapConfig()

    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")

    rng = numpy.random.Generator(numpy.random.Philox(seed=config.seed))
    indices = rng.integers(0, n, size=(config.replicates, n))

    return indices


# ---------------------------------------------------------------------------
# Efficiency measurement infrastructure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EfficiencyConfig:
    """Immutable benchmark configuration for efficiency measurement.

    Attributes
    ----------
    batch_size : int
        Batch size for measurement.
    semantic_sequence_length : int
        Token sequence length for the semantic encoder.
    byte_sequence_length : int
        Byte sequence length for the character branch.
    warmup_iterations : int
        Number of warmup iterations before timing.
    timed_iterations : int
        Number of iterations to time.
    """

    batch_size: int = 1
    semantic_sequence_length: int = 256
    byte_sequence_length: int = 512
    warmup_iterations: int = 10
    timed_iterations: int = 50

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be > 0, got {self.batch_size}"
            )
        if self.semantic_sequence_length <= 0:
            raise ValueError(
                f"semantic_sequence_length must be > 0, "
                f"got {self.semantic_sequence_length}"
            )
        if self.byte_sequence_length <= 0:
            raise ValueError(
                f"byte_sequence_length must be > 0, "
                f"got {self.byte_sequence_length}"
            )
        if self.warmup_iterations < 0:
            raise ValueError(
                f"warmup_iterations must be >= 0, "
                f"got {self.warmup_iterations}"
            )
        if self.timed_iterations <= 0:
            raise ValueError(
                f"timed_iterations must be > 0, "
                f"got {self.timed_iterations}"
            )


@dataclass(frozen=True)
class LatencyResult:
    """Result of a latency measurement.

    Attributes
    ----------
    mean_ms : float
        Mean latency in milliseconds.
    median_ms : float
        Median latency in milliseconds.
    p95_ms : float
        95th percentile latency in milliseconds.
    individual_ms : tuple[float, ...]
        Individual iteration timings in milliseconds.
    config : EfficiencyConfig
        Benchmark configuration used.
    device : str
        Device description.
    dtype : str
        Data type description.
    model_variant : str
        Model variant identifier.
    runtime_versions : dict
        Runtime version information.
    """

    mean_ms: float
    median_ms: float
    p95_ms: float
    individual_ms: Tuple[float, ...]
    config: EfficiencyConfig
    device: str
    dtype: str
    model_variant: str
    runtime_versions: Optional[Dict[str, str]] = None


def measure_latency(
    forward_fn: Any,
    *,
    config: Optional[EfficiencyConfig] = None,
    device: str = "cpu",
    dtype: str = "float32",
    model_variant: str = "unknown",
    runtime_versions: Optional[Dict[str, str]] = None,
) -> LatencyResult:
    """Measure inference latency with controlled warmup and timing.

    CUDA timing synchronizes before and after measured sections.

    Parameters
    ----------
    forward_fn : callable
        Zero-argument callable that performs one forward pass.
    config : EfficiencyConfig or None
        Benchmark configuration.  Uses defaults if None.
    device : str
        Device description.
    dtype : str
        Data type description.
    model_variant : str
        Model variant identifier.
    runtime_versions : dict or None
        Runtime version information.

    Returns
    -------
    LatencyResult
    """
    if config is None:
        config = EfficiencyConfig()

    # Warmup
    for _ in range(config.warmup_iterations):
        forward_fn()

    # Timed iterations
    timings: List[float] = []
    for _ in range(config.timed_iterations):
        if device.startswith("cuda"):
            import torch
            torch.cuda.synchronize()
        start = time.perf_counter()
        forward_fn()
        if device.startswith("cuda"):
            import torch
            torch.cuda.synchronize()
        end = time.perf_counter()
        timings.append((end - start) * 1000.0)  # Convert to ms

    timings_arr = numpy.array(timings, dtype=numpy.float64)
    return LatencyResult(
        mean_ms=float(numpy.mean(timings_arr)),
        median_ms=float(numpy.median(timings_arr)),
        p95_ms=float(numpy.percentile(timings_arr, 95.0)),
        individual_ms=tuple(float(t) for t in timings),
        config=config,
        device=device,
        dtype=dtype,
        model_variant=model_variant,
        runtime_versions=runtime_versions,
    )


@dataclass(frozen=True)
class MemoryResult:
    """Result of a CUDA memory measurement.

    Attributes
    ----------
    max_memory_allocated_bytes : int or None
        Peak allocated memory in bytes.  None if CUDA unavailable.
    max_memory_reserved_bytes : int or None
        Peak reserved memory in bytes.  None if CUDA unavailable.
    config : EfficiencyConfig
        Benchmark configuration used.
    device : str
        Device description.
    model_variant : str
        Model variant identifier.
    measurement_mode : str
        "inference" or "training_forward_backward".
    cuda_available : bool
        Whether CUDA was available for measurement.
    """

    max_memory_allocated_bytes: Optional[int]
    max_memory_reserved_bytes: Optional[int]
    config: EfficiencyConfig
    device: str
    model_variant: str
    measurement_mode: str
    cuda_available: bool


def measure_memory(
    forward_fn: Any,
    *,
    config: Optional[EfficiencyConfig] = None,
    device: str = "cpu",
    model_variant: str = "unknown",
    measurement_mode: str = "inference",
) -> MemoryResult:
    """Measure CUDA peak memory usage.

    Parameters
    ----------
    forward_fn : callable
        Zero-argument callable for the measurement.
    config : EfficiencyConfig or None
        Benchmark configuration.  Uses defaults if None.
    device : str
        Device description.
    model_variant : str
        Model variant identifier.
    measurement_mode : str
        "inference" or "training_forward_backward".

    Returns
    -------
    MemoryResult
    """
    if config is None:
        config = EfficiencyConfig()

    if not device.startswith("cuda"):
        return MemoryResult(
            max_memory_allocated_bytes=None,
            max_memory_reserved_bytes=None,
            config=config,
            device=device,
            model_variant=model_variant,
            measurement_mode=measurement_mode,
            cuda_available=False,
        )

    import torch

    torch.cuda.reset_peak_memory_stats()
    forward_fn()

    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()

    return MemoryResult(
        max_memory_allocated_bytes=int(allocated),
        max_memory_reserved_bytes=int(reserved),
        config=config,
        device=device,
        model_variant=model_variant,
        measurement_mode=measurement_mode,
        cuda_available=True,
    )
