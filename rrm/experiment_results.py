"""Experiment result versioning, JSON serialization, and Markdown rendering.

Implements the RRM 3.10 result versioning contract and machine-readable
result schema.  JSON is the source of truth.  Markdown is derived
presentation only.

Research provenance:
    - Result versioning contract: RRM 3.10 SCIENTIFIC PROTOCOL (LOCKED)
    - JSON source of truth: RRM 3.10 protocol section 8
    - Markdown is derived: RRM 3.10 protocol section 9
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy
import torch

from rrm.evaluation import TaskMetrics
from rrm.labels import PRIMARY_LABELS
from rrm.scientific_evaluation import (
    NON_SCIENTIFIC_SYNTHETIC_SMOKE,
    SCIENTIFIC_PRODUCTION,
    SCIENTIFIC_PARTIAL,
    VALID_SCIENTIFIC_STATUSES,
    ScientificMacroResult,
    compute_scientific_macro,
)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class SchemaValidationError(TypeError):
    """Raised when deserialized data violates the ScientificResult schema."""


def _validate_provenance_metadata(data: Dict[str, Any]) -> None:
    """Validate metadata fields have the correct types before constructing ProvenanceMetadata.

    Parameters
    ----------
    data : dict
        Metadata dictionary from deserialization input.

    Raises
    ------
    SchemaValidationError
        If a required field has the wrong type or an unexpected field is present.
    """
    expected_types: Dict[str, type] = {
        "protocol_version": str,
        "evaluation_round_id": str,
        "scientific_status": str,
        "dataset_version": str,
        "dataset_hash": str,
        "split_version": str,
        "split_hash": str,
        "tokenizer_identity": str,
        "tokenizer_hash": str,
        "semantic_pretraining_identity": str,
        "semantic_pretraining_hash": str,
        "supervised_checkpoint_identity": str,
        "supervised_checkpoint_hash": str,
        "code_commit": str,
        "model_variant": str,
        "seed": int,
        "training_config_identity": str,
        "metric_definition_version": str,
        "hardware_identity": str,
        "created_at": str,
    }
    optional_types: Dict[str, type] = {
        "thresholds": dict,
        "threshold_sources": dict,
        "bootstrap_seed": int,
        "bootstrap_replicates": int,
        "bootstrap_ci_level": float,
        "bootstrap_ci_method": str,
        "runtime_versions": dict,
    }

    for key, value in data.items():
        if key in expected_types:
            if not isinstance(value, expected_types[key]):
                raise SchemaValidationError(
                    f"Field '{key}' must be {expected_types[key].__name__}, got {type(value).__name__}: {value!r}"
                )
        elif key in optional_types:
            if value is not None and not isinstance(value, optional_types[key]):
                raise SchemaValidationError(
                    f"Field '{key}' must be {optional_types[key].__name__} or None, got {type(value).__name__}: {value!r}"
                )
        else:
            raise SchemaValidationError(
                f"Unexpected field '{key}' in metadata (allowed: {sorted(list(expected_types) + list(optional_types))})"
            )


# ---------------------------------------------------------------------------
# Provenance metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceMetadata:
    """Immutable provenance metadata for a scientific or smoke result.

    Attributes
    ----------
    protocol_version : str
        Protocol version this result was produced under.
    evaluation_round_id : str
        Unique identifier for this evaluation round.
    scientific_status : str
        One of VALID_SCIENTIFIC_STATUSES.
    dataset_version : str
        Dataset version identifier.
    dataset_hash : str
        SHA-256 of the dataset artifact.
    split_version : str
        Split manifest version.
    split_hash : str
        SHA-256 of the split assignment artifact.
    tokenizer_identity : str
        Tokenizer identity string.
    tokenizer_hash : str
        SHA-256 of the tokenizer file/state.
    semantic_pretraining_identity : str
        Semantic pretraining identity.
    semantic_pretraining_hash : str
        SHA-256 of the pretrained checkpoint.
    supervised_checkpoint_identity : str
        Supervised checkpoint identity.
    supervised_checkpoint_hash : str
        SHA-256 of the supervised checkpoint.
    code_commit : str
        Git commit hash.
    model_variant : str
        Model variant identifier (e.g., "full", "semantic_only").
    seed : int
        Random seed used.
    training_config_identity : str
        Training configuration identity.
    thresholds : dict or None
        Per-task threshold values.
    threshold_sources : dict or None
        Per-task threshold sources.
    metric_definition_version : str
        Version of the metric definitions used.
    bootstrap_seed : int or None
        Bootstrap seed, if bootstrapped.
    bootstrap_replicates : int or None
        Number of bootstrap replicates.
    bootstrap_ci_level : float or None
        Confidence interval level.
    bootstrap_ci_method : str or None
        Bootstrap CI method.
    runtime_versions : dict or None
        Runtime version information.
    hardware_identity : str
        Hardware description.
    created_at : str
        ISO-8601 timestamp.
    """

    protocol_version: str
    evaluation_round_id: str
    scientific_status: str
    dataset_version: str
    dataset_hash: str
    split_version: str
    split_hash: str
    tokenizer_identity: str
    tokenizer_hash: str
    semantic_pretraining_identity: str
    semantic_pretraining_hash: str
    supervised_checkpoint_identity: str
    supervised_checkpoint_hash: str
    code_commit: str
    model_variant: str
    seed: int
    training_config_identity: str
    thresholds: Optional[Dict[str, float]] = None
    threshold_sources: Optional[Dict[str, str]] = None
    metric_definition_version: str = "1.0"
    bootstrap_seed: Optional[int] = None
    bootstrap_replicates: Optional[int] = None
    bootstrap_ci_level: Optional[float] = None
    bootstrap_ci_method: Optional[str] = None
    runtime_versions: Optional[Dict[str, str]] = None
    hardware_identity: str = "unknown"
    created_at: str = ""


def _now_iso() -> str:
    """Return current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Default runtime versions helper
# ---------------------------------------------------------------------------


def default_runtime_versions() -> Dict[str, str]:
    """Collect runtime version information.

    Returns
    -------
    dict
        Keys: python, numpy, torch, sklearn.  Values: version strings.
    """
    versions: Dict[str, str] = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
    }
    try:
        versions["torch"] = torch.__version__
    except Exception:
        versions["torch"] = "unavailable"
    try:
        import sklearn
        versions["sklearn"] = sklearn.__version__
    except Exception:
        versions["sklearn"] = "unavailable"
    return versions


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScientificResult:
    """Complete scientific evaluation result.

    Attributes
    ----------
    metadata : ProvenanceMetadata
        Immutable provenance metadata.
    per_task : dict[str, TaskMetrics]
        Per-task metrics keyed by label name.
    macro : ScientificMacroResult
        Aggregate macro metrics with task-name tracking.
    thresholds : dict or None
        Per-task threshold values.
    confidence_intervals : dict or None
        Bootstrap confidence intervals.
    slice_results : dict or None
        Slice aggregation results.
    efficiency : dict or None
        Latency/memory results.
    limitations : tuple[str, ...]
        Explicit limitations for this result.
    scientific_status : str
        Scientific status string.
    """

    metadata: ProvenanceMetadata
    per_task: Dict[str, TaskMetrics]
    macro: ScientificMacroResult
    thresholds: Optional[Dict[str, Any]] = None
    confidence_intervals: Optional[Dict[str, Any]] = None
    slice_results: Optional[Dict[str, Any]] = None
    efficiency: Optional[Dict[str, Any]] = None
    limitations: Tuple[str, ...] = ()
    scientific_status: str = SCIENTIFIC_PRODUCTION


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def _task_metrics_to_dict(tm: TaskMetrics) -> Dict[str, Any]:
    """Convert TaskMetrics to a JSON-compatible dict."""
    return asdict(tm)


def _macro_metrics_to_dict(mm: ScientificMacroResult) -> Dict[str, Any]:
    """Convert ScientificMacroResult to a JSON-compatible dict.

    Tuples are converted to lists for JSON serialization.
    """
    raw: Dict[str, Any] = asdict(mm)
    return {k: list(v) if isinstance(v, tuple) else v for k, v in raw.items()}


def _provenance_to_dict(pm: ProvenanceMetadata) -> Dict[str, Any]:
    """Convert ProvenanceMetadata to a JSON-compatible dict."""
    return asdict(pm)


def serialize_scientific_result(
    result: ScientificResult,
) -> Dict[str, Any]:
    """Serialize a ScientificResult to a JSON-compatible dict.

    JSON is the source of truth.  Markdown is derived from this.

    Parameters
    ----------
    result : ScientificResult
        Result to serialize.

    Returns
    -------
    dict
        JSON-compatible dictionary.
    """
    per_task_dict: Dict[str, Any] = {}
    for label, tm in result.per_task.items():
        per_task_dict[label] = _task_metrics_to_dict(tm)

    thresholds_dict: Optional[Dict[str, Any]] = None
    if result.thresholds is not None:
        thresholds_dict = result.thresholds

    ci_dict: Optional[Dict[str, Any]] = None
    if result.confidence_intervals is not None:
        ci_dict = {
            k: (
                asdict(v) if hasattr(v, "__dataclass_fields__") else v
            )
            for k, v in result.confidence_intervals.items()
        }

    slice_dict: Optional[Dict[str, Any]] = None
    if result.slice_results is not None:
        slice_dict = {}
        for slice_label, slice_data in result.slice_results.items():
            if hasattr(slice_data, "__dataclass_fields__"):
                # SliceResult
                slice_dict[slice_label] = {
                    "record_count": slice_data.record_count,
                    "task_metrics": {
                        k: _task_metrics_to_dict(v)
                        for k, v in zip(
                            PRIMARY_LABELS, slice_data.task_metrics
                        )
                    },
                    "macro": (
                        _macro_metrics_to_dict(slice_data.macro)
                        if slice_data.macro is not None
                        else None
                    ),
                }
            else:
                slice_dict[slice_label] = slice_data

    efficiency_dict: Optional[Dict[str, Any]] = None
    if result.efficiency is not None:
        efficiency_dict = result.efficiency

    return {
        "metadata": _provenance_to_dict(result.metadata),
        "per_task": per_task_dict,
        "macro": _macro_metrics_to_dict(result.macro),
        "thresholds": thresholds_dict,
        "confidence_intervals": ci_dict,
        "slice_results": slice_dict,
        "efficiency": efficiency_dict,
        "limitations": list(result.limitations),
        "scientific_status": result.scientific_status,
    }


def deserialize_scientific_result(
    data: Dict[str, Any],
) -> ScientificResult:
    """Deserialize a ScientificResult from a JSON-compatible dict.

    Parameters
    ----------
    data : dict
        Serialized result dictionary.

    Returns
    -------
    ScientificResult
    """
    metadata_dict = data["metadata"]
    _validate_provenance_metadata(metadata_dict)
    metadata = ProvenanceMetadata(**metadata_dict)

    per_task: Dict[str, TaskMetrics] = {}
    for label, tm_dict in data["per_task"].items():
        per_task[label] = TaskMetrics(**tm_dict)

    macro_d = data["macro"]
    # Ensure tuple fields are reconstructed from lists
    if "macro_f1_task_names" in macro_d:
        macro_d = dict(macro_d)
        macro_d["macro_f1_task_names"] = tuple(macro_d["macro_f1_task_names"])
        macro_d["macro_auprc_task_names"] = tuple(macro_d["macro_auprc_task_names"])
    macro = ScientificMacroResult(**macro_d)

    return ScientificResult(
        metadata=metadata,
        per_task=per_task,
        macro=macro,
        thresholds=data.get("thresholds"),
        confidence_intervals=data.get("confidence_intervals"),
        slice_results=data.get("slice_results"),
        efficiency=data.get("efficiency"),
        limitations=tuple(data.get("limitations", [])),
        scientific_status=data.get(
            "scientific_status", SCIENTIFIC_PRODUCTION
        ),
    )


def result_to_json(
    result: ScientificResult,
    *,
    indent: int = 2,
) -> str:
    """Serialize a ScientificResult to a JSON string.

    Parameters
    ----------
    result : ScientificResult
        Result to serialize.
    indent : int
        JSON indentation level.

    Returns
    -------
    str
        JSON string.
    """
    data = serialize_scientific_result(result)
    return json.dumps(data, indent=indent, default=str)


def result_from_json(json_str: str) -> ScientificResult:
    """Deserialize a ScientificResult from a JSON string.

    Parameters
    ----------
    json_str : str
        JSON string.

    Returns
    -------
    ScientificResult
    """
    data = json.loads(json_str)
    return deserialize_scientific_result(data)


def save_result(result: ScientificResult, path: str) -> None:
    """Save a ScientificResult to a JSON file.

    Parameters
    ----------
    result : ScientificResult
        Result to save.
    path : str
        File path.
    """
    json_str = result_to_json(result)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json_str)


def load_result(path: str) -> ScientificResult:
    """Load a ScientificResult from a JSON file.

    Parameters
    ----------
    path : str
        File path.

    Returns
    -------
    ScientificResult
    """
    with open(path, "r", encoding="utf-8") as f:
        return result_from_json(f.read())


# ---------------------------------------------------------------------------
# Markdown rendering (derived from JSON source of truth)
# ---------------------------------------------------------------------------


def _format_metric(value: Any) -> str:
    """Format a metric value for display."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(
    result: ScientificResult,
    *,
    include_ci: bool = True,
    include_efficiency: bool = True,
) -> str:
    """Render a compact Markdown summary.

    Markdown is derived presentation only.  JSON is the source of truth.

    Parameters
    ----------
    result : ScientificResult
        Result to render.
    include_ci : bool
        Include confidence interval section.
    include_efficiency : bool
        Include efficiency section.

    Returns
    -------
    str
        Markdown string.
    """
    status_line = result.scientific_status
    if result.scientific_status == NON_SCIENTIFIC_SYNTHETIC_SMOKE:
        status_line += "\n\n**NON-SCIENTIFIC — SYNTHETIC PILOT — NOT PERFORMANCE EVIDENCE**"

    lines: List[str] = []
    lines.append("# RRM 3.10 — Scientific Evaluation Result")
    lines.append("")
    lines.append(f"**Run identity:** {result.metadata.evaluation_round_id}")
    lines.append(f"**Scientific status:** {status_line}")
    lines.append(f"**Protocol version:** {result.metadata.protocol_version}")
    lines.append("")

    # Primary metric
    lines.append("## Primary Metric")
    lines.append("")
    lines.append(f"**macro-AUPRC:** {_format_metric(result.macro.macro_auprc)}")
    auprc_names = result.macro.macro_auprc_task_names
    lines.append(
        f"Contributing tasks ({result.macro.macro_auprc_task_count}): "
        f"{', '.join(auprc_names) if auprc_names else 'none'}"
    )
    lines.append("")

    # Per-task table
    lines.append("## Per-Task Metrics")
    lines.append("")
    lines.append(
        "| Label | Support | Pos | Neg | Precision | Recall | F1 | AUPRC |"
    )
    lines.append(
        "|-------|---------|-----|-----|-----------|--------|----|-------|"
    )

    for label in PRIMARY_LABELS:
        tm = result.per_task.get(label)
        if tm is None:
            continue
        lines.append(
            f"| {tm.label} | {tm.known_support} | {tm.positive_support} "
            f"| {tm.negative_support} | {_format_metric(tm.precision)} "
            f"| {_format_metric(tm.recall)} | {_format_metric(tm.f1)} "
            f"| {_format_metric(tm.auprc)} |"
        )
    lines.append("")

    # Macro metrics
    lines.append("## Macro Metrics")
    lines.append("")
    f1_names = result.macro.macro_f1_task_names
    auprc_names_macro = result.macro.macro_auprc_task_names
    lines.append(
        f"**Macro F1:** {_format_metric(result.macro.macro_f1)} "
        f"({result.macro.macro_f1_task_count} tasks: "
        f"{', '.join(f1_names) if f1_names else 'none'})"
    )
    lines.append(
        f"**Macro AUPRC:** {_format_metric(result.macro.macro_auprc)} "
        f"({result.macro.macro_auprc_task_count} tasks: "
        f"{', '.join(auprc_names_macro) if auprc_names_macro else 'none'})"
    )
    lines.append("")

    # Threshold policy
    lines.append("## Threshold Policy")
    lines.append("")
    if result.thresholds:
        for label, thresh in sorted(result.thresholds.items()):
            source = result.metadata.threshold_sources or {}
            thresh_source = source.get(label, "unknown")
            lines.append(
                f"- **{label}:** threshold = {_format_metric(thresh)} "
                f"(source: {thresh_source})"
            )
    else:
        lines.append(
            "**Fixed threshold:** 0.5 (default, no validation tuning)"
        )
    lines.append("")

    # Confidence intervals
    if include_ci and result.confidence_intervals:
        lines.append("## Confidence Intervals")
        lines.append("")
        for key, ci_data in result.confidence_intervals.items():
            if hasattr(ci_data, "__dataclass_fields__"):
                ci = ci_data
                lines.append(
                    f"**{key}:** {_format_metric(ci.point_estimate)} "
                    f"[{_format_metric(ci.lower)}, {_format_metric(ci.upper)}] "
                    f"(n={ci.replicates}, level={ci.ci_level})"
                )
            elif isinstance(ci_data, dict):
                lines.append(
                    f"**{key}:** {_format_metric(ci_data.get('point_estimate'))} "
                    f"[{_format_metric(ci_data.get('lower'))}, "
                    f"{_format_metric(ci_data.get('upper'))}] "
                    f"(n={ci_data.get('replicates')}, level={ci_data.get('ci_level')})"
                )
        lines.append("")

    # Slice summaries
    if result.slice_results:
        lines.append("## Slice Summaries")
        lines.append("")
        for slice_label, slice_data in result.slice_results.items():
            if hasattr(slice_data, "__dataclass_fields__"):
                sr = slice_data  # type: SliceResult
                lines.append(f"### {sr.slice_label} (n={sr.record_count})")
                if sr.macro is not None:
                    lines.append(
                        f"Macro AUPRC: {_format_metric(sr.macro.macro_auprc)}, "
                        f"Macro F1: {_format_metric(sr.macro.macro_f1)}"
                    )
        lines.append("")

    # Efficiency
    if include_efficiency and result.efficiency:
        lines.append("## Efficiency")
        lines.append("")
        for key, eff_data in result.efficiency.items():
            if hasattr(eff_data, "__dataclass_fields__"):
                eff = eff_data
                lines.append(
                    f"**{key}:** mean={_format_metric(getattr(eff, 'mean_ms', None))}ms, "
                    f"median={_format_metric(getattr(eff, 'median_ms', None))}ms, "
                    f"p95={_format_metric(getattr(eff, 'p95_ms', None))}ms"
                )
        lines.append("")

    # Limitations
    if result.limitations:
        lines.append("## Limitations")
        lines.append("")
        for limitation in result.limitations:
            lines.append(f"- {limitation}")
        lines.append("")

    # Provenance
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- **Code commit:** {result.metadata.code_commit}")
    lines.append(f"- **Model variant:** {result.metadata.model_variant}")
    lines.append(f"- **Seed:** {result.metadata.seed}")
    lines.append(f"- **Created:** {result.metadata.created_at}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: build a synthetic-pilot result
# ---------------------------------------------------------------------------


def create_smoke_result(
    model_variant: str,
    seed: int,
    *,
    code_commit: str = "unknown",
    limitations: Sequence[str] = (),
) -> ScientificResult:
    """Create a non-scientific synthetic smoke result.

    Always uses ``NON_SCIENTIFIC_SYNTHETIC_SMOKE`` status.

    Parameters
    ----------
    model_variant : str
        Model variant identifier.
    seed : int
        Random seed used.
    code_commit : str
        Git commit hash.
    limitations : sequence of str
        Additional limitations.

    Returns
    -------
    ScientificResult
    """
    lim = list(limitations) + [
        "Results are from synthetic pilot data only.",
        "These results are NOT performance evidence for the real dataset.",
        "Do not use for model comparison.",
    ]

    metadata = ProvenanceMetadata(
        protocol_version="1.0",
        evaluation_round_id=f"smoke_{model_variant}_{seed}",
        scientific_status=NON_SCIENTIFIC_SYNTHETIC_SMOKE,
        dataset_version="rmc_pilot_v0.1",
        dataset_hash="synthetic_pilot_hash",
        split_version="pilot_split_v1",
        split_hash="synthetic_pilot_split_hash",
        tokenizer_identity="synthetic_pilot_tokenizer",
        tokenizer_hash="synthetic_tokenizer_hash",
        semantic_pretraining_identity="none",
        semantic_pretraining_hash="none",
        supervised_checkpoint_identity="none",
        supervised_checkpoint_hash="none",
        code_commit=code_commit,
        model_variant=model_variant,
        seed=seed,
        training_config_identity="none",
        created_at=_now_iso(),
        runtime_versions=default_runtime_versions(),
        hardware_identity="unknown",
    )

    # Empty per-task metrics (no real evaluation performed)
    per_task: Dict[str, TaskMetrics] = {}
    for label in PRIMARY_LABELS:
        per_task[label] = TaskMetrics(
            label=label,
            known_support=0,
            positive_support=0,
            negative_support=0,
            precision=None,
            recall=None,
            f1=None,
            auprc=None,
        )

    macro = compute_scientific_macro(list(per_task.values()))

    return ScientificResult(
        metadata=metadata,
        per_task=per_task,
        macro=macro,
        scientific_status=NON_SCIENTIFIC_SYNTHETIC_SMOKE,
        limitations=tuple(lim),
    )
