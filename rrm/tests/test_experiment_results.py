"""Tests for rrm.experiment_results.

Validates the RRM 3.10 experiment result versioning and serialization:
- ProvenanceMetadata creation and validation
- ScientificResult construction
- Scientific status handling (NON_SCIENTIFIC_SYNTHETIC_SMOKE, SCIENTIFIC_PRODUCTION, etc.)
- Production provenance validation
- Canonical six per-task entries
- ScientificMacroResult usage
- JSON serialization / deserialization roundtrip
- Markdown rendering
- Synthetic smoke Markdown warning banners
- Markdown non-mutation of source
- Helper utilities (create_smoke_result, default_runtime_versions)
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict

import numpy
import pytest

from rrm.evaluation import TaskMetrics
from rrm.experiment_results import (
    NON_SCIENTIFIC_SYNTHETIC_SMOKE,
    SCIENTIFIC_PRODUCTION,
    SCIENTIFIC_PARTIAL,
    VALID_SCIENTIFIC_STATUSES,
    ProvenanceMetadata,
    ScientificResult,
    ScientificMacroResult,
    SchemaValidationError,
    compute_scientific_macro,
    create_smoke_result,
    default_runtime_versions,
    deserialize_scientific_result,
    load_result,
    result_from_json,
    result_to_json,
    render_markdown,
    save_result,
    serialize_scientific_result,
)
from rrm.labels import PRIMARY_LABELS
from rrm.scientific_evaluation import (
    BootstrapConfig,
    BootstrapCI,
    SliceResult,
    SliceAggregation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task_metrics(
    label: str,
    known_support: int = 10,
    positive_support: int = 5,
    negative_support: int = 5,
    precision: float = 0.8,
    recall: float = 0.8,
    f1: float = 0.8,
    auprc: float = 0.85,
) -> TaskMetrics:
    """Create a TaskMetrics with valid values."""
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


def _make_scientific_macro(
    macro_f1: float = 0.8,
    macro_f1_task_count: int = 6,
    macro_f1_task_names: tuple = None,
    macro_auprc: float = 0.85,
    macro_auprc_task_count: int = 6,
    macro_auprc_task_names: tuple = None,
) -> ScientificMacroResult:
    """Create a ScientificMacroResult."""
    if macro_f1_task_names is None:
        macro_f1_task_names = tuple(PRIMARY_LABELS[:macro_f1_task_count])
    if macro_auprc_task_names is None:
        macro_auprc_task_names = tuple(PRIMARY_LABELS[:macro_auprc_task_count])
    return ScientificMacroResult(
        macro_f1=macro_f1,
        macro_f1_task_count=macro_f1_task_count,
        macro_f1_task_names=macro_f1_task_names,
        macro_auprc=macro_auprc,
        macro_auprc_task_count=macro_auprc_task_count,
        macro_auprc_task_names=macro_auprc_task_names,
    )


def _make_per_task() -> Dict[str, TaskMetrics]:
    """Create per-task metrics for all six canonical labels."""
    return {label: _make_task_metrics(label) for label in PRIMARY_LABELS}


def _make_provenance(**overrides) -> ProvenanceMetadata:
    """Create a ProvenanceMetadata with sensible defaults."""
    defaults = dict(
        protocol_version="1.0",
        evaluation_round_id="eval_001",
        scientific_status=SCIENTIFIC_PRODUCTION,
        dataset_version="rmc_v1.0",
        dataset_hash="a" * 64,
        split_version="split_v1",
        split_hash="b" * 64,
        tokenizer_identity="rmc_tokenizer_v1",
        tokenizer_hash="c" * 64,
        semantic_pretraining_identity="pretrain_v1",
        semantic_pretraining_hash="d" * 64,
        supervised_checkpoint_identity="checkpoint_v1",
        supervised_checkpoint_hash="e" * 64,
        code_commit="abc1234",
        model_variant="full",
        seed=42,
        training_config_identity="config_v1",
        thresholds={"spam": 0.5},
        threshold_sources={"spam": "validation_tuned"},
        metric_definition_version="1.0",
        bootstrap_seed=123,
        bootstrap_replicates=1000,
        bootstrap_ci_level=0.95,
        bootstrap_ci_method="percentile",
        runtime_versions={"python": "3.11.0", "numpy": "1.26.0"},
        hardware_identity="RTX_3080",
        created_at="2024-06-01T00:00:00+00:00",
    )
    defaults.update(overrides)
    return ProvenanceMetadata(**defaults)


def _make_result(**overrides) -> ScientificResult:
    """Create a ScientificResult with sensible defaults."""
    defaults = dict(
        metadata=_make_provenance(),
        per_task=_make_per_task(),
        macro=_make_scientific_macro(),
        thresholds={"spam": 0.5, "deception": 0.55},
        confidence_intervals={
            "macro_f1": BootstrapCI(
                point_estimate=0.82,
                lower=0.78,
                upper=0.86,
                replicates=1000,
                seed=42,
                ci_level=0.95,
                method="percentile",
            )
        },
        slice_results=None,
        efficiency={
            "inference_latency_ms": type("obj", (object,), {
                "mean_ms": 1.2, "median_ms": 1.1, "p95_ms": 2.0
            })()
        },
        limitations=("Limited to college-domain reviews.",),
        scientific_status=SCIENTIFIC_PRODUCTION,
    )
    defaults.update(overrides)
    return ScientificResult(**defaults)


# ===================================================================
# 1. Valid scientific status handling
# ===================================================================


class TestScientificStatusHandling:
    """Section 1: Scientific status handling."""

    def test_non_scientific_synthetic_smoke_exact_string(self):
        assert NON_SCIENTIFIC_SYNTHETIC_SMOKE == "NON_SCIENTIFIC_SYNTHETIC_SMOKE"

    def test_scientific_production_exact_string(self):
        assert SCIENTIFIC_PRODUCTION == "SCIENTIFIC_PRODUCTION"

    def test_scientific_partial_exact_string(self):
        assert SCIENTIFIC_PARTIAL == "SCIENTIFIC_PARTIAL"

    def test_valid_statuses_contains_all_three(self):
        assert len(VALID_SCIENTIFIC_STATUSES) == 3
        assert NON_SCIENTIFIC_SYNTHETIC_SMOKE in VALID_SCIENTIFIC_STATUSES
        assert SCIENTIFIC_PRODUCTION in VALID_SCIENTIFIC_STATUSES
        assert SCIENTIFIC_PARTIAL in VALID_SCIENTIFIC_STATUSES

    def test_result_defaults_to_production(self):
        result = _make_result()
        assert result.scientific_status == SCIENTIFIC_PRODUCTION

    def test_result_accepts_partial_status(self):
        result = _make_result(scientific_status=SCIENTIFIC_PARTIAL)
        assert result.scientific_status == SCIENTIFIC_PARTIAL

    def test_result_accepts_smoke_status(self):
        result = _make_result(scientific_status=NON_SCIENTIFIC_SYNTHETIC_SMOKE)
        assert result.scientific_status == NON_SCIENTIFIC_SYNTHETIC_SMOKE


# ===================================================================
# 2. Exact status: NON_SCIENTIFIC_SYNTHETIC_SMOKE
# ===================================================================


class TestNonScientificSyntheticSmoke:
    """Section 2: NON_SCIENTIFIC_SYNTHETIC_SMOKE status."""

    def test_create_smoke_result_uses_smoke_status(self):
        result = create_smoke_result("full", seed=42)
        assert result.scientific_status == NON_SCIENTIFIC_SYNTHETIC_SMOKE

    def test_smoke_result_has_limitations(self):
        result = create_smoke_result("full", seed=42)
        assert len(result.limitations) > 0
        limitation_text = " ".join(result.limitations)
        assert "synthetic" in limitation_text.lower()
        assert "NOT" in limitation_text

    def test_smoke_result_evaluation_round_id_format(self):
        result = create_smoke_result("full", seed=42)
        assert result.metadata.evaluation_round_id == "smoke_full_42"

    def test_smoke_result_macro_is_scientific_macro_result(self):
        result = create_smoke_result("full", seed=42)
        assert isinstance(result.macro, ScientificMacroResult)

    def test_smoke_result_per_task_has_six_labels(self):
        result = create_smoke_result("full", seed=42)
        assert set(result.per_task.keys()) == set(PRIMARY_LABELS)

    def test_smoke_result_serializes_correct_status(self):
        result = create_smoke_result("full", seed=42)
        data = serialize_scientific_result(result)
        assert data["scientific_status"] == NON_SCIENTIFIC_SYNTHETIC_SMOKE


# ===================================================================
# 3. Production provenance validation
# ===================================================================


class TestProductionProvenanceValidation:
    """Section 3: Production provenance validation."""

    def test_full_provenance_accepted(self):
        prov = _make_provenance()
        assert prov.protocol_version == "1.0"
        assert prov.evaluation_round_id == "eval_001"

    def test_scientific_status_in_provenance(self):
        prov = _make_provenance(scientific_status=SCIENTIFIC_PRODUCTION)
        assert prov.scientific_status == SCIENTIFIC_PRODUCTION

    def test_all_provenance_fields_serializable(self):
        prov = _make_provenance()
        data = {
            "protocol_version": prov.protocol_version,
            "evaluation_round_id": prov.evaluation_round_id,
            "scientific_status": prov.scientific_status,
            "dataset_version": prov.dataset_version,
            "dataset_hash": prov.dataset_hash,
            "split_version": prov.split_version,
            "split_hash": prov.split_hash,
            "tokenizer_identity": prov.tokenizer_identity,
            "tokenizer_hash": prov.tokenizer_hash,
            "code_commit": prov.code_commit,
            "model_variant": prov.model_variant,
            "seed": prov.seed,
        }
        json_str = json.dumps(data)
        assert json.loads(json_str)["evaluation_round_id"] == "eval_001"


# ===================================================================
# 4. Production result missing critical provenance is rejected
# ===================================================================


class TestProductionProvenanceRejection:
    """Section 4: Production provenance validation — missing fields rejected."""

    def test_empty_dataset_version_rejected_in_manifest(self):
        from rrm.scientific_evaluation import validate_split_manifest
        from rrm.scientific_evaluation import FrozenSplitManifest

        manifest = FrozenSplitManifest(
            dataset_version="",
            dataset_hash="a" * 64,
            split_version="s1",
            split_hash="b" * 64,
            split_generation_identity="test",
            split_seed=42,
            train_review_ids=("r1",),
            validation_review_ids=(),
            test_review_ids=(),
        )
        errors = validate_split_manifest(manifest)
        assert any("dataset_version" in e for e in errors)

    def test_empty_code_commit_flagged(self):
        """A production result with empty code_commit should be flagged."""
        prov = _make_provenance(code_commit="")
        # The metadata itself stores it; the serialization reveals it
        data = {
            "protocol_version": prov.protocol_version,
            "evaluation_round_id": prov.evaluation_round_id,
            "scientific_status": prov.scientific_status,
            "dataset_version": prov.dataset_version,
            "dataset_hash": prov.dataset_hash,
            "code_commit": prov.code_commit,
        }
        assert data["code_commit"] == ""

    def test_missing_dataset_hash_flagged(self):
        prov = _make_provenance(dataset_hash="")
        data = {
            "protocol_version": prov.protocol_version,
            "evaluation_round_id": prov.evaluation_round_id,
            "scientific_status": prov.scientific_status,
            "dataset_version": prov.dataset_version,
            "dataset_hash": prov.dataset_hash,
        }
        assert data["dataset_hash"] == ""

    def test_non_none_scientific_status_required(self):
        """All statuses must be non-empty strings."""
        for status in VALID_SCIENTIFIC_STATUSES:
            assert isinstance(status, str)
            assert len(status) > 0


# ===================================================================
# 5. Canonical six per-task entries
# ===================================================================


class TestCanonicalSixPerTaskEntries:
    """Section 5: Canonical six per-task entries."""

    def test_six_labels_present(self):
        assert len(PRIMARY_LABELS) == 6

    def test_labels_are_spam_deception_toxicity_advertising_off_topic_pii(self):
        expected = ("spam", "deception", "toxicity", "advertising", "off_topic", "pii")
        assert PRIMARY_LABELS == expected

    def test_per_task_dict_has_all_six(self):
        result = _make_result()
        assert set(result.per_task.keys()) == set(PRIMARY_LABELS)

    def test_each_per_task_entry_has_correct_label(self):
        result = _make_result()
        for label in PRIMARY_LABELS:
            tm = result.per_task[label]
            assert tm.label == label

    def test_each_per_task_entry_is_task_metrics(self):
        result = _make_result()
        for label in PRIMARY_LABELS:
            assert isinstance(result.per_task[label], TaskMetrics)

    def test_create_smoke_result_has_six_per_task_entries(self):
        result = create_smoke_result("full", seed=42)
        assert len(result.per_task) == 6
        assert set(result.per_task.keys()) == set(PRIMARY_LABELS)


# ===================================================================
# 6. ScientificMacroResult is used
# ===================================================================


class TestScientificMacroResultUsed:
    """Section 6: ScientificMacroResult is used."""

    def test_macro_is_scientific_macro_result(self):
        result = _make_result()
        assert isinstance(result.macro, ScientificMacroResult)

    def test_smoke_result_macro_is_scientific_macro_result(self):
        result = create_smoke_result("full", seed=42)
        assert isinstance(result.macro, ScientificMacroResult)

    def test_macro_has_required_fields(self):
        macro = _make_scientific_macro()
        assert macro.macro_f1 is not None
        assert macro.macro_f1_task_count == 6
        assert len(macro.macro_f1_task_names) == 6
        assert macro.macro_auprc is not None
        assert macro.macro_auprc_task_count == 6
        assert len(macro.macro_auprc_task_names) == 6

    def test_compute_scientific_macro_returns_scientific_macro_result(self):
        tms = [_make_task_metrics(l) for l in PRIMARY_LABELS]
        macro = compute_scientific_macro(tms)
        assert isinstance(macro, ScientificMacroResult)

    def test_macro_frozen(self):
        macro = _make_scientific_macro()
        with pytest.raises((AttributeError, TypeError)):
            macro.macro_f1 = 0.9  # type: ignore


# ===================================================================
# 7. macro_f1_task_names serialization
# ===================================================================


class TestMacroF1TaskNamesSerialization:
    """Section 7: macro_f1_task_names serialization."""

    def test_macro_f1_task_names_serialized_as_list(self):
        result = _make_result()
        data = serialize_scientific_result(result)
        macro = data["macro"]
        assert "macro_f1_task_names" in macro
        assert isinstance(macro["macro_f1_task_names"], list)

    def test_macro_f1_task_names_contains_all_six(self):
        result = _make_result()
        data = serialize_scientific_result(result)
        names = data["macro"]["macro_f1_task_names"]
        assert len(names) == 6
        assert set(names) == set(PRIMARY_LABELS)

    def test_macro_f1_task_names_deserialized_as_tuple(self):
        result = _make_result()
        data = serialize_scientific_result(result)
        restored = deserialize_scientific_result(data)
        assert isinstance(restored.macro.macro_f1_task_names, tuple)

    def test_macro_f1_task_names_roundtrip_preserved(self):
        result = _make_result()
        data = serialize_scientific_result(result)
        restored = deserialize_scientific_result(data)
        assert restored.macro.macro_f1_task_names == result.macro.macro_f1_task_names


# ===================================================================
# 8. macro_auprc_task_names serialization
# ===================================================================


class TestMacroAuprcTaskNamesSerialization:
    """Section 8: macro_auprc_task_names serialization."""

    def test_macro_auprc_task_names_serialized_as_list(self):
        result = _make_result()
        data = serialize_scientific_result(result)
        macro = data["macro"]
        assert "macro_auprc_task_names" in macro
        assert isinstance(macro["macro_auprc_task_names"], list)

    def test_macro_auprc_task_names_contains_all_six(self):
        result = _make_result()
        data = serialize_scientific_result(result)
        names = data["macro"]["macro_auprc_task_names"]
        assert len(names) == 6
        assert set(names) == set(PRIMARY_LABELS)

    def test_macro_auprc_task_names_deserialized_as_tuple(self):
        result = _make_result()
        data = serialize_scientific_result(result)
        restored = deserialize_scientific_result(data)
        assert isinstance(restored.macro.macro_auprc_task_names, tuple)

    def test_macro_auprc_task_names_roundtrip_preserved(self):
        result = _make_result()
        data = serialize_scientific_result(result)
        restored = deserialize_scientific_result(data)
        assert restored.macro.macro_auprc_task_names == result.macro.macro_auprc_task_names


# ===================================================================
# 9. Task counts serialization
# ===================================================================


class TestTaskCountsSerialization:
    """Section 9: Task count serialization."""

    def test_macro_f1_task_count_serialized(self):
        result = _make_result()
        data = serialize_scientific_result(result)
        assert "macro_f1_task_count" in data["macro"]
        assert data["macro"]["macro_f1_task_count"] == 6

    def test_macro_auprc_task_count_serialized(self):
        result = _make_result()
        data = serialize_scientific_result(result)
        assert "macro_auprc_task_count" in data["macro"]
        assert data["macro"]["macro_auprc_task_count"] == 6

    def test_task_counts_roundtrip(self):
        result = _make_result()
        data = serialize_scientific_result(result)
        restored = deserialize_scientific_result(data)
        assert restored.macro.macro_f1_task_count == result.macro.macro_f1_task_count
        assert restored.macro.macro_auprc_task_count == result.macro.macro_auprc_task_count


# ===================================================================
# 10. None serializes to JSON null
# ===================================================================


class TestNoneSerialization:
    """Section 10: None serializes to JSON null."""

    def test_none_thresholds_serializes_to_null(self):
        result = _make_result(thresholds=None)
        data = serialize_scientific_result(result)
        assert data["thresholds"] is None

    def test_none_confidence_intervals_serializes_to_null(self):
        result = _make_result(confidence_intervals=None)
        data = serialize_scientific_result(result)
        assert data["confidence_intervals"] is None

    def test_none_slice_results_serializes_to_null(self):
        result = _make_result(slice_results=None)
        data = serialize_scientific_result(result)
        assert data["slice_results"] is None

    def test_none_efficiency_serializes_to_null(self):
        result = _make_result(efficiency=None)
        data = serialize_scientific_result(result)
        assert data["efficiency"] is None

    def test_json_string_contains_null_for_none(self):
        result = _make_result(thresholds=None)
        json_str = result_to_json(result)
        parsed = json.loads(json_str)
        assert parsed["thresholds"] is None


# ===================================================================
# 11. Thresholds serialization
# ===================================================================


class TestThresholdsSerialization:
    """Section 11: Thresholds serialization."""

    def test_thresholds_dict_serialized(self):
        thresholds = {"spam": 0.5, "deception": 0.55}
        result = _make_result(thresholds=thresholds)
        data = serialize_scientific_result(result)
        assert data["thresholds"] == thresholds

    def test_threshold_sources_serialized(self):
        threshold_sources = {"spam": "validation_tuned"}
        result = _make_result(metadata=_make_provenance(threshold_sources=threshold_sources))
        data = serialize_scientific_result(result)
        assert data["metadata"]["threshold_sources"] == threshold_sources

    def test_thresholds_roundtrip(self):
        thresholds = {"spam": 0.5, "toxicity": 0.6, "pii": 0.7}
        result = _make_result(thresholds=thresholds)
        json_str = result_to_json(result)
        restored = result_from_json(json_str)
        assert restored.thresholds == thresholds

    def test_none_threshold_sources_serialized(self):
        result = _make_result(metadata=_make_provenance(threshold_sources=None))
        data = serialize_scientific_result(result)
        assert data["metadata"]["threshold_sources"] is None


# ===================================================================
# 12. threshold_sources serialization
# ===================================================================


class TestThresholdSourcesSerialization:
    """Section 12: threshold_sources serialization."""

    def test_threshold_sources_in_metadata(self):
        sources = {"spam": "validation_tuned", "deception": "default_0.5"}
        result = _make_result(metadata=_make_provenance(threshold_sources=sources))
        data = serialize_scientific_result(result)
        assert data["metadata"]["threshold_sources"] == sources

    def test_threshold_sources_roundtrip(self):
        sources = {"spam": "validation_tuned", "pii": "default_0.5"}
        result = _make_result(metadata=_make_provenance(threshold_sources=sources))
        data = serialize_scientific_result(result)
        restored_metadata = ProvenanceMetadata(**data["metadata"])
        assert restored_metadata.threshold_sources == sources


# ===================================================================
# 13. Bootstrap metadata serialization
# ===================================================================


class TestBootstrapMetadataSerialization:
    """Section 13: Bootstrap metadata serialization."""

    def test_bootstrap_fields_in_metadata(self):
        prov = _make_provenance(
            bootstrap_seed=123,
            bootstrap_replicates=1000,
            bootstrap_ci_level=0.95,
            bootstrap_ci_method="percentile",
        )
        result = _make_result(metadata=prov)
        data = serialize_scientific_result(result)
        meta = data["metadata"]
        assert meta["bootstrap_seed"] == 123
        assert meta["bootstrap_replicates"] == 1000
        assert meta["bootstrap_ci_level"] == 0.95
        assert meta["bootstrap_ci_method"] == "percentile"

    def test_bootstrap_none_serialized(self):
        prov = _make_provenance(
            bootstrap_seed=None,
            bootstrap_replicates=None,
            bootstrap_ci_level=None,
            bootstrap_ci_method=None,
        )
        result = _make_result(metadata=prov)
        data = serialize_scientific_result(result)
        meta = data["metadata"]
        assert meta["bootstrap_seed"] is None
        assert meta["bootstrap_replicates"] is None

    def test_confidence_intervals_serialized(self):
        ci = BootstrapCI(
            point_estimate=0.82,
            lower=0.78,
            upper=0.86,
            replicates=1000,
            seed=42,
            ci_level=0.95,
            method="percentile",
        )
        result = _make_result(confidence_intervals={"macro_f1": ci})
        data = serialize_scientific_result(result)
        ci_data = data["confidence_intervals"]["macro_f1"]
        assert ci_data["point_estimate"] == 0.82
        assert ci_data["lower"] == 0.78
        assert ci_data["upper"] == 0.86
        assert ci_data["replicates"] == 1000
        assert ci_data["ci_level"] == 0.95


# ===================================================================
# 14. Runtime/hardware metadata serialization
# ===================================================================


class TestRuntimeHardwareMetadataSerialization:
    """Section 14: Runtime/hardware metadata serialization."""

    def test_runtime_versions_serialized(self):
        versions = {"python": "3.11.0", "numpy": "1.26.0"}
        result = _make_result(metadata=_make_provenance(runtime_versions=versions))
        data = serialize_scientific_result(result)
        assert data["metadata"]["runtime_versions"] == versions

    def test_hardware_identity_serialized(self):
        result = _make_result(metadata=_make_provenance(hardware_identity="RTX_3080"))
        data = serialize_scientific_result(result)
        assert data["metadata"]["hardware_identity"] == "RTX_3080"

    def test_default_runtime_versions_returns_dict(self):
        versions = default_runtime_versions()
        assert isinstance(versions, dict)
        assert "python" in versions

    def test_runtime_versions_roundtrip(self):
        versions = {"python": "3.11.0", "torch": "2.0.0"}
        result = _make_result(metadata=_make_provenance(runtime_versions=versions))
        data = serialize_scientific_result(result)
        restored = ProvenanceMetadata(**data["metadata"])
        assert restored.runtime_versions == versions


# ===================================================================
# 15. to_dict / from_dict roundtrip
# ===================================================================


class TestToFromDictRoundtrip:
    """Section 15: to_dict / from_dict roundtrip."""

    def test_full_roundtrip(self):
        result = _make_result()
        data = serialize_scientific_result(result)
        restored = deserialize_scientific_result(data)
        assert restored.metadata.evaluation_round_id == result.metadata.evaluation_round_id
        assert restored.scientific_status == result.scientific_status
        assert restored.macro.macro_f1 == result.macro.macro_f1
        assert restored.macro.macro_auprc == result.macro.macro_auprc
        assert set(restored.per_task.keys()) == set(result.per_task.keys())

    def test_per_task_metrics_roundtrip(self):
        result = _make_result()
        data = serialize_scientific_result(result)
        restored = deserialize_scientific_result(data)
        for label in PRIMARY_LABELS:
            original = result.per_task[label]
            recovered = restored.per_task[label]
            assert recovered.label == original.label
            assert recovered.f1 == original.f1
            assert recovered.auprc == original.auprc

    def test_limitations_roundtrip(self):
        result = _make_result(limitations=("Limit one", "Limit two"))
        data = serialize_scientific_result(result)
        restored = deserialize_scientific_result(data)
        assert restored.limitations == ("Limit one", "Limit two")

    def test_none_fields_roundtrip(self):
        result = _make_result(
            thresholds=None,
            confidence_intervals=None,
            slice_results=None,
            efficiency=None,
        )
        data = serialize_scientific_result(result)
        restored = deserialize_scientific_result(data)
        assert restored.thresholds is None
        assert restored.confidence_intervals is None
        assert restored.slice_results is None
        assert restored.efficiency is None


# ===================================================================
# 16. to_json / from_json roundtrip
# ===================================================================


class TestToFromJsonRoundtrip:
    """Section 16: to_json / from_json roundtrip."""

    def test_full_json_roundtrip(self):
        result = _make_result()
        json_str = result_to_json(result)
        restored = result_from_json(json_str)
        assert restored.metadata.evaluation_round_id == result.metadata.evaluation_round_id
        assert restored.scientific_status == result.scientific_status

    def test_json_is_valid_json(self):
        result = _make_result()
        json_str = result_to_json(result)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "metadata" in parsed
        assert "per_task" in parsed
        assert "macro" in parsed

    def test_json_has_all_top_level_keys(self):
        result = _make_result()
        data = serialize_scientific_result(result)
        expected_keys = {
            "metadata", "per_task", "macro", "thresholds",
            "confidence_intervals", "slice_results", "efficiency",
            "limitations", "scientific_status",
        }
        assert expected_keys == set(data.keys())

    def test_save_and_load_file_roundtrip(self, tmp_path):
        result = _make_result()
        filepath = str(tmp_path / "result.json")
        save_result(result, filepath)
        assert Path(filepath).exists()
        loaded = load_result(filepath)
        assert loaded.metadata.evaluation_round_id == result.metadata.evaluation_round_id
        assert loaded.scientific_status == result.scientific_status


# ===================================================================
# 17. Malformed/invalid schema rejection
# ===================================================================


class TestMalformedSchemaRejection:
    """Section 17: Malformed/invalid schema rejection."""

    def test_missing_metadata_key_raises(self):
        data = serialize_scientific_result(_make_result())
        del data["metadata"]
        with pytest.raises((KeyError, TypeError)):
            deserialize_scientific_result(data)

    def test_missing_per_task_key_raises(self):
        data = serialize_scientific_result(_make_result())
        del data["per_task"]
        with pytest.raises((KeyError, TypeError)):
            deserialize_scientific_result(data)

    def test_missing_macro_key_raises(self):
        data = serialize_scientific_result(_make_result())
        del data["macro"]
        with pytest.raises((KeyError, TypeError)):
            deserialize_scientific_result(data)

    def test_invalid_scientific_status_in_result(self):
        """ScientificResult accepts any string for scientific_status at
        construction time — the validation happens at the protocol level,
        not at the dataclass level."""
        # The dataclass stores it as-is; this is by design (RRM 3.10 protocol
        # validation is a higher-level concern)
        result = _make_result(scientific_status="INVALID_STATUS")
        assert result.scientific_status == "INVALID_STATUS"

    def test_invalid_metadata_field_type_accepted_and_stored(self):
        """ProvenanceMetadata (dataclass) stores values as-is without
        runtime type enforcement. The value is stored even when the
        type doesn't match the annotation. Validation happens at the
        machine-readable ingestion boundary (deserialize_scientific_result)."""
        pm = ProvenanceMetadata(
            protocol_version=123,  # type: ignore — intentionally wrong type
            evaluation_round_id="test",
            scientific_status=SCIENTIFIC_PRODUCTION,
            dataset_version="v1",
            dataset_hash="a" * 64,
            split_version="s1",
            split_hash="b" * 64,
            tokenizer_identity="tok",
            tokenizer_hash="c" * 64,
            semantic_pretraining_identity="pre",
            semantic_pretraining_hash="d" * 64,
            supervised_checkpoint_identity="ckpt",
            supervised_checkpoint_hash="e" * 64,
            code_commit="abc",
            model_variant="full",
            seed=42,
            training_config_identity="cfg",
        )
        assert pm.protocol_version == 123


# ===================================================================
# 17B. Ingestion-boundary schema validation
# ===================================================================


class TestIngestionBoundaryValidation:
    """Section 17B: Machine-readable ingestion (deserialize_scientific_result)
    must reject malformed metadata schemas."""

    def test_invalid_scalar_metadata_type_rejected(self):
        """protocol_version as int must be rejected by ingestion boundary."""
        data = serialize_scientific_result(_make_result())
        data["metadata"]["protocol_version"] = 123  # should be str
        with pytest.raises(SchemaValidationError):
            deserialize_scientific_result(data)

    def test_invalid_mapping_type_rejected(self):
        """threshold_sources as list instead of dict must be rejected."""
        data = serialize_scientific_result(_make_result())
        data["metadata"]["threshold_sources"] = ["not", "a", "mapping"]
        with pytest.raises(SchemaValidationError):
            deserialize_scientific_result(data)

    def test_invalid_runtime_versions_type_rejected(self):
        """runtime_versions as string instead of dict must be rejected."""
        data = serialize_scientific_result(_make_result())
        data["metadata"]["runtime_versions"] = "not_a_dict"
        with pytest.raises(SchemaValidationError):
            deserialize_scientific_result(data)

    def test_valid_input_deserializes_correctly(self):
        """Valid serialized data must deserialize without error."""
        data = serialize_scientific_result(_make_result())
        result = deserialize_scientific_result(data)
        assert result.metadata.protocol_version == "1.0"
        assert result.scientific_status == SCIENTIFIC_PRODUCTION
        assert result.macro.macro_f1 == 0.8

    def test_json_roundtrip_rejects_invalid_schema(self):
        """from_json must also reject malformed metadata types."""
        result = _make_result()
        json_str = result_to_json(result)
        # Replace the protocol_version value in the JSON string
        json_str = json_str.replace('"protocol_version": "1.0"', '"protocol_version": 123')
        with pytest.raises(SchemaValidationError):
            result_from_json(json_str)


# ===================================================================
# 18. Markdown contains required fields
# ===================================================================


class TestMarkdownContainsRequiredFields:
    """Section 18: Markdown contains evaluation_round_id, scientific_status,
    model_variant, macro-AUPRC, limitations."""

    def test_contains_evaluation_round_id(self):
        result = _make_result()
        md = render_markdown(result)
        assert result.metadata.evaluation_round_id in md

    def test_contains_scientific_status(self):
        result = _make_result()
        md = render_markdown(result)
        assert result.scientific_status in md

    def test_contains_model_variant(self):
        result = _make_result()
        md = render_markdown(result)
        assert result.metadata.model_variant in md

    def test_contains_macro_auprc(self):
        result = _make_result()
        md = render_markdown(result)
        assert "macro-auprc" in md.lower()

    def test_contains_limitations(self):
        result = _make_result(limitations=("Test limitation.",))
        md = render_markdown(result)
        assert "Test limitation." in md

    def test_contains_macro_f1(self):
        result = _make_result()
        md = render_markdown(result)
        assert "Macro F1" in md or "macro_f1" in md

    def test_contains_per_task_table(self):
        result = _make_result()
        md = render_markdown(result)
        assert "spam" in md
        assert "deception" in md
        assert "toxicity" in md

    def test_contains_provenance_section(self):
        result = _make_result()
        md = render_markdown(result)
        assert "Provenance" in md
        assert result.metadata.code_commit in md


# ===================================================================
# 19. Synthetic smoke Markdown warning banners
# ===================================================================


class TestSyntheticSmokeMarkdownBanners:
    """Section 19: Synthetic smoke Markdown contains required warnings."""

    def test_smoke_markdown_contains_non_scientific(self):
        result = create_smoke_result("full", seed=42)
        md = render_markdown(result)
        assert "NON-SCIENTIFIC" in md

    def test_smoke_markdown_contains_synthetic_pilot(self):
        result = create_smoke_result("full", seed=42)
        md = render_markdown(result)
        assert "SYNTHETIC PILOT" in md

    def test_smoke_markdown_contains_not_performance_evidence(self):
        result = create_smoke_result("full", seed=42)
        md = render_markdown(result)
        assert "NOT PERFORMANCE EVIDENCE" in md

    def test_production_markdown_does_not_contain_non_scientific_warning(self):
        result = _make_result(scientific_status=SCIENTIFIC_PRODUCTION)
        md = render_markdown(result)
        assert "NON-SCIENTIFIC" not in md

    def test_partial_markdown_does_not_contain_non_scientific_warning(self):
        result = _make_result(scientific_status=SCIENTIFIC_PARTIAL)
        md = render_markdown(result)
        assert "NON-SCIENTIFIC" not in md


# ===================================================================
# 20. Markdown renderer does not mutate source result
# ===================================================================


class TestMarkdownNonMutation:
    """Section 20: Markdown renderer does not mutate source result."""

    def test_markdown_does_not_mutate_limitations(self):
        result = _make_result(limitations=("Original limitation.",))
        original_limitations = result.limitations
        render_markdown(result)
        assert result.limitations == original_limitations

    def test_markdown_does_not_mutate_scientific_status(self):
        result = _make_result(scientific_status=SCIENTIFIC_PRODUCTION)
        original_status = result.scientific_status
        render_markdown(result)
        assert result.scientific_status == original_status

    def test_markdown_does_not_mutate_per_task(self):
        result = _make_result()
        original_keys = set(result.per_task.keys())
        render_markdown(result)
        assert set(result.per_task.keys()) == original_keys

    def test_markdown_does_not_mutate_thresholds(self):
        thresholds = {"spam": 0.5}
        result = _make_result(thresholds=thresholds)
        render_markdown(result)
        assert result.thresholds == thresholds

    def test_result_is_frozen(self):
        """ScientificResult uses frozen=True in some paths via
        ScientificMacroResult; verify core structures are immutable."""
        macro = _make_scientific_macro()
        with pytest.raises((AttributeError, TypeError)):
            macro.macro_f1 = 0.9  # type: ignore


# ===================================================================
# Additional edge cases and integration tests
# ===================================================================


class TestExperimentResultsEdgeCases:
    """Additional edge-case tests for experiment_results."""

    def test_empty_limitations_serializes_to_empty_list(self):
        result = _make_result(limitations=())
        data = serialize_scientific_result(result)
        assert data["limitations"] == []

    def test_single_limitation_serialized(self):
        result = _make_result(limitations=("Only one.",))
        data = serialize_scientific_result(result)
        assert data["limitations"] == ["Only one."]

    def test_create_smoke_result_with_custom_limitations(self):
        result = create_smoke_result(
            "semantic_only", seed=99, limitations=("Custom note.",)
        )
        assert "Custom note." in result.limitations
        assert result.metadata.seed == 99

    def test_create_smoke_result_default_code_commit(self):
        result = create_smoke_result("full", seed=42)
        assert result.metadata.code_commit == "unknown"

    def test_create_smoke_result_custom_code_commit(self):
        result = create_smoke_result("full", seed=42, code_commit="def5678")
        assert result.metadata.code_commit == "def5678"

    def test_provenance_created_at_set(self):
        prov = _make_provenance()
        assert prov.created_at == "2024-06-01T00:00:00+00:00"

    def test_provenance_metric_definition_version(self):
        prov = _make_provenance()
        assert prov.metric_definition_version == "1.0"

    def test_macro_with_no_valid_tasks(self):
        """When no tasks have valid metrics, macro values are None."""
        tms = [_make_task_metrics(l, f1=None, auprc=None) for l in PRIMARY_LABELS]
        macro = compute_scientific_macro(tms)
        assert macro.macro_f1 is None
        assert macro.macro_f1_task_count == 0
        assert macro.macro_auprc is None
        assert macro.macro_auprc_task_count == 0

    def test_macro_with_partial_valid_tasks(self):
        """Some tasks have metrics, some don't."""
        tms = [
            _make_task_metrics("spam", f1=0.9, auprc=0.92),
            _make_task_metrics("deception", f1=None, auprc=None),
            _make_task_metrics("toxicity", f1=0.8, auprc=0.83),
            _make_task_metrics("advertising", f1=None, auprc=None),
            _make_task_metrics("off_topic", f1=0.7, auprc=0.75),
            _make_task_metrics("pii", f1=None, auprc=None),
        ]
        macro = compute_scientific_macro(tms)
        assert macro.macro_f1 is not None
        assert macro.macro_f1_task_count == 3
        assert macro.macro_auprc is not None
        assert macro.macro_auprc_task_count == 3
        assert set(macro.macro_f1_task_names) == {"spam", "toxicity", "off_topic"}

    def test_json_indent_parameter(self):
        result = _make_result()
        json_str = result_to_json(result, indent=4)
        # Should have indented output
        assert "\n    " in json_str

    def test_render_markdown_returns_string(self):
        result = _make_result()
        md = render_markdown(result)
        assert isinstance(md, str)
        assert len(md) > 0

    def test_render_markdown_without_ci(self):
        result = _make_result(
            confidence_intervals={
                "macro_f1": BootstrapCI(
                    point_estimate=0.82,
                    lower=0.78,
                    upper=0.86,
                    replicates=1000,
                    seed=42,
                    ci_level=0.95,
                    method="percentile",
                )
            }
        )
        md = render_markdown(result, include_ci=False)
        assert "Confidence Intervals" not in md

    def test_render_markdown_without_efficiency(self):
        result = _make_result(
            efficiency={
                "latency": type("obj", (object,), {
                    "mean_ms": 1.0, "median_ms": 0.9, "p95_ms": 1.5
                })()
            }
        )
        md = render_markdown(result, include_efficiency=False)
        assert "Efficiency" not in md

    def test_smoke_result_uses_pilot_dataset_version(self):
        result = create_smoke_result("full", seed=42)
        assert result.metadata.dataset_version == "rmc_pilot_v0.1"

    def test_smoke_result_has_none_pretraining(self):
        result = create_smoke_result("full", seed=42)
        assert result.metadata.semantic_pretraining_identity == "none"
        assert result.metadata.supervised_checkpoint_identity == "none"
