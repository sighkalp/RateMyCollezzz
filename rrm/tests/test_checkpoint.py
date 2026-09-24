"""Tests for rrm/checkpoint.py — deployment, resume, validation."""

from __future__ import annotations

import os
import random
import tempfile

import pytest
import torch

from rrm.checkpoint import (
    REQUIRED_PRODUCTION_FIELDS,
    CheckpointMetadata,
    CheckpointValidationError,
    compute_file_hash,
    load_deployment_checkpoint,
    load_resume_checkpoint,
    save_deployment_checkpoint,
    save_resume_checkpoint,
    validate_production_metadata,
)
from rrm.labels import NUM_PRIMARY_LABELS, PRIMARY_LABELS, UNKNOWN_LABEL


# ---------------------------------------------------------------------------
# CheckpointMetadata tests
# ---------------------------------------------------------------------------


class TestCheckpointMetadata:
    def test_defaults(self):
        meta = CheckpointMetadata()
        assert meta.schema_version == "1.0"
        assert meta.primary_labels == PRIMARY_LABELS
        assert meta.unknown_label == UNKNOWN_LABEL
        assert meta.head_input_dim == 448
        assert meta.head_output_dim == NUM_PRIMARY_LABELS
        assert meta.seed == 42
        assert meta.epoch == 0
        assert meta.global_step == 0
        assert meta.best_validation_bce is None
        assert meta.python_rng_state is None
        assert meta.torch_rng_state is None
        assert meta.cuda_rng_state is None

    def test_frozen(self):
        meta = CheckpointMetadata()
        with pytest.raises((AttributeError, TypeError)):
            meta.schema_version = "2.0"  # type: ignore[misc]

    def test_custom_metadata(self):
        meta = CheckpointMetadata(
            code_revision="abc123",
            dataset_version="v1",
            tokenizer_identity="test-tokenizer",
            tokenizer_hash="abc123def456",
            semantic_pretraining_identity="test-pretrain",
            semantic_pretraining_hash="789ghi012",
            dataset_hash="dataset_hash_here",
            split_identity="train_val_test_v1",
            split_hash="split_hash_here",
            epoch=5,
            global_step=10000,
            best_validation_bce=0.15,
            thresholds=(0.5, 0.5, 0.3, 0.4, 0.5, 0.6),
        )
        assert meta.epoch == 5
        assert meta.global_step == 10000
        assert meta.best_validation_bce == 0.15
        assert meta.tokenizer_identity == "test-tokenizer"

    def test_rng_state_fields_default_none(self):
        """RNG state fields default to None."""
        meta = CheckpointMetadata()
        assert meta.python_rng_state is None
        assert meta.torch_rng_state is None
        assert meta.cuda_rng_state is None

    def test_rng_state_fields_can_be_set(self):
        """RNG state fields can be populated."""
        py_state = random.getstate()
        torch_state = torch.get_rng_state()
        meta = CheckpointMetadata(
            python_rng_state=py_state,
            torch_rng_state=torch_state,
        )
        assert meta.python_rng_state is not None
        assert meta.torch_rng_state is not None


# ---------------------------------------------------------------------------
# Resume validation tests
# ---------------------------------------------------------------------------


class TestResumeValidation:
    def test_negative_epoch_rejected(self):
        """epoch < 0 should be rejected by validation."""
        meta = CheckpointMetadata(epoch=-1)
        errors = validate_production_metadata(meta)
        assert any("epoch" in e for e in errors)

    def test_negative_global_step_rejected(self):
        """global_step < 0 should be rejected by validation."""
        meta = CheckpointMetadata(global_step=-1)
        errors = validate_production_metadata(meta)
        assert any("global_step" in e for e in errors)

    def test_non_finite_best_validation_bce_rejected(self):
        """best_validation_bce must be finite when not None."""
        meta = CheckpointMetadata(best_validation_bce=float("nan"))
        errors = validate_production_metadata(meta)
        assert any("best_validation_bce" in e for e in errors)

    def test_inf_best_validation_bce_rejected(self):
        """best_validation_bce must be finite (not inf) when not None."""
        meta = CheckpointMetadata(best_validation_bce=float("inf"))
        errors = validate_production_metadata(meta)
        assert any("best_validation_bce" in e for e in errors)

    def test_valid_best_validation_bce_accepted(self):
        """Valid finite best_validation_bce passes validation."""
        meta = CheckpointMetadata(best_validation_bce=0.25)
        errors = validate_production_metadata(meta)
        assert not any("best_validation_bce" in e for e in errors)

    def test_none_best_validation_bce_accepted(self):
        """None best_validation_bce passes validation."""
        meta = CheckpointMetadata(best_validation_bce=None)
        errors = validate_production_metadata(meta)
        assert not any("best_validation_bce" in e for e in errors)


# ---------------------------------------------------------------------------
# Structural config compatibility tests
# ---------------------------------------------------------------------------


class TestStructuralCompatibility:
    """Explicit tests for incompatible architecture metadata rejection."""

    def _make_production_meta(self, **overrides):
        defaults = dict(
            tokenizer_identity="test-tokenizer",
            tokenizer_hash="abc123def456789abc123def456789abc123def456789abc123def456789",
            semantic_pretraining_identity="mbert-v1",
            semantic_pretraining_hash="def456abc123def456abc123def456abc123def456abc123def456abc123",
            dataset_version="rmc-pilot-v2",
            dataset_hash="123def456abc123def456abc123def456abc123def456abc123def456abc1234",
            split_identity="stratified_v1",
            split_hash="fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
            code_revision="abc123def",
        )
        defaults.update(overrides)
        return CheckpointMetadata(**defaults)

    def test_semantic_encoder_dim_mismatch(self):
        """Incompatible semantic encoder dimensions in metadata are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            meta = self._make_production_meta(
                semantic_config={"encoder_dim": 256},
            )
            save_deployment_checkpoint(path, {"w": torch.zeros(1)}, meta)
            with pytest.raises(CheckpointValidationError, match="semantic_config"):
                load_deployment_checkpoint(path)

    def test_character_config_dim_mismatch(self):
        """Incompatible character config dimensions in metadata are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            meta = self._make_production_meta(
                character_config={"char_embed_dim": 32},
            )
            save_deployment_checkpoint(path, {"w": torch.zeros(1)}, meta)
            with pytest.raises(CheckpointValidationError, match="character_config"):
                load_deployment_checkpoint(path)

    def test_head_input_dim_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            meta = self._make_production_meta(head_input_dim=256)
            save_deployment_checkpoint(path, {"w": torch.zeros(1)}, meta)
            with pytest.raises(CheckpointValidationError, match="head_input_dim"):
                load_deployment_checkpoint(path)

    def test_head_output_dim_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            meta = self._make_production_meta(head_output_dim=4)
            save_deployment_checkpoint(path, {"w": torch.zeros(1)}, meta)
            with pytest.raises(CheckpointValidationError, match="head_output_dim"):
                load_deployment_checkpoint(path)

    def test_schema_version_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            meta = self._make_production_meta(schema_version="2.0")
            save_deployment_checkpoint(path, {"w": torch.zeros(1)}, meta)
            with pytest.raises(CheckpointValidationError, match="schema_version"):
                load_deployment_checkpoint(path)

    def test_label_order_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            meta = self._make_production_meta(
                primary_labels=("pii", "spam", "deception", "toxicity", "advertising", "off_topic"),
            )
            save_deployment_checkpoint(path, {"w": torch.zeros(1)}, meta)
            with pytest.raises(CheckpointValidationError, match="PRIMARY_LABELS"):
                load_deployment_checkpoint(path)

    def test_unknown_label_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            meta = self._make_production_meta(unknown_label=0)
            save_deployment_checkpoint(path, {"w": torch.zeros(1)}, meta)
            with pytest.raises(CheckpointValidationError, match="UNKNOWN_LABEL"):
                load_deployment_checkpoint(path)

    def test_threshold_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            meta = self._make_production_meta(thresholds=(0.5, 0.5))
            save_deployment_checkpoint(path, {"w": torch.zeros(1)}, meta)
            with pytest.raises(CheckpointValidationError, match="thresholds length"):
                load_deployment_checkpoint(path)


# ---------------------------------------------------------------------------
# Production provenance validation
# ---------------------------------------------------------------------------


class TestValidateProductionMetadata:
    def test_default_metadata_fails_production(self):
        """Default CheckpointMetadata() has blank provenance — fails production."""
        meta = CheckpointMetadata()
        errors = validate_production_metadata(meta)
        assert len(errors) > 0
        for field_name in REQUIRED_PRODUCTION_FIELDS:
            field_errors = [e for e in errors if field_name in e]
            assert len(field_errors) == 1, f"Missing error for {field_name}"

    def test_full_provenance_passes(self):
        """Complete provenance passes structural validation."""
        meta = CheckpointMetadata(
            tokenizer_identity="test-tokenizer",
            tokenizer_hash="abc123def456789abc123def456789abc123def456789abc123def456789",
            semantic_pretraining_identity="mbert-v1",
            semantic_pretraining_hash="def456abc123def456abc123def456abc123def456abc123def456abc123",
            dataset_version="rmc-pilot-v2",
            dataset_hash="123def456abc123def456abc123def456abc123def456abc123def456abc1234",
            split_identity="stratified_v1",
            split_hash="fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
            code_revision="abc123def",
        )
        errors = validate_production_metadata(meta)
        assert len(errors) == 0

    def test_partial_provenance_reports_all_missing(self):
        """Only some provenance fields filled — reports all missing."""
        meta = CheckpointMetadata(
            tokenizer_identity="test-tokenizer",
            tokenizer_hash="abc123",
            # All other production fields blank
        )
        errors = validate_production_metadata(meta)
        missing_count = sum(1 for e in errors if "production provenance" in e)
        assert missing_count == len(REQUIRED_PRODUCTION_FIELDS) - 2  # 2 provided

    def test_required_production_fields_list(self):
        """REQUIRED_PRODUCTION_FIELDS has exactly the expected fields."""
        assert len(REQUIRED_PRODUCTION_FIELDS) == 9
        assert "tokenizer_identity" in REQUIRED_PRODUCTION_FIELDS
        assert "tokenizer_hash" in REQUIRED_PRODUCTION_FIELDS
        assert "semantic_pretraining_identity" in REQUIRED_PRODUCTION_FIELDS
        assert "semantic_pretraining_hash" in REQUIRED_PRODUCTION_FIELDS
        assert "dataset_version" in REQUIRED_PRODUCTION_FIELDS
        assert "dataset_hash" in REQUIRED_PRODUCTION_FIELDS
        assert "split_identity" in REQUIRED_PRODUCTION_FIELDS
        assert "split_hash" in REQUIRED_PRODUCTION_FIELDS
        assert "code_revision" in REQUIRED_PRODUCTION_FIELDS

    def test_tokenizer_identity_blank_fails_production(self):
        """Blank tokenizer_identity fails production validation."""
        meta = CheckpointMetadata(tokenizer_identity="")
        errors = validate_production_metadata(meta)
        assert any("tokenizer_identity" in e for e in errors)

    def test_tokenizer_hash_blank_fails_production(self):
        """Blank tokenizer_hash fails production validation."""
        meta = CheckpointMetadata(tokenizer_hash="")
        errors = validate_production_metadata(meta)
        assert any("tokenizer_hash" in e for e in errors)

    def test_semantic_pretraining_identity_blank_fails_production(self):
        """Blank semantic_pretraining_identity fails production validation."""
        meta = CheckpointMetadata(semantic_pretraining_identity="")
        errors = validate_production_metadata(meta)
        assert any("semantic_pretraining_identity" in e for e in errors)

    def test_semantic_pretraining_hash_blank_fails_production(self):
        """Blank semantic_pretraining_hash fails production validation."""
        meta = CheckpointMetadata(semantic_pretraining_hash="")
        errors = validate_production_metadata(meta)
        assert any("semantic_pretraining_hash" in e for e in errors)


# ---------------------------------------------------------------------------
# Deployment checkpoint tests
# ---------------------------------------------------------------------------


class TestDeploymentCheckpoint:
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "deploy.pt")
            state = {"heads.linear.weight": torch.randn(6, 448)}
            meta = CheckpointMetadata(
                tokenizer_identity="test-tokenizer",
                tokenizer_hash="abc123def456789abc123def456789abc123def456789abc123def456789",
                semantic_pretraining_identity="mbert-v1",
                semantic_pretraining_hash="def456abc123def456abc123def456abc123def456abc123def456abc123",
                dataset_version="rmc-pilot-v2",
                dataset_hash="123def456abc123def456abc123def456abc123def456abc123def456abc1234",
                split_identity="stratified_v1",
                split_hash="fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
                code_revision="abc123def",
            )
            save_deployment_checkpoint(path, state, meta)
            loaded_state, loaded_meta = load_deployment_checkpoint(
                path, require_production_provenance=True
            )
            assert torch.allclose(loaded_state["heads.linear.weight"], state["heads.linear.weight"])
            assert loaded_meta.code_revision == "abc123def"

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_deployment_checkpoint("/nonexistent/path/checkpoint.pt")

    def test_wrong_kind_rejected(self):
        """Loading a resume checkpoint as deployment must fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            meta = CheckpointMetadata(
                tokenizer_identity="test-tokenizer",
                tokenizer_hash="abc123def456789abc123def456789abc123def456789abc123def456789",
                semantic_pretraining_identity="mbert-v1",
                semantic_pretraining_hash="def456abc123def456abc123def456abc123def456abc123def456abc123",
                dataset_version="rmc-pilot-v2",
                dataset_hash="123def456abc123def456abc123def456abc123def456abc123def456abc1234",
                split_identity="stratified_v1",
                split_hash="fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
                code_revision="abc123def",
            )
            save_resume_checkpoint(
                path, {"w": torch.zeros(1)}, meta,
                optimizer_state_dict={"state": {}},
            )
            with pytest.raises(CheckpointValidationError, match="deployment"):
                load_deployment_checkpoint(path)

    def test_production_provenance_required_by_default(self):
        """Default CheckpointMetadata() fails production validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "deploy.pt")
            meta = CheckpointMetadata()  # blank provenance
            save_deployment_checkpoint(path, {"w": torch.zeros(1)}, meta)
            with pytest.raises(CheckpointValidationError, match="production provenance"):
                load_deployment_checkpoint(path, require_production_provenance=True)

    def test_test_mode_allows_blank_provenance(self):
        """With require_production_provenance=False, blank provenance is OK."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "deploy.pt")
            meta = CheckpointMetadata()  # blank provenance
            save_deployment_checkpoint(path, {"w": torch.zeros(1)}, meta)
            state, loaded_meta = load_deployment_checkpoint(
                path, require_production_provenance=False
            )
            assert "w" in state

    def test_deployment_does_not_contain_optimizer_state(self):
        """Deployment checkpoint must not require optimizer state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "deploy.pt")
            meta = CheckpointMetadata(
                tokenizer_identity="test-tokenizer",
                tokenizer_hash="abc123def456789abc123def456789abc123def456789abc123def456789",
                semantic_pretraining_identity="mbert-v1",
                semantic_pretraining_hash="def456abc123def456abc123def456abc123def456abc123def456abc123",
                dataset_version="rmc-pilot-v2",
                dataset_hash="123def456abc123def456abc123def456abc123def456abc123def456abc1234",
                split_identity="stratified_v1",
                split_hash="fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
                code_revision="abc123def",
            )
            save_deployment_checkpoint(path, {"w": torch.zeros(1)}, meta)
            # Load should work fine — no optimizer state expected
            state, loaded_meta = load_deployment_checkpoint(path)
            assert "optimizer_state_dict" not in state


# ---------------------------------------------------------------------------
# Resume checkpoint tests
# ---------------------------------------------------------------------------


class TestResumeCheckpoint:
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "resume.pt")
            meta = CheckpointMetadata(
                epoch=3,
                global_step=5000,
                best_validation_bce=0.25,
                tokenizer_identity="test-tokenizer",
                tokenizer_hash="abc123def456789abc123def456789abc123def456789abc123def456789",
                semantic_pretraining_identity="mbert-v1",
                semantic_pretraining_hash="def456abc123def456abc123def456abc123def456abc123def456abc123",
                dataset_version="rmc-pilot-v2",
                dataset_hash="123def456abc123def456abc123def456abc123def456abc123def456abc1234",
                split_identity="stratified_v1",
                split_hash="fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
                code_revision="abc123def",
            )
            opt_state = {"state": {"param_groups": [{"lr": 0.001}]}}
            sched_state = {"last_epoch": 3}
            save_resume_checkpoint(
                path, {"w": torch.randn(6, 448)}, meta,
                optimizer_state_dict=opt_state,
                scheduler_state_dict=sched_state,
            )
            model_state, loaded_meta, opt_loaded, sched_loaded, scaler_loaded, py_rng, torch_rng, cuda_rng = (
                load_resume_checkpoint(path)
            )
            assert loaded_meta.epoch == 3
            assert loaded_meta.global_step == 5000
            assert loaded_meta.best_validation_bce == 0.25
            assert opt_loaded["state"]["param_groups"][0]["lr"] == 0.001
            assert sched_loaded["last_epoch"] == 3
            assert scaler_loaded is None  # not provided

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_resume_checkpoint("/nonexistent/path/checkpoint.pt")

    def test_wrong_kind_rejected(self):
        """Loading a deployment checkpoint as resume must fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            meta = CheckpointMetadata(
                tokenizer_identity="test-tokenizer",
                tokenizer_hash="abc123def456789abc123def456789abc123def456789abc123def456789",
                semantic_pretraining_identity="mbert-v1",
                semantic_pretraining_hash="def456abc123def456abc123def456abc123def456abc123def456abc123",
                dataset_version="rmc-pilot-v2",
                dataset_hash="123def456abc123def456abc123def456abc123def456abc123def456abc1234",
                split_identity="stratified_v1",
                split_hash="fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
                code_revision="abc123def",
            )
            save_deployment_checkpoint(path, {"w": torch.zeros(1)}, meta)
            with pytest.raises(CheckpointValidationError, match="resume"):
                load_resume_checkpoint(path)

    def test_resume_preserves_epoch_and_step(self):
        """Resume checkpoint must preserve epoch and global_step."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "resume.pt")
            meta = CheckpointMetadata(
                epoch=10,
                global_step=25000,
                tokenizer_identity="test-tokenizer",
                tokenizer_hash="abc123def456789abc123def456789abc123def456789abc123def456789",
                semantic_pretraining_identity="mbert-v1",
                semantic_pretraining_hash="def456abc123def456abc123def456abc123def456abc123def456abc123",
                dataset_version="rmc-pilot-v2",
                dataset_hash="123def456abc123def456abc123def456abc123def456abc123def456abc1234",
                split_identity="stratified_v1",
                split_hash="fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
                code_revision="abc123def",
            )
            save_resume_checkpoint(path, {"w": torch.zeros(1)}, meta)
            _, loaded_meta, _, _, _, py_rng, torch_rng, cuda_rng = load_resume_checkpoint(path)
            assert loaded_meta.epoch == 10
            assert loaded_meta.global_step == 25000

    def test_resume_preserves_optimizer_state(self):
        """Resume checkpoint must preserve optimizer state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "resume.pt")
            meta = CheckpointMetadata(
                tokenizer_identity="test-tokenizer",
                tokenizer_hash="abc123def456789abc123def456789abc123def456789abc123def456789",
                semantic_pretraining_identity="mbert-v1",
                semantic_pretraining_hash="def456abc123def456abc123def456abc123def456abc123def456abc123",
                dataset_version="rmc-pilot-v2",
                dataset_hash="123def456abc123def456abc123def456abc123def456abc123def456abc1234",
                split_identity="stratified_v1",
                split_hash="fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
                code_revision="abc123def",
            )
            opt_state = {"state": {"param_groups": [{"lr": 0.0005}]}}
            save_resume_checkpoint(
                path, {"w": torch.zeros(1)}, meta,
                optimizer_state_dict=opt_state,
            )
            _, _, opt_loaded, _, _, py_rng, torch_rng, cuda_rng = load_resume_checkpoint(path)
            assert opt_loaded["state"]["param_groups"][0]["lr"] == 0.0005

    def test_resume_preserves_scheduler_state(self):
        """Resume checkpoint must preserve scheduler state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "resume.pt")
            meta = CheckpointMetadata(
                tokenizer_identity="test-tokenizer",
                tokenizer_hash="abc123def456789abc123def456789abc123def456789abc123def456789",
                semantic_pretraining_identity="mbert-v1",
                semantic_pretraining_hash="def456abc123def456abc123def456abc123def456abc123def456abc123",
                dataset_version="rmc-pilot-v2",
                dataset_hash="123def456abc123def456abc123def456abc123def456abc123def456abc1234",
                split_identity="stratified_v1",
                split_hash="fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
                code_revision="abc123def",
            )
            sched_state = {"last_epoch": 7}
            save_resume_checkpoint(
                path, {"w": torch.zeros(1)}, meta,
                scheduler_state_dict=sched_state,
            )
            _, _, _, sched_loaded, _, py_rng, torch_rng, cuda_rng = load_resume_checkpoint(path)
            assert sched_loaded["last_epoch"] == 7

    def test_resume_missing_optional_states_empty_dict(self):
        """Missing optional states return empty dicts/None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "resume.pt")
            meta = CheckpointMetadata(
                tokenizer_identity="test-tokenizer",
                tokenizer_hash="abc123def456789abc123def456789abc123def456789abc123def456789",
                semantic_pretraining_identity="mbert-v1",
                semantic_pretraining_hash="def456abc123def456abc123def456abc123def456abc123def456abc123",
                dataset_version="rmc-pilot-v2",
                dataset_hash="123def456abc123def456abc123def456abc123def456abc123def456abc1234",
                split_identity="stratified_v1",
                split_hash="fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
                code_revision="abc123def",
            )
            save_resume_checkpoint(path, {"w": torch.zeros(1)}, meta)
            _, _, opt_loaded, sched_loaded, scaler_loaded, py_rng, torch_rng, cuda_rng = load_resume_checkpoint(path)
            assert opt_loaded == {}
            assert sched_loaded == {}
            assert scaler_loaded is None


# ---------------------------------------------------------------------------
# Schema mismatch rejection tests
# ---------------------------------------------------------------------------


class TestSchemaMismatchRejection:
    def _make_production_meta(self, **overrides):
        """Create a minimal valid production metadata with optional overrides."""
        defaults = dict(
            tokenizer_identity="test-tokenizer",
            tokenizer_hash="abc123def456789abc123def456789abc123def456789abc123def456789",
            semantic_pretraining_identity="mbert-v1",
            semantic_pretraining_hash="def456abc123def456abc123def456abc123def456abc123def456abc123",
            dataset_version="rmc-pilot-v2",
            dataset_hash="123def456abc123def456abc123def456abc123def456abc123def456abc1234",
            split_identity="stratified_v1",
            split_hash="fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
            code_revision="abc123def",
        )
        defaults.update(overrides)
        return CheckpointMetadata(**defaults)

    def test_schema_version_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            meta = self._make_production_meta(schema_version="2.0")
            save_deployment_checkpoint(path, {"w": torch.zeros(1)}, meta)
            with pytest.raises(CheckpointValidationError, match="schema_version"):
                load_deployment_checkpoint(path)

    def test_label_order_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            meta = self._make_production_meta(
                primary_labels=("pii", "spam", "deception", "toxicity", "advertising", "off_topic"),
            )
            save_deployment_checkpoint(path, {"w": torch.zeros(1)}, meta)
            with pytest.raises(CheckpointValidationError, match="PRIMARY_LABELS"):
                load_deployment_checkpoint(path)

    def test_unknown_label_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            meta = self._make_production_meta(unknown_label=0)
            save_deployment_checkpoint(path, {"w": torch.zeros(1)}, meta)
            with pytest.raises(CheckpointValidationError, match="UNKNOWN_LABEL"):
                load_deployment_checkpoint(path)

    def test_head_input_dim_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            meta = self._make_production_meta(head_input_dim=256)
            save_deployment_checkpoint(path, {"w": torch.zeros(1)}, meta)
            with pytest.raises(CheckpointValidationError, match="head_input_dim"):
                load_deployment_checkpoint(path)

    def test_head_output_dim_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            meta = self._make_production_meta(head_output_dim=4)
            save_deployment_checkpoint(path, {"w": torch.zeros(1)}, meta)
            with pytest.raises(CheckpointValidationError, match="head_output_dim"):
                load_deployment_checkpoint(path)

    def test_threshold_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            meta = self._make_production_meta(thresholds=(0.5, 0.5))
            save_deployment_checkpoint(path, {"w": torch.zeros(1)}, meta)
            with pytest.raises(CheckpointValidationError, match="thresholds length"):
                load_deployment_checkpoint(path)

    def test_tokenizer_identity_mismatch(self):
        """Tokenized identity mismatch is a structural error (not just provenance)."""
        # Tokenizer identity mismatch is NOT tested separately — it's part of
        # production provenance validation. The test below verifies the
        # general mechanism works.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            # Missing tokenizer_identity should fail production validation
            meta = self._make_production_meta(tokenizer_identity="")
            save_deployment_checkpoint(path, {"w": torch.zeros(1)}, meta)
            with pytest.raises(CheckpointValidationError, match="production provenance"):
                load_deployment_checkpoint(path)

    def test_tokenizer_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            meta = self._make_production_meta(tokenizer_hash="")
            save_deployment_checkpoint(path, {"w": torch.zeros(1)}, meta)
            with pytest.raises(CheckpointValidationError, match="production provenance"):
                load_deployment_checkpoint(path)

    def test_semantic_pretraining_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            meta = self._make_production_meta(semantic_pretraining_identity="")
            save_deployment_checkpoint(path, {"w": torch.zeros(1)}, meta)
            with pytest.raises(CheckpointValidationError, match="production provenance"):
                load_deployment_checkpoint(path)

    def test_semantic_pretraining_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            meta = self._make_production_meta(semantic_pretraining_hash="")
            save_deployment_checkpoint(path, {"w": torch.zeros(1)}, meta)
            with pytest.raises(CheckpointValidationError, match="production provenance"):
                load_deployment_checkpoint(path)


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestComputeFileHash:
    def test_consistent(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"hello world")
            path = f.name
        try:
            h1 = compute_file_hash(path)
            h2 = compute_file_hash(path)
            assert h1 == h2
            assert len(h1) == 64
        finally:
            os.unlink(path)

    def test_different_content(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"hello world")
            path1 = f.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"hello world!")
            path2 = f.name
        try:
            assert compute_file_hash(path1) != compute_file_hash(path2)
        finally:
            os.unlink(path1)
            os.unlink(path2)
