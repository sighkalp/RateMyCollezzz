"""Checkpoint management for the RRM (RRM 3.9).

Implements two distinct checkpoint types:

- **Deployment checkpoint**: model state_dict + metadata.  Contains full
  provenance for reproducibility audit.  No optimizer/scheduler state.
- **Training-resume checkpoint**: model + optimizer + scheduler + training
  state.  Supports warm-restart of interrupted training runs.

Research provenance:
    - Checkpoint schema: RRM EXPERIMENTAL DESIGN CHOICE
    - Strict validation on load: STANDARD ENGINEERING PRACTICE
    - Deployment vs resume distinction: STANDARD ENGINEERING PRACTICE

Architecture rule:
    RRM UNDERSTANDS. TRUST DECIDES.

This module manages model state persistence only.
It does NOT make moderation decisions.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from rrm.labels import NUM_PRIMARY_LABELS, PRIMARY_LABELS, UNKNOWN_LABEL


# ---------------------------------------------------------------------------
# Metadata schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointMetadata:
    """Immutable metadata for an RRM checkpoint.

    Fields are divided into:

    - **Always-required structural fields**: schema_version, primary_labels,
      unknown_label, head dimensions.  Validated on every load.
    - **Production provenance fields**: tokenizer identity/hash, semantic
      pretraining identity/hash, dataset version/hash, split identity/hash,
      code_revision.  Required for production deployment checkpoints;
      optional for smoke/test checkpoints.
    - **Training-resume fields**: epoch, global_step, best_validation_bce,
      optimizer_config, scheduler_config.
    - **RNG state fields**: python_rng_state, torch_rng_state,
      cuda_rng_state.  Optional, for exact resumption of RNG sequence.
    - **Informational fields**: seed, loss_normalization_policy,
      checkpoint_selection_criterion, thresholds, threshold_selection_policy,
      metric_definition_version, runtime_versions.
    """

    # Structural — always validated
    schema_version: str = "1.0"
    primary_labels: Tuple[str, ...] = PRIMARY_LABELS
    unknown_label: int = UNKNOWN_LABEL
    semantic_config: Dict[str, Any] = field(default_factory=dict)
    character_config: Dict[str, Any] = field(default_factory=dict)
    head_input_dim: int = 448
    head_output_dim: int = NUM_PRIMARY_LABELS

    # Production provenance — required for production deployment
    tokenizer_identity: str = ""
    tokenizer_hash: str = ""
    semantic_pretraining_identity: str = ""
    semantic_pretraining_hash: str = ""
    dataset_version: str = ""
    dataset_hash: str = ""
    split_identity: str = ""
    split_hash: str = ""
    code_revision: str = ""

    # Training-resume
    epoch: int = 0
    global_step: int = 0
    best_validation_bce: Optional[float] = None
    optimizer_config: Dict[str, Any] = field(default_factory=dict)
    scheduler_config: Dict[str, Any] = field(default_factory=dict)

    # RNG state — optional, for exact resumption
    python_rng_state: Optional[Tuple] = None
    torch_rng_state: Optional[bytes] = None
    cuda_rng_state: Optional[list] = None

    # Informational
    seed: int = 42
    loss_normalization_policy: str = "global_supervised_position"
    checkpoint_selection_criterion: str = "validation_masked_bce"
    thresholds: Optional[Tuple[float, ...]] = None
    threshold_selection_policy: str = "validation_only_frozen_before_test"
    metric_definition_version: str = "1.0"
    runtime_versions: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Production provenance validation
# ---------------------------------------------------------------------------

# Fields that must be non-empty for production deployment validation
REQUIRED_PRODUCTION_FIELDS: Tuple[str, ...] = (
    "tokenizer_identity",
    "tokenizer_hash",
    "semantic_pretraining_identity",
    "semantic_pretraining_hash",
    "dataset_version",
    "dataset_hash",
    "split_identity",
    "split_hash",
    "code_revision",
)


def validate_production_metadata(meta: CheckpointMetadata) -> List[str]:
    """Validate that production provenance fields are populated.

    Returns a list of error messages (empty if valid).

    Parameters
    ----------
    meta : CheckpointMetadata
        Metadata to validate.

    Returns
    -------
    list of str
        Error messages for missing/blank required fields.
    """
    errors: List[str] = []

    # Structural validation first
    if meta.schema_version != "1.0":
        errors.append(
            f"schema_version mismatch: got {meta.schema_version!r}, "
            f"expected '1.0'"
        )
    if tuple(meta.primary_labels) != PRIMARY_LABELS:
        errors.append(
            f"PRIMARY_LABELS order mismatch: got {meta.primary_labels}, "
            f"expected {PRIMARY_LABELS}"
        )
    if meta.unknown_label != UNKNOWN_LABEL:
        errors.append(
            f"UNKNOWN_LABEL mismatch: got {meta.unknown_label}, "
            f"expected {UNKNOWN_LABEL}"
        )
    if meta.head_input_dim != 448:
        errors.append(
            f"head_input_dim mismatch: got {meta.head_input_dim}, "
            f"expected 448"
        )
    if meta.head_output_dim != NUM_PRIMARY_LABELS:
        errors.append(
            f"head_output_dim mismatch: got {meta.head_output_dim}, "
            f"expected {NUM_PRIMARY_LABELS}"
        )

    # Semantic encoder config validation
    if meta.semantic_config:
        if meta.semantic_config.get("encoder_dim") != 768:
            errors.append(
                f"semantic_config encoder_dim mismatch: got "
                f"{meta.semantic_config.get('encoder_dim')}, expected 768"
            )

    # Character branch config validation
    if meta.character_config:
        if meta.character_config.get("char_embed_dim") != 64:
            errors.append(
                f"character_config char_embed_dim mismatch: got "
                f"{meta.character_config.get('char_embed_dim')}, expected 64"
            )

    # Resume-specific validation (applies to all checkpoints — structural)
    if meta.epoch < 0:
        errors.append(
            f"epoch must be >= 0, got {meta.epoch}"
        )
    if meta.global_step < 0:
        errors.append(
            f"global_step must be >= 0, got {meta.global_step}"
        )
    if meta.best_validation_bce is not None:
        if not isinstance(meta.best_validation_bce, (int, float)):
            errors.append(
                f"best_validation_bce must be numeric, got {type(meta.best_validation_bce)}"
            )
        elif not (meta.best_validation_bce == meta.best_validation_bce and  # NaN check
                  meta.best_validation_bce != float("inf") and
                  meta.best_validation_bce != float("-inf")):
            errors.append(
                f"best_validation_bce must be finite, got {meta.best_validation_bce}"
            )

    # Production provenance validation
    for field_name in REQUIRED_PRODUCTION_FIELDS:
        value = getattr(meta, field_name)
        if not value:
            errors.append(
                f"production provenance missing: {field_name} is blank"
            )

    # Thresholds length
    if meta.thresholds is not None:
        if len(meta.thresholds) != NUM_PRIMARY_LABELS:
            errors.append(
                f"thresholds length mismatch: got {len(meta.thresholds)}, "
                f"expected {NUM_PRIMARY_LABELS}"
            )

    return errors


class CheckpointValidationError(Exception):
    """Raised when a checkpoint fails load-time validation."""


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_metadata(meta: CheckpointMetadata) -> Dict[str, Any]:
    """Convert CheckpointMetadata to a JSON-serializable dict."""
    return asdict(meta)


def _deserialize_metadata(data: Dict[str, Any]) -> CheckpointMetadata:
    """Reconstruct CheckpointMetadata from a dict."""
    d = dict(data)
    d["primary_labels"] = tuple(d["primary_labels"])
    if d.get("thresholds") is not None:
        d["thresholds"] = tuple(d["thresholds"])
    return CheckpointMetadata(**d)


def _tensor_to_cpu(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Move all tensors in a state_dict to CPU for serialization."""
    result = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu()
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Deployment checkpoint
# ---------------------------------------------------------------------------


def save_deployment_checkpoint(
    path: str,
    model_state_dict: Dict[str, Any],
    metadata: CheckpointMetadata,
) -> None:
    """Save a deployment checkpoint.

    A deployment checkpoint contains:
    - model state_dict (CPU tensors)
    - metadata (with full production provenance)

    It does NOT contain:
    - optimizer state
    - scheduler state
    - GradScaler state

    Parameters
    ----------
    path : str
        File path for the checkpoint.
    model_state_dict : dict
        Model state_dict.  Optimizer/scheduler state must NOT be included.
    metadata : CheckpointMetadata
        Checkpoint metadata.  Must include production provenance for
        production use (validated by validate_production_metadata).
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    payload = {
        "checkpoint_kind": "deployment",
        "metadata": _serialize_metadata(metadata),
        "model_state_dict": _tensor_to_cpu(model_state_dict),
    }
    torch.save(payload, path)


def load_deployment_checkpoint(
    path: str,
    device: str = "cpu",
    require_production_provenance: bool = True,
) -> Tuple[Dict[str, Any], CheckpointMetadata]:
    """Load a deployment checkpoint.

    Parameters
    ----------
    path : str
        Checkpoint file path.
    device : str
        Device to map tensors to.  Default "cpu".
    require_production_provenance : bool
        If True, validate that production provenance fields are populated.
        Set False for test/smoke validation only.

    Returns
    -------
    (model_state_dict, metadata)

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.
    CheckpointValidationError
        If the checkpoint fails validation or production provenance is
        missing when required.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location=device, weights_only=False)

    kind = payload.get("checkpoint_kind", "")
    if kind != "deployment":
        raise CheckpointValidationError(
            f"Expected deployment checkpoint, got {kind!r}"
        )

    metadata = _deserialize_metadata(payload["metadata"])
    model_state_dict = payload["model_state_dict"]

    errors = validate_production_metadata(metadata)
    if require_production_provenance:
        pass  # errors already include provenance checks
    else:
        # Filter to only structural errors for test mode
        structural_errors = [
            e for e in errors
            if "production provenance" not in e
        ]
        errors = structural_errors

    if errors:
        raise CheckpointValidationError(
            "Checkpoint validation failed:\n" + "\n".join(errors)
        )

    return model_state_dict, metadata


# ---------------------------------------------------------------------------
# Training-resume checkpoint
# ---------------------------------------------------------------------------


def save_resume_checkpoint(
    path: str,
    model_state_dict: Dict[str, Any],
    metadata: CheckpointMetadata,
    optimizer_state_dict: Optional[Dict[str, Any]] = None,
    scheduler_state_dict: Optional[Dict[str, Any]] = None,
    scaler_state_dict: Optional[Dict[str, Any]] = None,
    python_rng_state: Optional[Tuple] = None,
    torch_rng_state: Optional[bytes] = None,
    cuda_rng_state: Optional[list] = None,
) -> None:
    """Save a training-resume checkpoint.

    A resume checkpoint contains everything needed to restart training:
    - model state_dict
    - optimizer state_dict
    - scheduler state_dict
    - GradScaler state_dict (optional)
    - metadata with training progress fields
    - RNG states (optional, for exact resumption)

    Parameters
    ----------
    path : str
        File path for the checkpoint.
    model_state_dict : dict
        Model state_dict.
    metadata : CheckpointMetadata
        Checkpoint metadata including epoch, global_step, etc.
    optimizer_state_dict : dict or None
        Optimizer state dict from ``optimizer.state_dict()``.
    scheduler_state_dict : dict or None
        Scheduler state dict from ``scheduler.state_dict()``.
    scaler_state_dict : dict or None
        GradScaler state dict from ``scaler.state_dict()``.
    python_rng_state : tuple or None
        Python RNG state from ``random.getstate()``.
    torch_rng_state : bytes or None
        PyTorch RNG state from ``torch.get_rng_state()``.
    cuda_rng_state : list or None
        CUDA RNG state from ``torch.cuda.get_rng_state_all()``.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    payload = {
        "checkpoint_kind": "resume",
        "metadata": _serialize_metadata(metadata),
        "model_state_dict": _tensor_to_cpu(model_state_dict),
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = optimizer_state_dict
    if scheduler_state_dict is not None:
        payload["scheduler_state_dict"] = scheduler_state_dict
    if scaler_state_dict is not None:
        payload["scaler_state_dict"] = scaler_state_dict
    if python_rng_state is not None:
        payload["python_rng_state"] = python_rng_state
    if torch_rng_state is not None:
        payload["torch_rng_state"] = torch_rng_state
    if cuda_rng_state is not None:
        payload["cuda_rng_state"] = cuda_rng_state

    torch.save(payload, path)


def load_resume_checkpoint(
    path: str,
    device: str = "cpu",
) -> Tuple[
    Dict[str, Any],
    CheckpointMetadata,
    Dict[str, Any],  # optimizer state
    Dict[str, Any],  # scheduler state
    Optional[Dict[str, Any]],  # scaler state
    Optional[Tuple],  # python_rng_state
    Optional[bytes],  # torch_rng_state
    Optional[list],  # cuda_rng_state
]:
    """Load a training-resume checkpoint.

    Returns all state needed to resume training.  Resume-only fields
    (optimizer, scheduler, scaler, RNG) are always returned — they may be
    empty dicts/None if not present in the checkpoint.

    Parameters
    ----------
    path : str
        Checkpoint file path.
    device : str
        Device to map tensors to.  Default "cpu".

    Returns
    -------
    (model_state_dict, metadata, optimizer_state, scheduler_state, scaler_state,
     python_rng_state, torch_rng_state, cuda_rng_state)

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.
    CheckpointValidationError
        If the checkpoint fails validation.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location=device, weights_only=False)

    kind = payload.get("checkpoint_kind", "")
    if kind != "resume":
        raise CheckpointValidationError(
            f"Expected resume checkpoint, got {kind!r}"
        )

    metadata = _deserialize_metadata(payload["metadata"])
    model_state_dict = payload["model_state_dict"]
    optimizer_state = payload.get("optimizer_state_dict", {})
    scheduler_state = payload.get("scheduler_state_dict", {})
    scaler_state = payload.get("scaler_state_dict", None)
    python_rng = payload.get("python_rng_state", None)
    torch_rng = payload.get("torch_rng_state", None)
    cuda_rng = payload.get("cuda_rng_state", None)

    errors = validate_production_metadata(metadata)
    # Resume checkpoints may have empty provenance (in-progress training)
    structural_errors = [
        e for e in errors
        if "production provenance" not in e
    ]
    if structural_errors:
        raise CheckpointValidationError(
            "Checkpoint validation failed:\n" + "\n".join(structural_errors)
        )

    return (
        model_state_dict, metadata, optimizer_state, scheduler_state,
        scaler_state, python_rng, torch_rng, cuda_rng,
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def compute_file_hash(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
