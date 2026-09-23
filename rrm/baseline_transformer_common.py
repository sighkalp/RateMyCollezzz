"""Shared model-independent machinery for transformer baselines.

This module owns the contracts that BERT and RoBERTa baselines must
both satisfy:

- masked BCE loss with UNKNOWN=-1 exclusion
- artifact path validation
- model/tokenizer injection contract
- evaluable-validation gate
- supervised-position gradient normalization (divide-then-clip)
- seed setting
- dataset / collate / tokenization
- per-label metrics using canonical PRIMARY_LABELS column mapping
- common training loop
- common evaluation loop with AMP support

Research provenance:
    - Masked BCE with safe targets: STANDARD ENGINEERING PRACTICE
      (PyTorch BCE requires targets in [0, 1])
    - Supervised-position gradient normalization: RRM EXPERIMENTAL
      DESIGN CHOICE
    - Gradient normalization BEFORE clipping: STANDARD ENGINEERING
      PRACTICE (correct gradient scaling before norm constraint)
    - fp16 + gradient checkpointing for 4 GB VRAM: STANDARD
      ENGINEERING PRACTICE
    - All numeric hyperparameters: RRM EXPERIMENTAL DESIGN CHOICES

Architectural note:
    RRM UNDERSTANDS. TRUST DECIDES.

    This module produces risk-probability evidence only.  It makes
    NO moderation decisions.  No Trust-layer action may depend on
    this component.
"""

from __future__ import annotations

import dataclasses
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from rrm.baseline_tfidf_lr import (
    PRIMARY_LABELS,
    UNKNOWN_LABEL,
    BaselineEvaluation,
    LabelMetrics,
    _validate_records,
    validate_no_exact_leakage,
)

# ---------------------------------------------------------------------------
# Masked BCE loss
# ---------------------------------------------------------------------------


def masked_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    unknown_label: int = UNKNOWN_LABEL,
) -> Optional[torch.Tensor]:
    """Compute masked BCE loss, excluding UNKNOWN label positions.

    UNKNOWN=-1 positions are replaced with a safe dummy target (0.0)
    before BCE computation, then zeroed out via the supervision mask.
    This avoids passing -1 to ``binary_cross_entropy_with_logits``,
    which requires targets in [0, 1].

    Parameters
    ----------
    logits : torch.Tensor
        Model output logits, shape (batch, num_labels).
    labels : torch.Tensor
        Label tensor, shape (batch, num_labels).  Values must be
        0, 1, or UNKNOWN_LABEL (-1).
    unknown_label : int
        Value representing UNKNOWN.  Default UNKNOWN_LABEL (-1).

    Returns
    -------
    torch.Tensor or None
        Scalar loss tensor if at least one position is supervised,
        or None if all positions are UNKNOWN (caller should skip
        this microbatch).
    """
    supervision_mask = (labels != unknown_label)  # (batch, num_labels)

    safe_targets = torch.where(
        supervision_mask,
        labels,
        torch.zeros_like(labels),
    ).to(dtype=logits.dtype)

    loss_matrix = F.binary_cross_entropy_with_logits(
        logits,
        safe_targets,
        reduction="none",
    )  # (batch, num_labels)

    mask_float = supervision_mask.to(dtype=loss_matrix.dtype)
    supervised_count = mask_float.sum()

    if supervised_count == 0:
        return None

    loss = (loss_matrix * mask_float).sum() / supervised_count
    return loss


def _masked_bce_sum_and_count(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    unknown_label: int = UNKNOWN_LABEL,
) -> Optional[Tuple[torch.Tensor, int]]:
    """Compute masked BCE loss sum and supervised position count.

    Returns (loss_sum, supervised_count) where:

    loss_sum = sum of BCE across all supervised label positions
    supervised_count = exact number of supervised positions

    Returns None if all positions are UNKNOWN (caller skips microbatch).

    The caller normalizes by total supervised count at optimizer-step time,
    NOT per-microbatch.  This produces correct gradient weighting when
    UNKNOWN density varies across microbatches and for partial groups.

    Parameters
    ----------
    logits : torch.Tensor
        Model output logits, shape (batch, num_labels).
    labels : torch.Tensor
        Label tensor, shape (batch, num_labels).
    unknown_label : int
        Value representing UNKNOWN.  Default UNKNOWN_LABEL (-1).

    Returns
    -------
    (loss_sum, supervised_count) or None
    """
    supervision_mask = (labels != unknown_label)  # (batch, num_labels)

    safe_targets = torch.where(
        supervision_mask,
        labels,
        torch.zeros_like(labels),
    ).to(dtype=logits.dtype)

    loss_matrix = F.binary_cross_entropy_with_logits(
        logits,
        safe_targets,
        reduction="none",
    )  # (batch, num_labels)

    mask_float = supervision_mask.to(dtype=loss_matrix.dtype)
    supervised_count = int(mask_float.sum().item())

    if supervised_count == 0:
        return None

    loss_sum = (loss_matrix * mask_float).sum()
    return loss_sum, supervised_count


# ---------------------------------------------------------------------------
# Path / artifact validation helpers
# ---------------------------------------------------------------------------

# Resolve the repository root from this module's location.
# This file lives at <repo_root>/rrm/baseline_transformer_common.py.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent


def _validate_artifact_dir(artifact_dir: Optional[Path]) -> Optional[Path]:
    """Validate that artifact_dir is outside the repository.

    Parameters
    ----------
    artifact_dir : Path or None
        Proposed artifact directory.

    Returns
    -------
    Path or None
        Resolved path if valid, None if artifact_dir is None.

    Raises
    ------
    ValueError
        If artifact_dir resolves inside the repository.
    """
    if artifact_dir is None:
        return None
    resolved = Path(artifact_dir).resolve()
    try:
        resolved.relative_to(_REPO_ROOT)
    except ValueError:
        pass  # outside repo — OK
    else:
        raise ValueError(
            f"artifact_dir must be outside the repository. "
            f"Got {artifact_dir!s} which resolves to {resolved} "
            f"inside {_REPO_ROOT}"
        )
    return resolved


def _validate_model_tokenizer_injection(
    model: Optional[Any],
    tokenizer: Optional[Any],
) -> None:
    """Enforce both-or-none contract for model/tokenizer injection.

    Parameters
    ----------
    model : torch.nn.Module or None
    tokenizer : PreTrainedTokenizer or None

    Raises
    ------
    ValueError
        If exactly one of model/tokenizer is provided.
    """
    if (model is None) != (tokenizer is None):
        raise ValueError(
            "model and tokenizer must both be provided or both be None. "
            f"Got model={'provided' if model is not None else 'None'} and "
            f"tokenizer={'provided' if tokenizer is not None else 'None'}."
        )


def _validate_requested_labels(
    requested: Tuple[str, ...],
    primary: Tuple[str, ...],
) -> None:
    """Validate that requested evaluation labels are a valid subset.

    Parameters
    ----------
    requested : tuple[str, ...]
        Requested label names for evaluation.
    primary : tuple[str, ...]
        All available primary label names.

    Raises
    ------
    ValueError
        If requested labels are empty, contain duplicates, or include
        names not in primary.
    """
    if not requested:
        raise ValueError(
            "requested labels must be a non-empty subset of "
            f"PRIMARY_LABELS, got empty tuple."
        )
    seen: set = set()
    for name in requested:
        if name in seen:
            raise ValueError(
                f"Duplicate label name in requested labels: {name!r}."
            )
        if name not in primary:
            raise ValueError(
                f"Requested label {name!r} is not in PRIMARY_LABELS "
                f"{primary}."
            )
        seen.add(name)


def _check_evaluable_validation(
    records: list[dict],
    labels: Tuple[str, ...],
) -> None:
    """Check that at least one label in validation has both classes.

    Parameters
    ----------
    records : list of dict
        Validation records.
    labels : tuple[str, ...]
        Primary label names.

    Raises
    ------
    ValueError
        If no label contains both class 0 and class 1.
    """
    for label_name in labels:
        values: set = set()
        for r in records:
            v = r.get(label_name, UNKNOWN_LABEL)
            if v != UNKNOWN_LABEL:
                values.add(v)
        if len(values) >= 2:
            return
    raise ValueError(
        "Validation data must contain at least one label with both "
        "class 0 and class 1 examples after UNKNOWN masking. "
        f"Labels checked: {labels}. "
        "Without this, macro-F1 cannot be computed and checkpoint "
        "selection is impossible."
    )


def _normalize_and_step(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    supervised_count: int,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    max_grad_norm: float,
    use_amp: bool,
    scaler: Optional[torch.amp.GradScaler],
) -> None:
    """Normalize accumulated gradients and perform optimizer step.

    Applies the correct optimizer-update sequence:

    1. (Caller has already called scaler.unscale_ if AMP)
    2. Divide every existing gradient by total supervised positions
       in this accumulation group.
    3. Clip the NORMALIZED gradients using max_grad_norm.
    4. Optimizer / scaler step.
    5. Scheduler step.
    6. Zero gradients.

    There is exactly ONE clipping operation per optimizer update.
    Division happens BEFORE clipping, not after.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer whose gradients to normalize and step.
    model : torch.nn.Module
        The model whose parameters' gradients are normalized.
    supervised_count : int
        Total number of supervised label positions across the
        accumulation group.
    scheduler : torch.optim.lr_scheduler.LambdaLR
        Learning rate scheduler to step.
    max_grad_norm : float
        Maximum gradient norm for clipping (from config, not
        hard-coded).
    use_amp : bool
        Whether AMP is in use.
    scaler : GradScaler or None
        GradScaler instance if AMP is in use.
    """
    # Step 2: Divide gradients by supervised count BEFORE clipping.
    norm = float(supervised_count)
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.div_(norm)

    # Step 3: Clip the NORMALIZED gradients (exactly once).
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # Step 4-6: Optimizer step, scheduler, zero grad.
    # When AMP skips an update (Inf/NaN gradients), scheduler must also
    # be skipped so the learning-rate schedule stays consistent with
    # actual parameter updates.
    if use_amp:
        scale_before = scaler.get_scale()  # type: ignore[union-attr]
        scaler.step(optimizer)  # type: ignore[union-attr]
        scaler.update()  # type: ignore[union-attr]
        scale_after = scaler.get_scale()  # type: ignore[union-attr]
        optimizer_stepped = scale_after >= scale_before
    else:
        optimizer.step()
        optimizer_stepped = True

    if optimizer_stepped:
        scheduler.step()

    optimizer.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_transformer_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Controls Python random, NumPy, torch CPU, and torch CUDA.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset / tokenization
# ---------------------------------------------------------------------------


class ReviewDataset(Dataset):
    """PyTorch Dataset for tokenized review records."""

    def __init__(
        self,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
        labels: list[list[int]],
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(
                self.input_ids[idx], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                self.attention_mask[idx], dtype=torch.long
            ),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def _collate_fn(
    batch: list[Dict[str, torch.Tensor]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """Collate function with dynamic padding.

    Parameters
    ----------
    batch : list of dict
        List of samples from ReviewDataset.
    pad_token_id : int
        Token ID to use for padding.

    Returns
    -------
    dict with stacked input_ids, attention_mask, labels tensors.
    """
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        padded_input_ids = torch.cat([
            item["input_ids"],
            torch.full((pad_len,), pad_token_id, dtype=torch.long),
        ])
        padded_attention_mask = torch.cat([
            item["attention_mask"],
            torch.zeros(pad_len, dtype=torch.long),
        ])

        input_ids_list.append(padded_input_ids)
        attention_mask_list.append(padded_attention_mask)
        labels_list.append(item["labels"])

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
    }


def _tokenize_records(
    tokenizer: Any,
    records: list[dict],
    max_length: int,
) -> Tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Tokenize records into input_ids, attention_mask, and label lists.

    Uses only input_ids and attention_mask from tokenizer output.
    Does not pass token_type_ids to the model (appropriate for
    RoBERTa-family tokenizers).

    Label columns are always in PRIMARY_LABELS canonical order,
    regardless of which labels will later be evaluated.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        Tokenizer instance.
    records : list of dict
        Validated records with 'review_text' and label fields.
    max_length : int
        Maximum sequence length.

    Returns
    -------
    (input_ids_list, attention_mask_list, labels_list)
    """
    texts = [r["review_text"] for r in records]

    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )

    labels = []
    for r in records:
        # Always produce all six columns in canonical PRIMARY_LABELS order.
        # Evaluation uses canonical column mapping to select the right index.
        row = [r[label] for label in PRIMARY_LABELS]
        labels.append(row)

    return encoded["input_ids"], encoded["attention_mask"], labels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _compute_label_metrics(
    y_true: numpy.ndarray,
    y_pred: numpy.ndarray,
    y_proba: numpy.ndarray,
    label_name: str,
) -> LabelMetrics:
    """Compute per-label metrics using scikit-learn.

    Parameters
    ----------
    y_true : numpy.ndarray
        True binary labels (0 or 1), shape (n,).
    y_pred : numpy.ndarray
        Predicted binary labels, shape (n,).
    y_proba : numpy.ndarray
        Predicted probabilities, shape (n,).
    label_name : str
        Label name for the metrics container.

    Returns
    -------
    LabelMetrics
    """
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auprc = float(average_precision_score(y_true, y_proba))

    neg_count = int(numpy.sum(y_true == 0))
    pos_count = int(numpy.sum(y_true == 1))

    return LabelMetrics(
        label=label_name,
        support=len(y_true),
        negative_count=neg_count,
        positive_count=pos_count,
        precision=precision,
        recall=recall,
        f1=f1,
        auprc=auprc,
    )


# sklearn metrics imports (deferred to avoid import overhead if unused)
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Internal training engine
# ---------------------------------------------------------------------------


class _TransformerTrainingResult:
    """Internal result from a training run.

    Not exposed as a public API.  The wrapper fit function converts
    this into its own public fitted dataclass.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: str,
        best_epoch: int,
        best_validation_macro_f1: Optional[float],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.best_epoch = best_epoch
        self.best_validation_macro_f1 = best_validation_macro_f1


def _fit_transformer_baseline(
    model: torch.nn.Module,
    tokenizer: Any,
    train_records: Iterable[Mapping[str, Any]],
    validation_records: Iterable[Mapping[str, Any]],
    *,
    labels: Tuple[str, ...],
    max_seq_length: int,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    epochs: int,
    weight_decay: float,
    warmup_ratio: float,
    max_grad_norm: float,
    threshold: float,
    seed: int,
    use_fp16: bool,
    gradient_checkpointing: bool,
    artifact_dir: Optional[Path],
    device: str,
    model_name: str = "",
    model_revision: Optional[str] = None,
) -> _TransformerTrainingResult:
    """Generic training loop for transformer baselines.

    This function is model-agnostic.  It operates on an already-created
    model and tokenizer and knows nothing about whether the encoder is
    BERT, RoBERTa, or any other Hugging Face transformer.

    Parameters
    ----------
    model : torch.nn.Module
        Pretrained transformer model for sequence classification.
    tokenizer : PreTrainedTokenizer
        Tokenizer for the model.
    train_records : iterable of mapping-like
        Training records.
    validation_records : iterable of mapping-like
        Validation records.
    labels : tuple[str, ...]
        Label field names (must be a subset of PRIMARY_LABELS).
    max_seq_length : int
        Maximum token sequence length.
    train_batch_size : int
        Microbatch size.
    eval_batch_size : int
        Evaluation batch size.
    gradient_accumulation_steps : int
        Microbatches per optimizer step.
    learning_rate : float
        Peak learning rate.
    epochs : int
        Number of training epochs.
    weight_decay : float
        L2 regularization weight.
    warmup_ratio : float
        Warmup fraction of total steps.
    max_grad_norm : float
        Maximum gradient norm for clipping (passed through to
        the step helper; NOT hard-coded).
    threshold : float
        Decision threshold (stored for reference).
    seed : int
        Random seed.
    use_fp16 : bool
        Enable fp16 autocast on CUDA during training and evaluation.
    gradient_checkpointing : bool
        Enable gradient checkpointing to reduce activation memory.
    artifact_dir : Path or None
        Checkpoint directory.
    device : str
        Device string (e.g. "cuda", "cpu").
    model_name : str
        Model identifier for checkpoint metadata.
    model_revision : str or None
        Requested model revision for checkpoint metadata.

    Returns
    -------
    _TransformerTrainingResult
    """
    # --- Reproducibility ---
    set_transformer_seed(seed)

    torch_device = torch.device(device)

    # --- Validate records (must have all PRIMARY_LABELS for tensor construction) ---
    train_list, _ = _validate_records(
        train_records, "train", labels=PRIMARY_LABELS
    )
    val_list, _ = _validate_records(
        validation_records, "validation", labels=PRIMARY_LABELS
    )
    validate_no_exact_leakage(train_list, val_list)

    # --- Early validation gate ---
    _check_evaluable_validation(val_list, PRIMARY_LABELS)

    # --- Artifact path safety ---
    artifact_path = _validate_artifact_dir(artifact_dir)

    # --- Move model to device ---
    model = model.to(torch_device)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # --- Tokenize (always all PRIMARY_LABELS columns) ---
    train_input_ids, train_attention_mask, train_labels = _tokenize_records(
        tokenizer, train_list, max_seq_length
    )
    val_input_ids, val_attention_mask, val_labels = _tokenize_records(
        tokenizer, val_list, max_seq_length
    )

    train_dataset = ReviewDataset(
        train_input_ids, train_attention_mask, train_labels
    )
    val_dataset = ReviewDataset(
        val_input_ids, val_attention_mask, val_labels
    )

    # --- DataLoaders ---
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        generator=g,
        num_workers=0,
        collate_fn=lambda b: _collate_fn(b, tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: _collate_fn(b, tokenizer.pad_token_id),
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        foreach=False,
    )

    # --- Scheduler ---
    total_optimizer_steps = epochs * math.ceil(
        len(train_dataset) / train_batch_size
        / gradient_accumulation_steps
    )
    warmup_steps = int(total_optimizer_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    # --- AMP ---
    use_amp = use_fp16 and torch_device.type == "cuda"
    scaler = None
    if use_amp:
        scaler = torch.amp.GradScaler(device="cuda")

    # --- Best checkpoint tracking ---
    best_macro_f1: Optional[float] = None
    best_epoch = 0
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    # --- Training loop ---
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulation_counter = 0
        group_supervised_count = 0
        contributing_microbatches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(torch_device)
            attention_mask = batch["attention_mask"].to(torch_device)
            labels_tensor = batch["labels"].to(torch_device)

            # Forward pass with AMP if enabled
            if use_amp:
                with torch.amp.autocast(
                    device_type="cuda", dtype=torch.float16
                ):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    logits = outputs.logits
                    result = _masked_bce_sum_and_count(
                        logits, labels_tensor
                    )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
                result = _masked_bce_sum_and_count(logits, labels_tensor)

            if result is None:
                # All-UNKNOWN microbatch — skip entirely
                continue

            loss_sum, supervised_count = result
            group_supervised_count += supervised_count
            contributing_microbatches += 1

            # Accumulate loss sum (not averaged per-microbatch)
            if use_amp:
                scaler.scale(loss_sum).backward()  # type: ignore[union-attr]
            else:
                loss_sum.backward()

            accumulation_counter += 1

            if accumulation_counter % gradient_accumulation_steps == 0:
                # Full accumulation group — step
                # CORRECT ORDER: unscale → divide → clip → step
                if use_amp:
                    scaler.unscale_(optimizer)  # type: ignore[union-attr]
                _normalize_and_step(
                    optimizer, model, group_supervised_count, scheduler,
                    max_grad_norm, use_amp, scaler,
                )
                group_supervised_count = 0
                contributing_microbatches = 0

        # End of epoch: handle partial accumulation group
        # CORRECT ORDER: unscale → divide → clip → step
        if contributing_microbatches > 0:
            if use_amp:
                scaler.unscale_(optimizer)  # type: ignore[union-attr]
            _normalize_and_step(
                optimizer, model, group_supervised_count, scheduler,
                max_grad_norm, use_amp, scaler,
            )

        # --- End-of-epoch evaluation ---
        val_result = _evaluate_transformer_baseline(
            model, tokenizer, None, device,
            train_list, val_list,
            requested_labels=labels,
            max_seq_length=max_seq_length,
            eval_batch_size=eval_batch_size,
            threshold=threshold,
            use_fp16=use_fp16,
        )

        current_macro_f1 = val_result.macro_f1
        if current_macro_f1 is not None and (
            best_macro_f1 is None or current_macro_f1 > best_macro_f1
        ):
            best_macro_f1 = current_macro_f1
            best_epoch = epoch + 1  # 1-indexed
            # Capture CPU copy of best state dict
            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

            # Save best checkpoint if artifact_dir provided
            if artifact_path is not None:
                artifact_path.mkdir(parents=True, exist_ok=True)
                checkpoint_path = artifact_path / "best_model.pt"
                torch.save(
                    {
                        "model_state_dict": best_state_dict,
                        "epoch": best_epoch,
                        "best_validation_macro_f1": best_macro_f1,
                        "primary_labels": PRIMARY_LABELS,
                        "seed": seed,
                        "model_name": model_name,
                        "model_revision": model_revision,
                    },
                    checkpoint_path,
                )

    # --- Restore best checkpoint into model ---
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return _TransformerTrainingResult(
        model=model,
        tokenizer=tokenizer,
        device=device,
        best_epoch=best_epoch,
        best_validation_macro_f1=best_macro_f1,
    )


# ---------------------------------------------------------------------------
# Internal evaluation engine
# ---------------------------------------------------------------------------


def _evaluate_transformer_baseline(
    model: torch.nn.Module,
    tokenizer: Any,
    config: Any,
    device: str,
    train_records: Iterable[Mapping[str, Any]],
    eval_records: Iterable[Mapping[str, Any]],
    *,
    requested_labels: Tuple[str, ...] = PRIMARY_LABELS,
    max_seq_length: int = 128,
    eval_batch_size: int = 4,
    threshold: float = 0.5,
    use_fp16: bool = False,
) -> BaselineEvaluation:
    """Evaluate a transformer baseline model.

    This function is model-agnostic.  It operates on an already-fitted
    model and tokenizer.

    Model outputs always have six logits (PRIMARY_LABELS order).
    Requested evaluation labels are mapped to the correct canonical
    column via PRIMARY_LABELS.index(label_name).

    Parameters
    ----------
    model : torch.nn.Module
        Fitted transformer model.
    tokenizer : PreTrainedTokenizer
        Tokenizer used for this model.
    config : object or None
        Config object (used for max_seq_length if provided).
    device : str
        Device string.
    train_records : iterable of mapping-like
        Training records (used for leakage validation).
    eval_records : iterable of mapping-like
        Evaluation records.
    requested_labels : tuple[str, ...]
        Label names to evaluate.  Must be a non-empty subset of
        PRIMARY_LABELS.  Duplicates rejected.  Column mapping uses
        canonical PRIMARY_LABELS.index(), NOT enumerate position.
    max_seq_length : int
        Maximum sequence length.
    eval_batch_size : int
        Batch size for evaluation.
    threshold : float
        Decision threshold for binary predictions.
    use_fp16 : bool
        Enable fp16 autocast during forward pass when on CUDA.

    Returns
    -------
    BaselineEvaluation
    """
    # --- Validate requested labels ---
    _validate_requested_labels(requested_labels, PRIMARY_LABELS)

    # Materialize iterables
    train_list = list(train_records)
    eval_list = list(eval_records)

    # Validate records against ALL PRIMARY_LABELS (model needs all six columns)
    train_validated, _ = _validate_records(
        train_list, "train", labels=PRIMARY_LABELS
    )
    eval_validated, _ = _validate_records(
        eval_list, "eval", labels=PRIMARY_LABELS
    )
    validate_no_exact_leakage(train_validated, eval_validated)

    # Resolve max_seq_length from config if not explicitly provided
    if config is not None and max_seq_length == 128:
        max_seq_length = getattr(config, "max_seq_length", max_seq_length)

    torch_device = torch.device(device)
    model.eval()

    eval_input_ids, eval_attention_mask, eval_labels = _tokenize_records(
        tokenizer, eval_validated, max_seq_length
    )
    eval_dataset = ReviewDataset(
        eval_input_ids, eval_attention_mask, eval_labels
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: _collate_fn(b, tokenizer.pad_token_id),
    )

    # Pre-compute canonical column indices for each requested label.
    # The model always produces six logits in PRIMARY_LABELS order.
    # Using PRIMARY_LABELS.index() ensures correct column mapping
    # even for non-contiguous or out-of-order requested labels.
    column_map: Dict[str, int] = {
        name: PRIMARY_LABELS.index(name) for name in requested_labels
    }

    all_predictions: Dict[str, List[numpy.ndarray]] = {}
    all_probabilities: Dict[str, List[numpy.ndarray]] = {}
    all_true: Dict[str, List[numpy.ndarray]] = {}

    # Determine AMP usage for evaluation forward pass
    eval_use_amp = use_fp16 and torch_device.type == "cuda"

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(torch_device)
            attention_mask = batch["attention_mask"].to(torch_device)
            labels_batch = batch["labels"].numpy()  # (batch, 6)

            # Forward pass with AMP if enabled
            if eval_use_amp:
                with torch.amp.autocast(
                    device_type="cuda", dtype=torch.float16
                ):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    logits = outputs.logits
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits

            # Cast to float32 for sigmoid (avoid fp16 precision issues)
            probas = torch.sigmoid(logits.float()).cpu().numpy()
            preds = (probas >= threshold).astype(int)

            for label_name in requested_labels:
                col = column_map[label_name]
                if label_name not in all_predictions:
                    all_predictions[label_name] = [preds[:, col]]
                    all_probabilities[label_name] = [probas[:, col]]
                    all_true[label_name] = [labels_batch[:, col]]
                else:
                    all_predictions[label_name].append(preds[:, col])
                    all_probabilities[label_name].append(probas[:, col])
                    all_true[label_name].append(labels_batch[:, col])

    # Concatenate batches
    concat_predictions: Dict[str, numpy.ndarray] = {}
    concat_probabilities: Dict[str, numpy.ndarray] = {}
    concat_true: Dict[str, numpy.ndarray] = {}

    for label_name in requested_labels:
        concat_predictions[label_name] = numpy.concatenate(
            all_predictions[label_name]
        )
        concat_probabilities[label_name] = numpy.concatenate(
            all_probabilities[label_name]
        )
        concat_true[label_name] = numpy.concatenate(
            all_true[label_name]
        )

    # Compute metrics per label
    per_label: list = []
    evaluated_labels: list = []
    skipped_labels: Dict[str, str] = {}

    for label_name in requested_labels:
        y_true = concat_true[label_name]

        # Mask unknown rows
        known_mask = y_true != UNKNOWN_LABEL
        if not numpy.any(known_mask):
            skipped_labels[label_name] = "no supervised evaluation data"
            continue

        y_known = y_true[known_mask]
        y_pred_known = concat_predictions[label_name][known_mask]
        y_proba_known = concat_probabilities[label_name][known_mask]

        # Require both classes
        unique_classes = numpy.unique(y_known)
        if len(unique_classes) < 2:
            skipped_labels[label_name] = (
                "evaluation target has only one class"
            )
            continue

        metrics = _compute_label_metrics(
            y_known, y_pred_known, y_proba_known, label_name
        )
        per_label.append(metrics)
        evaluated_labels.append(label_name)

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
