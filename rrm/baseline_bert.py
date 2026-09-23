"""Multilingual BERT fine-tuning baseline for the RRM (RRM 3.4C).

This module implements a multilingual BERT baseline for the RRM research
plan.  It loads pretrained weights from ``google-bert/bert-base-multilingual-cased``,
fine-tunes a shared encoder with six binary classification logits, and
uses a custom masked BCE loss that independently excludes UNKNOWN=-1
labels.

Architectural note:
    RRM UNDERSTANDS. TRUST DECIDES.

    This baseline produces risk-probability evidence only.  It makes
    NO moderation decisions.  No Trust-layer action may depend on
    this component.

Research provenance:
    - BERT baseline: REQUIRED BY PROJECT PLAN (IMPLEMENTATION_PLAN.md
      section 3.4)
    - google-bert/bert-base-multilingual-cased: PAPER-DERIVED (Devlin
      et al., multilingual BERT)
    - Shared 6-logit multi-label architecture: RRM EXPERIMENTAL DESIGN
      CHOICE
    - Custom masked BCE with safe targets: STANDARD ENGINEERING PRACTICE
      (PyTorch BCE requires targets in [0, 1])
    - fp16 + gradient checkpointing for 4 GB VRAM: STANDARD ENGINEERING
      PRACTICE
    - All numeric hyperparameters: RRM EXPERIMENTAL DESIGN CHOICES
"""

from __future__ import annotations

import dataclasses
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from rrm.baseline_tfidf_lr import (
    PRIMARY_LABELS,
    UNKNOWN_LABEL,
    BaselineEvaluation,
    LabelMetrics,
    _validate_records,
    validate_no_exact_leakage,
)

# ---------------------------------------------------------------------------
# Private import note:
# _validate_records and validate_no_exact_leakage are imported from
# baseline_tfidf_lr.py for reuse across RRM 3.4 baselines.  They are
# private (_-prefixed) in the source module and documented here as
# internal baseline reuse.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BertBaselineConfig:
    """Immutable configuration for the BERT baseline.

    All numeric values are initial engineering defaults, NOT proven
    optimal hyperparameters.

    Attributes
    ----------
    model_name : str
        Hugging Face model identifier for pretrained weights.
    model_revision : str or None
        Specific model revision (commit hash) for reproducibility.
        None uses the default branch.
    max_seq_length : int
        Maximum token sequence length.
    train_batch_size : int
        Microbatch size during training.
    eval_batch_size : int
        Batch size during evaluation.
    gradient_accumulation_steps : int
        Number of microbatches to accumulate before each optimizer step.
    learning_rate : float
        Peak learning rate for AdamW.
    epochs : int
        Number of training epochs.
    weight_decay : float
        L2 regularization weight decay.
    warmup_ratio : float
        Fraction of total training steps for linear warmup.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    threshold : float
        Decision threshold for binary predictions.
    seed : int
        Random seed for reproducibility.
    use_fp16 : bool
        Enable fp16 autocast on CUDA.
    gradient_checkpointing : bool
        Enable gradient checkpointing to reduce activation memory.
    """

    model_name: str = "google-bert/bert-base-multilingual-cased"
    model_revision: Optional[str] = None

    max_seq_length: int = 128
    train_batch_size: int = 2
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8

    learning_rate: float = 2e-5
    epochs: int = 4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0

    threshold: float = 0.5
    seed: int = 42

    use_fp16: bool = True
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError(
                f"model_name must be a non-empty str, got {self.model_name!r}"
            )
        if self.max_seq_length < 1:
            raise ValueError(
                f"max_seq_length must be >= 1, got {self.max_seq_length}"
            )
        if self.train_batch_size < 1:
            raise ValueError(
                f"train_batch_size must be >= 1, got {self.train_batch_size}"
            )
        if self.eval_batch_size < 1:
            raise ValueError(
                f"eval_batch_size must be >= 1, got {self.eval_batch_size}"
            )
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, "
                f"got {self.gradient_accumulation_steps}"
            )
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be > 0, got {self.learning_rate}"
            )
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be >= 0, got {self.weight_decay}"
            )
        if not (0 <= self.warmup_ratio < 1):
            raise ValueError(
                f"warmup_ratio must be in [0, 1), got {self.warmup_ratio}"
            )
        if self.max_grad_norm <= 0:
            raise ValueError(
                f"max_grad_norm must be > 0, got {self.max_grad_norm}"
            )
        if not (0 < self.threshold < 1):
            raise ValueError(
                f"threshold must be in (0, 1), got {self.threshold}"
            )
        if not isinstance(self.seed, int):
            raise TypeError(
                f"seed must be int, got {type(self.seed).__name__}"
            )


# ---------------------------------------------------------------------------
# Fitted model container
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FittedBertBaseline:
    """Immutable container for a fitted BERT baseline.

    Attributes
    ----------
    model : BertForSequenceClassification
        The fine-tuned model.
    tokenizer : PreTrainedTokenizer
        The tokenizer used for this baseline.
    config : BertBaselineConfig
        Configuration used for training.
    device : str
        Device description the model is on.
    best_epoch : int
        Epoch that achieved the best validation macro-F1.
    best_validation_macro_f1 : float or None
        Best validation macro-F1 achieved.
    model_name : str
        Model identifier used.
    model_revision : str or None
        Resolved model revision if available.
    """

    model: Any  # BertForSequenceClassification
    tokenizer: Any  # PreTrainedTokenizer
    config: BertBaselineConfig
    device: str
    best_epoch: int
    best_validation_macro_f1: Optional[float]
    model_name: str
    model_revision: Optional[str]


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
# This file lives at <repo_root>/rrm/baseline_bert.py.
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
    model : BertForSequenceClassification or None
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
    use_amp: bool,
    scaler: Optional[torch.amp.GradScaler],
) -> None:
    """Normalize accumulated gradients and perform optimizer step.

    Divides all parameter gradients by the total supervised label
    count across the accumulation group, then performs gradient
    clipping and the optimizer step.  This produces correct gradient
    weighting when UNKNOWN density varies across microbatches and
    for partial accumulation groups.

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
    use_amp : bool
        Whether AMP is in use.
    scaler : GradScaler or None
        GradScaler instance if AMP is in use.
    """
    norm = supervised_count
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.div_(norm)

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    if use_amp:
        scaler.step(optimizer)  # type: ignore[union-attr]
        scaler.update()  # type: ignore[union-attr]
    else:
        optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_bert_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Controls Python random, NumPy, torch CPU, and torch CUDA.

    Note:
        Complete bit-for-bit CUDA reproducibility is not guaranteed
        due to nondeterministic GPU operations.  This function sets
        the best-effort deterministic configuration.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model / tokenizer loading
# ---------------------------------------------------------------------------


def _load_production_model(
    config: BertBaselineConfig,
    device: torch.device,
) -> Tuple[BertForSequenceClassification, Any]:
    """Load pretrained model and tokenizer from Hugging Face.

    This is the production loading path.  It requires network access
    on first call to download model weights.

    Parameters
    ----------
    config : BertBaselineConfig
        Configuration with model_name and optional model_revision.
    device : torch.device
        Target device.

    Returns
    -------
    (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        revision=config.model_revision,
    )

    # Build a BertConfig with num_labels=6 and multi_label problem type
    model_config = BertConfig.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        num_labels=len(PRIMARY_LABELS),
        problem_type="multi_label_classification",
    )

    model = BertForSequenceClassification.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        config=model_config,
    )

    model = model.to(device)

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    return model, tokenizer


def _create_tiny_model_and_tokenizer(
    vocab_size: int = 100,
    max_position_embeddings: int = 64,
) -> Tuple[BertForSequenceClassification, Any]:
    """Create a tiny BERT model and tokenizer for offline unit tests.

    Uses randomly initialized weights.  Does NOT download anything.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size for the tiny config.
    max_position_embeddings : int
        Maximum position embeddings.

    Returns
    -------
    (model, tokenizer)
    """
    import tempfile

    label2id = {label: i for i, label in enumerate(PRIMARY_LABELS)}
    id2label = {i: label for label, i in label2id.items()}

    tiny_config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=max_position_embeddings,
        label2id=label2id,
        id2label=id2label,
    )
    model = BertForSequenceClassification(tiny_config)

    # Create a minimal vocab and instantiate a BertTokenizer directly
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    for i in range(vocab_size - len(tokens)):
        tokens.append(f"w{i}")

    tmpdir = tempfile.mkdtemp()
    vocab_file = os.path.join(tmpdir, "vocab.txt")
    with open(vocab_file, "w", encoding="utf-8") as f:
        f.write("\n".join(tokens))

    tokenizer = BertTokenizer(vocab_file)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Dataset
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
    """Collate function with dynamic padding."""
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
    """Tokenize records into input_ids, attention_mask, and label tensors.

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
# Training
# ---------------------------------------------------------------------------


def fit_bert_baseline(
    train_records: Iterable[Mapping[str, Any]],
    validation_records: Iterable[Mapping[str, Any]],
    *,
    config: Optional[BertBaselineConfig] = None,
    artifact_dir: Optional[Path] = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    device: Optional[str] = None,
) -> FittedBertBaseline:
    """Fine-tune a multilingual BERT baseline on training data.

    Parameters
    ----------
    train_records : iterable of mapping-like
        Training records with 'review_id', 'review_text', and label
        fields.
    validation_records : iterable of mapping-like
        Validation records with the same structure.  Used for
        end-of-epoch evaluation and best-checkpoint selection.
    config : BertBaselineConfig or None
        Configuration.  Uses defaults if None.
    artifact_dir : Path or None
        Directory for saving best checkpoints.  If None, no
        checkpoints are saved.  Must be an explicitly provided
        external path — no repository-local fallback.
    model : BertForSequenceClassification or None
        Pre-created model.  If None, loads from ``config.model_name``.
    tokenizer : PreTrainedTokenizer or None
        Pre-created tokenizer.  If None, loads from
        ``config.model_name``.
    device : str or None
        Device string (e.g. "cuda", "cpu").  If None, auto-detects
        CUDA if available.

    Returns
    -------
    FittedBertBaseline
        Fitted baseline with model, tokenizer, and metadata.

    Raises
    ------
    TypeError
        On type violations.
    ValueError
        On value violations or data leakage.
    RuntimeError
        On CUDA OOM during the smoke probe.
    """
    if config is None:
        config = BertBaselineConfig()

    # --- Reproducibility ---
    set_bert_seed(config.seed)

    # --- Device ---
    if device is not None:
        torch_device = torch.device(device)
    elif torch.cuda.is_available():
        torch_device = torch.device("cuda")
    else:
        torch_device = torch.device("cpu")

    # --- Validate injection contract ---
    _validate_model_tokenizer_injection(model, tokenizer)

    # --- Validate records ---
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

    # --- Model / tokenizer ---
    if model is None or tokenizer is None:
        prod_model, prod_tokenizer = _load_production_model(
            config, torch_device
        )
        if model is None:
            model = prod_model
        if tokenizer is None:
            tokenizer = prod_tokenizer
    else:
        model = model.to(torch_device)
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

    # --- Tokenize ---
    train_input_ids, train_attention_mask, train_labels = _tokenize_records(
        tokenizer, train_list, config.max_seq_length
    )
    val_input_ids, val_attention_mask, val_labels = _tokenize_records(
        tokenizer, val_list, config.max_seq_length
    )

    train_dataset = ReviewDataset(
        train_input_ids, train_attention_mask, train_labels
    )
    val_dataset = ReviewDataset(
        val_input_ids, val_attention_mask, val_labels
    )

    # --- DataLoaders ---
    g = torch.Generator()
    g.manual_seed(config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        generator=g,
        num_workers=0,
        collate_fn=lambda b: _collate_fn(b, tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: _collate_fn(b, tokenizer.pad_token_id),
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        foreach=False,
    )

    # --- Scheduler ---
    total_optimizer_steps = config.epochs * math.ceil(
        len(train_dataset) / config.train_batch_size
        / config.gradient_accumulation_steps
    )
    warmup_steps = int(total_optimizer_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    # --- AMP ---
    use_amp = config.use_fp16 and torch_device.type == "cuda"
    scaler = None
    if use_amp:
        scaler = torch.amp.GradScaler(device="cuda")

    # --- Best checkpoint tracking ---
    best_macro_f1: Optional[float] = None
    best_epoch = 0
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    # --- Training loop ---
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulation_counter = 0
        group_supervised_count = 0
        contributing_microbatches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(torch_device)
            attention_mask = batch["attention_mask"].to(torch_device)
            labels_tensor = batch["labels"].to(torch_device)

            # Forward pass
            if use_amp:
                with torch.amp.autocast(
                    device_type="cuda", dtype=torch.float16
                ):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    logits = outputs.logits
                    result = _masked_bce_sum_and_count(logits, labels_tensor)
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

            if accumulation_counter % config.gradient_accumulation_steps == 0:
                # Full accumulation group — step
                if use_amp:
                    scaler.unscale_(optimizer)  # type: ignore[union-attr]
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )
                _normalize_and_step(
                    optimizer, model, group_supervised_count, scheduler,
                    use_amp, scaler,
                )
                group_supervised_count = 0
                contributing_microbatches = 0

        # End of epoch: handle partial accumulation group
        if contributing_microbatches > 0:
            if use_amp:
                scaler.unscale_(optimizer)  # type: ignore[union-attr]
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm
            )
            _normalize_and_step(
                optimizer, model, group_supervised_count, scheduler,
                use_amp, scaler,
            )

    # --- End-of-epoch evaluation ---
    fitted_for_eval = FittedBertBaseline(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=str(torch_device),
        best_epoch=epoch + 1,
        best_validation_macro_f1=None,
        model_name=config.model_name,
        model_revision=config.model_revision,
    )
    val_result = evaluate_bert_baseline(
        fitted_for_eval, train_list, val_list,
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
                    "config": dataclasses.asdict(config),
                    "epoch": best_epoch,
                    "best_validation_macro_f1": best_macro_f1,
                    "primary_labels": PRIMARY_LABELS,
                    "seed": config.seed,
                    "model_name": config.model_name,
                    "model_revision": config.model_revision,
                },
                checkpoint_path,
            )

    # --- Restore best checkpoint into model ---
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return FittedBertBaseline(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=str(torch_device),
        best_epoch=best_epoch,
        best_validation_macro_f1=best_macro_f1,
        model_name=config.model_name,
        model_revision=config.model_revision,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_bert_baseline(
    fitted: FittedBertBaseline,
    train_records: Iterable[Mapping[str, Any]],
    eval_records: Iterable[Mapping[str, Any]],
    *,
    labels: Tuple[str, ...] = PRIMARY_LABELS,
) -> BaselineEvaluation:
    """Evaluate a fitted BERT baseline.

    Parameters
    ----------
    fitted : FittedBertBaseline
        Fitted baseline from :func:`fit_bert_baseline`.
    train_records : iterable of mapping-like
        Training records (used for leakage validation).
    eval_records : iterable of mapping-like
        Evaluation records.
    labels : tuple[str, ...]
        Label names to evaluate.

    Returns
    -------
    BaselineEvaluation
        Evaluation metrics per label, plus macro-F1.
    """
    # Validate
    train_list, _ = _validate_records(
        train_records, "train", labels=labels
    )
    eval_list, _ = _validate_records(
        eval_records, "eval", labels=labels
    )
    validate_no_exact_leakage(train_list, eval_list)

    device = torch.device(fitted.device)
    model = fitted.model
    tokenizer = fitted.tokenizer
    config = fitted.config

    model.eval()

    eval_input_ids, eval_attention_mask, eval_labels = _tokenize_records(
        tokenizer, eval_list, config.max_seq_length
    )
    eval_dataset = ReviewDataset(
        eval_input_ids, eval_attention_mask, eval_labels
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: _collate_fn(b, tokenizer.pad_token_id),
    )

    all_probas: Optional[numpy.ndarray] = None
    all_labels: Optional[numpy.ndarray] = None

    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_tensor = batch["labels"].numpy()  # (batch, 6)

        if fitted.config.use_fp16 and device.type == "cuda":
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                logits = model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
        else:
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

        probas = torch.sigmoid(logits.float()).cpu().numpy()

        all_probas = probas if all_probas is None else numpy.vstack([all_probas, probas])
        all_labels = labels_tensor if all_labels is None else numpy.vstack([all_labels, labels_tensor])

    if all_probas is None or all_labels is None:
        return BaselineEvaluation(
            per_label=(),
            macro_f1=None,
            evaluated_labels=(),
            skipped_labels={},
        )

    per_label: list = []
    evaluated_labels: list = []
    skipped_labels: Dict[str, str] = {}

    for label_name in labels:
        if label_name not in set(PRIMARY_LABELS):
            skipped_labels[label_name] = "label not in PRIMARY_LABELS"
            continue
        # Column index is fixed by PRIMARY_LABELS order
        i = PRIMARY_LABELS.index(label_name)
        y_true_col = all_labels[:, i]
        y_proba_col = all_probas[:, i]

        known_mask = y_true_col != UNKNOWN_LABEL
        y_known = y_true_col[known_mask]
        p_known = y_proba_col[known_mask]

        if len(y_known) == 0:
            skipped_labels[label_name] = "no supervised evaluation data"
            continue

        unique_classes = numpy.unique(y_known)
        if len(unique_classes) < 2:
            skipped_labels[label_name] = (
                "evaluation target has only one class"
            )
            continue

        y_pred = (p_known >= config.threshold).astype(int)

        metrics = _compute_label_metrics(y_known, y_pred, p_known, label_name)
        per_label.append(metrics)
        evaluated_labels.append(label_name)

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
