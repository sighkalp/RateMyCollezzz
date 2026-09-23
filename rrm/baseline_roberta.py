"""RoBERTa fine-tuning baseline for the RRM (RRM 3.4D).

This module wraps the shared transformer-baseline machinery in
``baseline_transformer_common`` with RoBERTa-specific model loading
and configuration.  It loads pretrained weights from
``FacebookAI/roberta-base``, fine-tunes a shared encoder with six
binary classification logits, and uses the same custom masked BCE
loss that excludes UNKNOWN=-1 labels.

Research provenance:
    - RoBERTa baseline: REQUIRED BY PROJECT PLAN (IMPLEMENTATION_PLAN.md
      section 3.4)
    - FacebookAI/roberta-base: PAPER-DERIVED (Liu et al., RoBERTa)
    - Shared 6-logit multi-label architecture: RRM EXPERIMENTAL DESIGN
      CHOICE
    - Custom masked BCE with safe targets: STANDARD ENGINEERING PRACTICE
    - Supervised-position gradient normalization (divide before clip):
      RRM EXPERIMENTAL DESIGN CHOICE
    - fp16 + gradient checkpointing for 4 GB VRAM: STANDARD ENGINEERING
      PRACTICE
    - All numeric hyperparameters: RRM EXPERIMENTAL DESIGN CHOICES

Architectural note:
    RRM UNDERSTANDS. TRUST DECIDES.

    This baseline produces risk-probability evidence only.  It makes
    NO moderation decisions.  No Trust-layer action may depend on
    this component.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from transformers import (
    AutoTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)

from rrm.baseline_transformer_common import (
    PRIMARY_LABELS,
    UNKNOWN_LABEL,
    BaselineEvaluation,
    LabelMetrics,
    ReviewDataset,
    _collate_fn,
    _compute_label_metrics,
    _evaluate_transformer_baseline,
    _fit_transformer_baseline,
    _tokenize_records,
    masked_bce_with_logits,
    set_transformer_seed,
)
from rrm.baseline_tfidf_lr import (
    _validate_records,
    validate_no_exact_leakage,
)

# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------
set_roberta_seed = set_transformer_seed


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RobertaBaselineConfig:
    """Immutable configuration for the RoBERTa baseline.

    All numeric values are initial engineering defaults, NOT proven
    optimal hyperparameters.  These match the BERT baseline for
    controlled comparison.

    Attributes
    ----------
    model_name : str
        Hugging Face model identifier for pretrained weights.
    model_revision : str or None
        Requested model revision (commit hash) for reproducibility.
        None uses the default branch.  This field stores the *requested*
        revision; the resolved commit hash is not captured here.
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

    model_name: str = "FacebookAI/roberta-base"
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
class FittedRobertaBaseline:
    """Immutable container for a fitted RoBERTa baseline.

    Attributes
    ----------
    model : RobertaForSequenceClassification
        The fine-tuned model.
    tokenizer : RobertaTokenizer
        The tokenizer used for this baseline.
    config : RobertaBaselineConfig
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
        Requested model revision (not the resolved commit hash).
    """

    model: Any  # RobertaForSequenceClassification
    tokenizer: Any  # RobertaTokenizer
    config: RobertaBaselineConfig
    device: str
    best_epoch: int
    best_validation_macro_f1: Optional[float]
    model_name: str
    model_revision: Optional[str]


# ---------------------------------------------------------------------------
# Model / tokenizer loading
# ---------------------------------------------------------------------------


def _load_production_model(
    config: RobertaBaselineConfig,
    device: torch.device,
) -> Tuple[RobertaForSequenceClassification, Any]:
    """Load pretrained model and tokenizer from Hugging Face.

    This is the production loading path.  It requires network access
    on first call to download model weights.

    Parameters
    ----------
    config : RobertaBaselineConfig
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

    # Build a RobertaConfig with num_labels=6 and multi_label problem type
    model_config = RobertaConfig.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        num_labels=len(PRIMARY_LABELS),
        problem_type="multi_label_classification",
    )

    model = RobertaForSequenceClassification.from_pretrained(
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
    vocab_size: int = 300,
    max_position_embeddings: int = 64,
) -> Tuple[RobertaForSequenceClassification, Any]:
    """Create a tiny RoBERTa model and tokenizer for offline unit tests.

    Uses randomly initialized weights.  Does NOT download anything.

    The tokenizer is trained with ``tokenizers.ByteLevelBPETokenizer``
    on a small deterministic corpus inside a ``tempfile.TemporaryDirectory``.
    The temporary directory is cleaned up before this function returns.
    The tokenizer caches the vocabulary and remains usable afterward.

    The tokenizer's actual vocabulary size is determined from the
    instantiated tokenizer, and the model config's ``vocab_size`` is
    guaranteed to be at least as large as every tokenizer ID.

    Parameters
    ----------
    vocab_size : int
        Target vocabulary size for BPE training.  Must be >= 256 to
        accommodate byte-level BPE's initial 256 byte tokens.
    max_position_embeddings : int
        Maximum position embeddings.

    Returns
    -------
    (model, tokenizer)
    """
    if vocab_size < 256:
        raise ValueError(
            f"RoBERTa byte-level BPE requires vocab_size >= 256 "
            f"(to hold byte tokens 0-255), got {vocab_size}."
        )

    from tokenizers import ByteLevelBPETokenizer

    label2id = {label: i for i, label in enumerate(PRIMARY_LABELS)}
    id2label = {i: label for label, i in label2id.items()}

    # Representative corpus for tokenizer training (fixture strings only)
    corpus_lines = [
        "Good college and helpful faculty.",
        "GOOD College!!!",
        "placement bahut accha hai",
        "hostel thik hai but mess average",
        "coooool campus",
        "fees 120000 per year",
        "Great placement with high package",
        "bad hostel worst mess food",
        "ACCHA infrastructure hai bahut",
        "campus life is awesome!!",
        "123 456 789 numbers here",
        "sooo goood sooo baaad",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_file = os.path.join(tmpdir, "corpus.txt")
        with open(corpus_file, "w", encoding="utf-8") as f:
            for line in corpus_lines:
                f.write(line + "\n")

        # Train ByteLevelBPETokenizer
        tokenizer_bpe = ByteLevelBPETokenizer()
        tokenizer_bpe.train(
            files=[corpus_file],
            vocab_size=vocab_size,
            min_frequency=1,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        )

        # Save and load tokenizer via from_pretrained (most reliable)
        import glob as _glob

        # save_model creates vocab.json + merges.txt
        tokenizer_bpe.save_model(tmpdir)
        tokenizer = RobertaTokenizer.from_pretrained(tmpdir)

    # Determine actual vocabulary size from the instantiated tokenizer
    actual_vocab_size = len(tokenizer)

    # Build model with vocab_size guaranteed to cover every tokenizer ID
    model_config = RobertaConfig(
        vocab_size=max(vocab_size, actual_vocab_size),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=1,
        label2id=label2id,
        id2label=id2label,
    )
    model = RobertaForSequenceClassification(model_config)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def fit_roberta_baseline(
    train_records: Iterable[Mapping[str, Any]],
    validation_records: Iterable[Mapping[str, Any]],
    *,
    config: Optional[RobertaBaselineConfig] = None,
    artifact_dir: Optional[Path] = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    device: Optional[str] = None,
) -> FittedRobertaBaseline:
    """Fine-tune a RoBERTa baseline on training data.

    Parameters
    ----------
    train_records : iterable of mapping-like
        Training records with 'review_id', 'review_text', and label
        fields.
    validation_records : iterable of mapping-like
        Validation records with the same structure.  Used for
        end-of-epoch evaluation and best-checkpoint selection.
    config : RobertaBaselineConfig or None
        Configuration.  Uses defaults if None.
    artifact_dir : Path or None
        Directory for saving best checkpoints.  If None, no
        checkpoints are saved.  Must be an explicitly provided
        external path — no repository-local fallback.
    model : RobertaForSequenceClassification or None
        Pre-created model.  If None, loads from ``config.model_name``.
    tokenizer : RobertaTokenizer or None
        Pre-created tokenizer.  If None, loads from
        ``config.model_name``.
    device : str or None
        Device string (e.g. "cuda", "cpu").  If None, auto-detects
        CUDA if available.

    Returns
    -------
    FittedRobertaBaseline
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
        config = RobertaBaselineConfig()

    # --- Device ---
    if device is not None:
        torch_device = torch.device(device)
    elif torch.cuda.is_available():
        torch_device = torch.device("cuda")
    else:
        torch_device = torch.device("cpu")

    # --- Validate injection contract ---
    if (model is None) != (tokenizer is None):
        raise ValueError(
            "model and tokenizer must both be provided or both be None. "
            f"Got model={'provided' if model is not None else 'None'} and "
            f"tokenizer={'provided' if tokenizer is not None else 'None'}."
        )

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

    # --- Delegate to common training engine ---
    training_result = _fit_transformer_baseline(
        model=model,
        tokenizer=tokenizer,
        train_records=train_records,
        validation_records=validation_records,
        labels=PRIMARY_LABELS,
        max_seq_length=config.max_seq_length,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        threshold=config.threshold,
        seed=config.seed,
        use_fp16=config.use_fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        artifact_dir=artifact_dir,
        device=str(torch_device),
        model_name=config.model_name,
        model_revision=config.model_revision,
    )

    return FittedRobertaBaseline(
        model=training_result.model,
        tokenizer=training_result.tokenizer,
        config=config,
        device=training_result.device,
        best_epoch=training_result.best_epoch,
        best_validation_macro_f1=training_result.best_validation_macro_f1,
        model_name=config.model_name,
        model_revision=config.model_revision,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_roberta_baseline(
    fitted: FittedRobertaBaseline,
    train_records: Iterable[Mapping[str, Any]],
    eval_records: Iterable[Mapping[str, Any]],
    *,
    labels: Tuple[str, ...] = PRIMARY_LABELS,
) -> BaselineEvaluation:
    """Evaluate a fitted RoBERTa baseline.

    Parameters
    ----------
    fitted : FittedRobertaBaseline
        Fitted baseline from :func:`fit_roberta_baseline`.
    train_records : iterable of mapping-like
        Training records (used for leakage validation).
    eval_records : iterable of mapping-like
        Evaluation records.
    labels : tuple[str, ...]
        Label names to evaluate.  Must be a non-empty subset of
        PRIMARY_LABELS.

    Returns
    -------
    BaselineEvaluation
        Evaluation metrics per label, plus macro-F1.
    """
    return _evaluate_transformer_baseline(
        model=fitted.model,
        tokenizer=fitted.tokenizer,
        config=fitted.config,
        device=fitted.device,
        train_records=train_records,
        eval_records=eval_records,
        requested_labels=labels,
        max_seq_length=fitted.config.max_seq_length,
        eval_batch_size=fitted.config.eval_batch_size,
        threshold=fitted.config.threshold,
        use_fp16=fitted.config.use_fp16,
    )
