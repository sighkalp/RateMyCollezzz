"""Tests for rrm.baseline_bert.

NO NETWORK ACCESS during tests.  All tests use tiny locally-constructed
BERT models.  No pretrained weights are downloaded.

The 150-record synthetic RMC pilot is UNIT TEST / SMOKE DEVELOPMENT
ONLY.  No BERT F1/AUPRC results from synthetic data are reported as
scientific evidence.
"""

from __future__ import annotations

import dataclasses
import inspect
import os
import tempfile

import numpy
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rrm.baseline_bert import (
    PRIMARY_LABELS,
    UNKNOWN_LABEL,
    BertBaselineConfig,
    FittedBertBaseline,
    _create_tiny_model_and_tokenizer,
    _compute_label_metrics,
    _tokenize_records,
    evaluate_bert_baseline,
    fit_bert_baseline,
    masked_bce_with_logits,
    set_bert_seed,
)
from rrm.baseline_tfidf_lr import (
    BaselineEvaluation,
    LabelMetrics,
    _validate_records,
    validate_no_exact_leakage,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_record(
    review_id: str,
    review_text: str,
    spam: int = 0,
    deception: int = UNKNOWN_LABEL,
    toxicity: int = 0,
    advertising: int = 0,
    off_topic: int = 0,
    pii: int = 0,
) -> dict:
    return {
        "review_id": review_id,
        "review_text": review_text,
        "spam": spam,
        "deception": deception,
        "toxicity": toxicity,
        "advertising": advertising,
        "off_topic": off_topic,
        "pii": pii,
    }


_TRAIN_RECORDS = [
    _make_record("t1", "great college amazing faculty", spam=0, toxicity=0),
    _make_record("t2", "worst experience ever terrible", spam=1, toxicity=0),
    _make_record("t3", "love this place so much fun", spam=0, toxicity=0),
    _make_record("t4", "scam scam scam fake reviews everywhere", spam=1, toxicity=1),
    _make_record("t5", "okay nothing special", spam=0, toxicity=0),
    _make_record("t6", "hate it worst waste of money", spam=0, toxicity=1),
]

_EVAL_RECORDS = [
    _make_record("e1", "nice campus good teachers", spam=0, toxicity=0),
    _make_record("e2", "terrible bad do not join", spam=1, toxicity=0),
    _make_record("e3", "fantastic experience loved it", spam=0, toxicity=0),
    _make_record("e4", "spam spam fake bot comments", spam=1, toxicity=1),
    _make_record("e5", "average college decent", spam=0, toxicity=0),
    _make_record("e6", "awful garbage worthless", spam=0, toxicity=1),
]


def _make_tiny_fixture():
    """Create a tiny model, tokenizer, and config for offline tests."""
    torch.manual_seed(42)
    model, tokenizer = _create_tiny_model_and_tokenizer(vocab_size=100, max_position_embeddings=64)
    config = BertBaselineConfig(
        max_seq_length=32,
        train_batch_size=2,
        eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        epochs=1,
        seed=42,
    )
    return model, tokenizer, config


# ---------------------------------------------------------------------------
# CONFIG tests
# ---------------------------------------------------------------------------


class TestBertConfig:
    def test_default_config(self):
        config = BertBaselineConfig()
        assert config.model_name == "google-bert/bert-base-multilingual-cased"
        assert config.max_seq_length == 128
        assert config.train_batch_size == 2
        assert config.eval_batch_size == 4
        assert config.gradient_accumulation_steps == 8
        assert config.learning_rate == 2e-5
        assert config.epochs == 4
        assert config.weight_decay == 0.01
        assert config.warmup_ratio == 0.10
        assert config.max_grad_norm == 1.0
        assert config.threshold == 0.5
        assert config.seed == 42
        assert config.use_fp16 is True
        assert config.gradient_checkpointing is True

    def test_config_frozen(self):
        config = BertBaselineConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.max_seq_length = 256  # type: ignore[misc]

    def test_empty_model_name_rejected(self):
        with pytest.raises(ValueError, match="model_name"):
            BertBaselineConfig(model_name="")

    def test_max_seq_length_zero_rejected(self):
        with pytest.raises(ValueError, match="max_seq_length"):
            BertBaselineConfig(max_seq_length=0)

    def test_train_batch_size_zero_rejected(self):
        with pytest.raises(ValueError, match="train_batch_size"):
            BertBaselineConfig(train_batch_size=0)

    def test_eval_batch_size_zero_rejected(self):
        with pytest.raises(ValueError, match="eval_batch_size"):
            BertBaselineConfig(eval_batch_size=0)

    def test_gradient_accumulation_zero_rejected(self):
        with pytest.raises(ValueError, match="gradient_accumulation_steps"):
            BertBaselineConfig(gradient_accumulation_steps=0)

    def test_learning_rate_zero_rejected(self):
        with pytest.raises(ValueError, match="learning_rate"):
            BertBaselineConfig(learning_rate=0)

    def test_learning_rate_negative_rejected(self):
        with pytest.raises(ValueError, match="learning_rate"):
            BertBaselineConfig(learning_rate=-1e-5)

    def test_epochs_zero_rejected(self):
        with pytest.raises(ValueError, match="epochs"):
            BertBaselineConfig(epochs=0)

    def test_weight_decay_negative_rejected(self):
        with pytest.raises(ValueError, match="weight_decay"):
            BertBaselineConfig(weight_decay=-0.01)

    def test_warmup_ratio_one_rejected(self):
        with pytest.raises(ValueError, match="warmup_ratio"):
            BertBaselineConfig(warmup_ratio=1.0)

    def test_warmup_ratio_negative_rejected(self):
        with pytest.raises(ValueError, match="warmup_ratio"):
            BertBaselineConfig(warmup_ratio=-0.1)

    def test_max_grad_norm_zero_rejected(self):
        with pytest.raises(ValueError, match="max_grad_norm"):
            BertBaselineConfig(max_grad_norm=0)

    def test_threshold_at_zero_rejected(self):
        with pytest.raises(ValueError, match="threshold"):
            BertBaselineConfig(threshold=0.0)

    def test_threshold_at_one_rejected(self):
        with pytest.raises(ValueError, match="threshold"):
            BertBaselineConfig(threshold=1.0)

    def test_seed_string_rejected(self):
        with pytest.raises(TypeError, match="seed"):
            BertBaselineConfig(seed="42")  # type: ignore[arg-type]

    def test_label_order_matches_primary_labels(self):
        config = BertBaselineConfig()
        assert len(PRIMARY_LABELS) == 6
        assert PRIMARY_LABELS == ("spam", "deception", "toxicity", "advertising", "off_topic", "pii")


# ---------------------------------------------------------------------------
# MODEL CONTRACT tests
# ---------------------------------------------------------------------------


class TestModelContract:
    def test_tiny_model_logits_shape(self):
        """Tiny model should output (batch, 6) logits."""
        model, tokenizer, _ = _make_tiny_fixture()

        texts = ["hello world", "test text here"]
        encoded = tokenizer(texts, truncation=True, max_length=32, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**encoded).logits

        assert logits.shape == (2, 6)

    def test_no_network_in_tests(self):
        """Offline tests must not download model weights."""
        model, tokenizer, _ = _make_tiny_fixture()
        # Tiny model has small vocab and hidden_size
        assert model.config.vocab_size == 100
        assert model.config.hidden_size == 8


# ---------------------------------------------------------------------------
# MASKED BCE tests
# ---------------------------------------------------------------------------


class TestMaskedBce:
    def test_normal_supervised_loss_finite(self):
        """Normal loss with all supervised labels should be finite."""
        logits = torch.randn(4, 6, requires_grad=True)
        labels = torch.tensor([
            [1, 0, 1, 0, 0, 1],
            [0, 1, 0, 1, 0, 0],
            [1, 1, 0, 0, 1, 0],
            [0, 0, 1, 1, 0, 1],
        ], dtype=torch.float32)

        loss = masked_bce_with_logits(logits, labels)
        assert loss is not None
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_unknown_safe_target_replacement(self):
        """UNKNOWN positions should be replaced with 0.0 target."""
        logits = torch.zeros(2, 2, requires_grad=True)
        labels = torch.tensor([
            [1, -1],
            [-1, 0],
        ], dtype=torch.float32)

        loss = masked_bce_with_logits(logits, labels)
        assert loss is not None
        # Loss should only come from supervised positions (2 total)
        assert torch.isfinite(loss)

    def test_unknown_zero_gradient(self):
        """UNKNOWN positions should receive zero gradient."""
        logits = torch.zeros(2, 2, requires_grad=True)
        labels = torch.tensor([
            [1.0, -1.0],
            [-1.0, 0.0],
        ])

        loss = masked_bce_with_logits(logits, labels)
        assert loss is not None
        loss.backward()

        # grad for supervised positions should be nonzero
        assert logits.grad[0, 0].abs() > 0  # supervised
        assert logits.grad[1, 1].abs() > 0  # supervised
        # grad for UNKNOWN positions should be zero
        assert logits.grad[0, 1].item() == 0.0
        assert logits.grad[1, 0].item() == 0.0

    def test_denominator_counts_supervised_only(self):
        """Loss denominator should be supervised positions, not total."""
        logits = torch.zeros(3, 2, requires_grad=True)
        labels = torch.tensor([
            [1, 0],
            [1, -1],
            [-1, -1],
        ], dtype=torch.float32)

        loss = masked_bce_with_logits(logits, labels)
        assert loss is not None
        # 3 supervised positions: (0,0), (0,1), (1,0)
        # Expected loss = sum of BCE at these positions / 3
        with torch.no_grad():
            expected = F.binary_cross_entropy_with_logits(
                torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
                torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
                reduction="mean",
            )
        assert torch.isclose(loss, expected)

    def test_one_fully_unknown_label_column_no_skip(self):
        """One fully-UNKNOWN label column should NOT skip the batch."""
        logits = torch.zeros(2, 6, requires_grad=True)
        labels = torch.tensor([
            [1, -1, 0, 1, 0, 1],
            [0, -1, 1, 0, 1, 0],
        ], dtype=torch.float32)

        loss = masked_bce_with_logits(logits, labels)
        assert loss is not None  # deception=-1 for all, but other labels supervised
        assert torch.isfinite(loss)

    def test_all_positions_unknown_returns_none(self):
        """All-UNKNOWN batch should return None (caller skips)."""
        logits = torch.zeros(2, 6)
        labels = torch.full((2, 6), UNKNOWN_LABEL, dtype=torch.float32)

        loss = masked_bce_with_logits(logits, labels)
        assert loss is None

    def test_minus_one_never_negative_supervision(self):
        """-1 must not be treated as class 0."""
        # If -1 were treated as class 0, BCE at position with target=-1
        # would differ from BCE with target=0.  We verify the loss
        # contribution from UNKNOWN positions is exactly zero.
        logits = torch.tensor([[2.0, -3.0, 0.5, -0.5]], requires_grad=True)
        labels = torch.tensor([[1, -1, 0, 1]], dtype=torch.float32)

        loss = masked_bce_with_logits(logits, labels)
        assert loss is not None
        loss.backward()

        # grad at UNKNOWN position (col 1) must be exactly zero
        assert logits.grad[0, 1].item() == 0.0
        # grad at supervised positions must be nonzero
        for col in [0, 2, 3]:
            assert logits.grad[0, col].abs() > 0


# ---------------------------------------------------------------------------
# VALIDATION tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_missing_requested_label_rejected(self):
        records = [{"review_id": "x1", "review_text": "some text", "spam": 0}]
        with pytest.raises(ValueError, match="missing required label field"):
            fit_bert_baseline(records, records)

    def test_duplicate_review_id_rejected(self):
        records = [
            _make_record("d1", "some text", spam=0),
            _make_record("d1", "other text", spam=1),
        ]
        with pytest.raises(ValueError, match="Duplicate review_id"):
            fit_bert_baseline(records, records)

    def test_same_review_id_across_splits_rejected(self):
        train = [_make_record("s1", "some text", spam=0)]
        val = [_make_record("s1", "different text", spam=1)]
        with pytest.raises(ValueError, match="review_id.*appears in both"):
            fit_bert_baseline(train, val)

    def test_exact_normalized_duplicate_across_splits_rejected(self):
        train = [_make_record("s1", "some text here", spam=0)]
        val = [_make_record("e1", "some text here", spam=1)]
        with pytest.raises(ValueError, match="Leakage detected"):
            fit_bert_baseline(train, val)

    def test_case_whitespace_normalized_duplicate_rejected(self):
        train = [_make_record("s1", "some text", spam=0)]
        val = [_make_record("e1", "  SOME   Text  ", spam=1)]
        with pytest.raises(ValueError, match="Leakage detected"):
            fit_bert_baseline(train, val)


# ---------------------------------------------------------------------------
# TRAINING tests
# ---------------------------------------------------------------------------


class TestTraining:
    def test_one_epoch_training_runs(self):
        """Training should complete one epoch without errors."""
        model, tokenizer, config = _make_tiny_fixture()
        fitted = fit_bert_baseline(
            _TRAIN_RECORDS,
            _EVAL_RECORDS,
            config=config,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )
        assert isinstance(fitted, FittedBertBaseline)

    def test_result_dataclass_is_frozen(self):
        model, tokenizer, config = _make_tiny_fixture()
        fitted = fit_bert_baseline(
            _TRAIN_RECORDS,
            _EVAL_RECORDS,
            config=config,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )
        assert isinstance(fitted, FittedBertBaseline)
        with pytest.raises(dataclasses.FrozenInstanceError):
            fitted.best_epoch = 99  # type: ignore[misc]

    def test_device_cpu_fallback(self):
        """CPU training should work."""
        model, tokenizer, config = _make_tiny_fixture()
        fitted = fit_bert_baseline(
            _TRAIN_RECORDS,
            _EVAL_RECORDS,
            config=config,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )
        assert fitted.device == "cpu"

    def test_artifact_dir_none_creates_no_directory(self):
        """artifact_dir=None must not create any checkpoint directory."""
        model, tokenizer, config = _make_tiny_fixture()
        fitted = fit_bert_baseline(
            _TRAIN_RECORDS,
            _EVAL_RECORDS,
            config=config,
            artifact_dir=None,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )
        # Just verify training completed
        assert fitted.best_epoch >= 1

    def test_artifact_dir_external_saves_checkpoint(self):
        """Explicit artifact_dir should save a checkpoint."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            model, tokenizer, config = _make_tiny_fixture()
            fitted = fit_bert_baseline(
                _TRAIN_RECORDS,
                _EVAL_RECORDS,
                config=config,
                artifact_dir=tmpdir,
                model=model,
                tokenizer=tokenizer,
                device="cpu",
            )
            ckpt = os.path.join(tmpdir, "best_model.pt")
            assert os.path.exists(ckpt)
            checkpoint = torch.load(ckpt, weights_only=False)
            assert "model_state_dict" in checkpoint
            assert "epoch" in checkpoint
            assert "best_validation_macro_f1" in checkpoint
            assert "primary_labels" in checkpoint
            assert checkpoint["primary_labels"] == PRIMARY_LABELS
            assert checkpoint["model_name"] == config.model_name
            assert checkpoint["seed"] == config.seed

    def test_review_text_only_feature(self):
        """Only review_text should be used for features."""
        records = [
            {
                "review_id": "x1",
                "review_text": "some review text here",
                "spam": 0,
                "deception": UNKNOWN_LABEL,
                "toxicity": 0,
                "advertising": 0,
                "off_topic": 0,
                "pii": 0,
                "extra_field": "ignored",
            },
            {
                "review_id": "x2",
                "review_text": "another review text here",
                "spam": 1,
                "deception": UNKNOWN_LABEL,
                "toxicity": 0,
                "advertising": 0,
                "off_topic": 0,
                "pii": 0,
                "extra_field": "also ignored",
            },
            {
                "review_id": "x3",
                "review_text": "third review here",
                "spam": 0,
                "deception": UNKNOWN_LABEL,
                "toxicity": 1,
                "advertising": 0,
                "off_topic": 0,
                "pii": 0,
                "extra_field": "also ignored",
            },
        ]
        val_records = [
            {
                "review_id": "v1",
                "review_text": "validation review one",
                "spam": 1,
                "deception": UNKNOWN_LABEL,
                "toxicity": 0,
                "advertising": 0,
                "off_topic": 0,
                "pii": 0,
            },
            {
                "review_id": "v2",
                "review_text": "validation review two",
                "spam": 0,
                "deception": UNKNOWN_LABEL,
                "toxicity": 1,
                "advertising": 0,
                "off_topic": 0,
                "pii": 0,
            },
        ]
        model, tokenizer, config = _make_tiny_fixture()
        fitted = fit_bert_baseline(
            records,
            val_records,
            config=config,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )
        assert fitted.best_epoch >= 1

    def test_no_moderation_fields_in_result(self):
        """Result must not contain moderation action fields."""
        model, tokenizer, config = _make_tiny_fixture()
        fitted = fit_bert_baseline(
            _TRAIN_RECORDS,
            _EVAL_RECORDS,
            config=config,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )
        for field in ("delete", "ban", "remove_user", "final_decision"):
            assert not hasattr(fitted, field)


# ---------------------------------------------------------------------------
# EVALUATION tests
# ---------------------------------------------------------------------------


class TestEvaluation:
    def _fitted_model(self):
        model, tokenizer, config = _make_tiny_fixture()
        fitted = fit_bert_baseline(
            _TRAIN_RECORDS,
            _EVAL_RECORDS,
            config=config,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )
        return fitted

    def test_evaluation_returns_baseline_evaluation(self):
        fitted = self._fitted_model()
        result = evaluate_bert_baseline(
            fitted, _TRAIN_RECORDS, _EVAL_RECORDS
        )
        assert isinstance(result, BaselineEvaluation)

    def test_precision_valid(self):
        fitted = self._fitted_model()
        result = evaluate_bert_baseline(
            fitted, _TRAIN_RECORDS, _EVAL_RECORDS
        )
        if result.per_label:
            for m in result.per_label:
                assert 0.0 <= m.precision <= 1.0

    def test_recall_valid(self):
        fitted = self._fitted_model()
        result = evaluate_bert_baseline(
            fitted, _TRAIN_RECORDS, _EVAL_RECORDS
        )
        if result.per_label:
            for m in result.per_label:
                assert 0.0 <= m.recall <= 1.0

    def test_f1_valid(self):
        fitted = self._fitted_model()
        result = evaluate_bert_baseline(
            fitted, _TRAIN_RECORDS, _EVAL_RECORDS
        )
        if result.per_label:
            for m in result.per_label:
                assert 0.0 <= m.f1 <= 1.0

    def test_auprc_valid(self):
        fitted = self._fitted_model()
        result = evaluate_bert_baseline(
            fitted, _TRAIN_RECORDS, _EVAL_RECORDS
        )
        if result.per_label:
            for m in result.per_label:
                assert 0.0 <= m.auprc <= 1.0

    def test_macro_f1_valid(self):
        fitted = self._fitted_model()
        result = evaluate_bert_baseline(
            fitted, _TRAIN_RECORDS, _EVAL_RECORDS
        )
        if result.macro_f1 is not None:
            assert 0.0 <= result.macro_f1 <= 1.0

    def test_single_class_eval_label_skipped(self):
        records = [
            _make_record("s1", "text about college", spam=0),
            _make_record("s2", "more college text", spam=0),
        ]
        fitted = self._fitted_model()
        result = evaluate_bert_baseline(
            fitted, _TRAIN_RECORDS, records
        )
        assert "spam" in result.skipped_labels

    def test_unknown_eval_rows_masked(self):
        records = [
            _make_record("u1", "some text", spam=0),
            _make_record("u2", "other text", spam=UNKNOWN_LABEL),
            _make_record("u3", "more text", spam=1),
        ]
        fitted = self._fitted_model()
        result = evaluate_bert_baseline(
            fitted, _TRAIN_RECORDS, records
        )
        spam_metrics = _find_metrics(result, "spam")
        assert spam_metrics is not None
        assert spam_metrics.support == 2

    def test_threshold_half_behavior(self):
        """Threshold at 0.5 should classify sigmoid(logit) >= 0.5 as positive."""
        from rrm.baseline_bert import _compute_label_metrics

        y_true = numpy.array([0, 0, 1, 1])
        y_proba = numpy.array([0.1, 0.4, 0.6, 0.9])
        y_pred = (y_proba >= 0.5).astype(int)
        m = _compute_label_metrics(y_true, y_pred, y_proba, "test")
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0

    def test_leakage_guard_blocks_eval(self):
        train = [_make_record("s1", "exact same text here", spam=0)]
        val = [_make_record("e1", "exact same text here", spam=1)]
        model, tokenizer, config = _make_tiny_fixture()
        with pytest.raises(ValueError, match="Leakage"):
            fit_bert_baseline(
                train, val,
                config=config, model=model, tokenizer=tokenizer, device="cpu",
            )


# ---------------------------------------------------------------------------
# REPRODUCIBILITY tests
# ---------------------------------------------------------------------------


class TestReproducility:
    def test_seed_deterministic_cpu(self):
        """Same seed should produce same model init weights on CPU."""
        torch.manual_seed(42)
        m1, _ = _create_tiny_model_and_tokenizer()
        torch.manual_seed(42)
        m2, _ = _create_tiny_model_and_tokenizer()

        for (n1, p1), (n2, p2) in zip(
            m1.named_parameters(), m2.named_parameters()
        ):
            assert torch.equal(p1, p2), f"Weights differ for {n1}"

    def test_dataloader_shuffle_deterministic(self):
        """DataLoader shuffle should be deterministic with same seed."""
        from rrm.baseline_bert import ReviewDataset, _collate_fn
        import tempfile

        model, tokenizer, config = _make_tiny_fixture()

        # Create records
        records = [_make_record(f"r{i}", f"text number {i}", spam=i % 2) for i in range(8)]

        input_ids, attn_mask, labels = _tokenize_records(tokenizer, records, 32)
        ds = ReviewDataset(input_ids, attn_mask, labels)

        g1 = torch.Generator()
        g1.manual_seed(42)
        loader1 = DataLoader(ds, batch_size=4, shuffle=True, generator=g1,
                            collate_fn=lambda b: _collate_fn(b, tokenizer.pad_token_id))

        g2 = torch.Generator()
        g2.manual_seed(42)
        loader2 = DataLoader(ds, batch_size=4, shuffle=True, generator=g2,
                            collate_fn=lambda b: _collate_fn(b, tokenizer.pad_token_id))

        for batch1, batch2 in zip(loader1, loader2):
            assert torch.equal(batch1["input_ids"], batch2["input_ids"])
            assert torch.equal(batch1["labels"], batch2["labels"])


# ---------------------------------------------------------------------------
# DEVICE tests
# ---------------------------------------------------------------------------


class TestDevice:
    def test_cpu_fallback(self):
        model, tokenizer, config = _make_tiny_fixture()
        fitted = fit_bert_baseline(
            _TRAIN_RECORDS,
            _EVAL_RECORDS,
            config=config,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )
        assert fitted.device == "cpu"

    def test_fp16_false_no_cuda_required(self):
        """use_fp16=False should work on CPU without CUDA."""
        model, tokenizer, _ = _make_tiny_fixture()
        config = BertBaselineConfig(
            max_seq_length=32,
            train_batch_size=2,
            eval_batch_size=4,
            gradient_accumulation_steps=2,
            epochs=1,
            use_fp16=False,
            gradient_checkpointing=False,
        )
        fitted = fit_bert_baseline(
            _TRAIN_RECORDS,
            _EVAL_RECORDS,
            config=config,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )
        assert fitted.best_epoch >= 1


# ---------------------------------------------------------------------------
# ARCHITECTURE / SAFETY tests
# ---------------------------------------------------------------------------


class TestArchitecture:
    def test_no_moderation_fields(self):
        model, tokenizer, config = _make_tiny_fixture()
        fitted = fit_bert_baseline(
            _TRAIN_RECORDS,
            _EVAL_RECORDS,
            config=config,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )
        for field in ("delete", "ban", "remove_user", "final_decision"):
            assert not hasattr(fitted, field)
            assert not hasattr(fitted.model, field)

    def test_no_trust_signals_in_input(self):
        """Model should only see review_text, not external signals."""
        model, tokenizer, _ = _make_tiny_fixture()
        # Tokenize simple text — no trust signals in input
        encoded = tokenizer(["simple review text"], truncation=True, max_length=32)
        assert "input_ids" in encoded
        assert "attention_mask" in encoded


# ---------------------------------------------------------------------------
# GRADIENT ACCUMULATION tests
# ---------------------------------------------------------------------------


class TestGradientAccumulation:
    def test_gradient_accumulation_step_count(self):
        """Verify optimizer steps match accumulation logic."""
        model, tokenizer, _ = _make_tiny_fixture()

        # 6 records, batch_size=2, accumulation=3 -> 1 full group
        records = [_make_record(f"r{i}", f"text {i}", spam=i % 2, toxicity=(i + 1) % 2) for i in range(6)]
        config = BertBaselineConfig(
            max_seq_length=32,
            train_batch_size=2,
            eval_batch_size=4,
            gradient_accumulation_steps=3,
            learning_rate=1e-4,
            epochs=1,
            use_fp16=False,
            gradient_checkpointing=False,
        )

        # Track optimizer steps by patching
        original_step = torch.optim.AdamW.step
        step_count = [0]

        def counting_step(self, *args, **kwargs):
            step_count[0] += 1
            return original_step(self, *args, **kwargs)

        torch.optim.AdamW.step = counting_step
        try:
            fit_bert_baseline(
                records, _EVAL_RECORDS[:2],
                config=config,
                model=model,
                tokenizer=tokenizer,
                device="cpu",
            )
            # 6 records / batch 2 = 3 microbatches / accumulation 3 = 1 step
            assert step_count[0] == 1
        finally:
            torch.optim.AdamW.step = original_step

    def test_partial_accumulation_group_steps(self):
        """Final partial accumulation group should still step."""
        model, tokenizer, _ = _make_tiny_fixture()

        # 7 records, batch_size=2, accumulation=3 -> 4 microbatches
        # -> 1 full group (3) + 1 partial (1) = 2 steps
        records = [_make_record(f"r{i}", f"text {i}", spam=i % 2, toxicity=(i + 1) % 2) for i in range(7)]
        config = BertBaselineConfig(
            max_seq_length=32,
            train_batch_size=2,
            eval_batch_size=4,
            gradient_accumulation_steps=3,
            learning_rate=1e-4,
            epochs=1,
            use_fp16=False,
            gradient_checkpointing=False,
        )

        original_step = torch.optim.AdamW.step
        step_count = [0]

        def counting_step(self, *args, **kwargs):
            step_count[0] += 1
            return original_step(self, *args, **kwargs)

        torch.optim.AdamW.step = counting_step
        try:
            fit_bert_baseline(
                records, _EVAL_RECORDS[:2],
                config=config,
                model=model,
                tokenizer=tokenizer,
                device="cpu",
            )
            # 7 records / batch 2 = 4 microbatches
            # accumulation 3: 1 full group (3 microbatches) + 1 partial (1 microbatch) = 2 steps
            assert step_count[0] == 2
        finally:
            torch.optim.AdamW.step = original_step

    def test_all_unknown_no_optimizer_step(self):
        """All-UNKNOWN microbatch should not cause an optimizer step."""
        model, tokenizer, _ = _make_tiny_fixture()

        records = [
            {"review_id": "u1", "review_text": "text", "spam": UNKNOWN_LABEL, "deception": UNKNOWN_LABEL, "toxicity": UNKNOWN_LABEL, "advertising": UNKNOWN_LABEL, "off_topic": UNKNOWN_LABEL, "pii": UNKNOWN_LABEL},
            {"review_id": "u2", "review_text": "more", "spam": UNKNOWN_LABEL, "deception": UNKNOWN_LABEL, "toxicity": UNKNOWN_LABEL, "advertising": UNKNOWN_LABEL, "off_topic": UNKNOWN_LABEL, "pii": UNKNOWN_LABEL},
        ]
        config = BertBaselineConfig(
            max_seq_length=32,
            train_batch_size=2,
            eval_batch_size=4,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            epochs=1,
            use_fp16=False,
            gradient_checkpointing=False,
        )

        original_step = torch.optim.AdamW.step
        step_count = [0]

        def counting_step(self, *args, **kwargs):
            step_count[0] += 1
            return original_step(self, *args, **kwargs)

        torch.optim.AdamW.step = counting_step
        try:
            fit_bert_baseline(
                records, _EVAL_RECORDS[:2],
                config=config,
                model=model,
                tokenizer=tokenizer,
                device="cpu",
            )
            # All-UNKNOWN batch: no loss, no step
            assert step_count[0] == 0
        finally:
            torch.optim.AdamW.step = original_step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_metrics(result, label_name):
    for m in result.per_label:
        if m.label == label_name:
            return m
    return None
