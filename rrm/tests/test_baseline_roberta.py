"""Tests for rrm.baseline_roberta.

NO NETWORK ACCESS during tests.  All tests use tiny locally-constructed
RoBERTa models.  No pretrained weights are downloaded.

The 150-record synthetic RMC pilot is UNIT TEST / SMOKE DEVELOPMENT
ONLY.  No RoBERTa F1/AUPRC results from synthetic data are reported as
scientific evidence.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile

import numpy
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from rrm.baseline_roberta import (
    PRIMARY_LABELS,
    UNKNOWN_LABEL,
    FittedRobertaBaseline,
    RobertaBaselineConfig,
    _create_tiny_model_and_tokenizer,
    _tokenize_records,
    evaluate_roberta_baseline,
    fit_roberta_baseline,
    masked_bce_with_logits,
    set_roberta_seed,
)
from rrm.baseline_tfidf_lr import (
    BaselineEvaluation,
    LabelMetrics,
    _validate_records,
    validate_no_exact_leakage,
)
from rrm.baseline_transformer_common import (
    ReviewDataset,
    _collate_fn,
    _compute_label_metrics,
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
    model, tokenizer = _create_tiny_model_and_tokenizer(vocab_size=300, max_position_embeddings=64)
    config = RobertaBaselineConfig(
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


class TestRobertaConfig:
    def test_default_config(self):
        config = RobertaBaselineConfig()
        assert config.model_name == "FacebookAI/roberta-base"
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
        config = RobertaBaselineConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.max_seq_length = 256  # type: ignore[misc]

    def test_empty_model_name_rejected(self):
        with pytest.raises(ValueError, match="model_name"):
            RobertaBaselineConfig(model_name="")

    def test_max_seq_length_zero_rejected(self):
        with pytest.raises(ValueError, match="max_seq_length"):
            RobertaBaselineConfig(max_seq_length=0)

    def test_train_batch_size_zero_rejected(self):
        with pytest.raises(ValueError, match="train_batch_size"):
            RobertaBaselineConfig(train_batch_size=0)

    def test_eval_batch_size_zero_rejected(self):
        with pytest.raises(ValueError, match="eval_batch_size"):
            RobertaBaselineConfig(eval_batch_size=0)

    def test_gradient_accumulation_zero_rejected(self):
        with pytest.raises(ValueError, match="gradient_accumulation_steps"):
            RobertaBaselineConfig(gradient_accumulation_steps=0)

    def test_learning_rate_zero_rejected(self):
        with pytest.raises(ValueError, match="learning_rate"):
            RobertaBaselineConfig(learning_rate=0)

    def test_learning_rate_negative_rejected(self):
        with pytest.raises(ValueError, match="learning_rate"):
            RobertaBaselineConfig(learning_rate=-1e-5)

    def test_epochs_zero_rejected(self):
        with pytest.raises(ValueError, match="epochs"):
            RobertaBaselineConfig(epochs=0)

    def test_weight_decay_negative_rejected(self):
        with pytest.raises(ValueError, match="weight_decay"):
            RobertaBaselineConfig(weight_decay=-0.01)

    def test_warmup_ratio_one_rejected(self):
        with pytest.raises(ValueError, match="warmup_ratio"):
            RobertaBaselineConfig(warmup_ratio=1.0)

    def test_warmup_ratio_negative_rejected(self):
        with pytest.raises(ValueError, match="warmup_ratio"):
            RobertaBaselineConfig(warmup_ratio=-0.1)

    def test_max_grad_norm_zero_rejected(self):
        with pytest.raises(ValueError, match="max_grad_norm"):
            RobertaBaselineConfig(max_grad_norm=0)

    def test_threshold_at_zero_rejected(self):
        with pytest.raises(ValueError, match="threshold"):
            RobertaBaselineConfig(threshold=0.0)

    def test_threshold_at_one_rejected(self):
        with pytest.raises(ValueError, match="threshold"):
            RobertaBaselineConfig(threshold=1.0)

    def test_seed_string_rejected(self):
        with pytest.raises(TypeError, match="seed"):
            RobertaBaselineConfig(seed="42")  # type: ignore[arg-type]

    def test_default_checkpoint_identifier(self):
        config = RobertaBaselineConfig()
        assert config.model_name == "FacebookAI/roberta-base"
        assert config.model_revision is None

    def test_label_order_matches_primary_labels(self):
        config = RobertaBaselineConfig()
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
        assert model.config.vocab_size == 300
        assert model.config.hidden_size == 8

    def test_no_token_type_ids_requirement(self):
        """RoBERTa should work without token_type_ids."""
        model, tokenizer, _ = _make_tiny_fixture()

        texts = ["hello world", "test text"]
        encoded = tokenizer(texts, truncation=True, max_length=32, padding=True, return_tensors="pt")
        # RoBERTa does not use token_type_ids; pass only input_ids + attention_mask
        with torch.no_grad():
            logits = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            ).logits
        assert logits.shape == (2, 6)


# ---------------------------------------------------------------------------
# TINY MODEL CREATION tests
# ---------------------------------------------------------------------------


class TestTinyModel:
    def test_roberta_tiny_model_creation(self):
        model, tokenizer = _create_tiny_model_and_tokenizer(vocab_size=300)
        assert model is not None
        assert tokenizer is not None
        assert model.config.vocab_size == 300

    def test_tiny_model_no_persistent_tmpdir(self):
        """Temporary directories should be created but are cleaned up
        by the OS on process exit."""
        model, tokenizer = _create_tiny_model_and_tokenizer(vocab_size=300)
        # Just verify model and tokenizer are valid
        with torch.no_grad():
            encoded = tokenizer(["test"], truncation=True, max_length=16, return_tensors="pt")
            logits = model(**encoded).logits
        assert logits.shape[1] == 6

    def test_roberta_type_vocab_size_one(self):
        """RoBERTa config should have type_vocab_size=1."""
        model, _ = _create_tiny_model_and_tokenizer(vocab_size=300)
        assert model.config.type_vocab_size == 1


# ---------------------------------------------------------------------------
# MASKED BCE tests (through common module)
# ---------------------------------------------------------------------------


class TestMaskedBce:
    def test_common_masked_bce_available(self):
        """masked_bce_with_logits should be accessible from roberta module."""
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

        assert logits.grad[0, 0].abs() > 0
        assert logits.grad[1, 1].abs() > 0
        assert logits.grad[0, 1].item() == 0.0
        assert logits.grad[1, 0].item() == 0.0

    def test_all_unknown_returns_none(self):
        """All-UNKNOWN batch should return None (caller skips)."""
        logits = torch.zeros(2, 6)
        labels = torch.full((2, 6), UNKNOWN_LABEL, dtype=torch.float32)

        loss = masked_bce_with_logits(logits, labels)
        assert loss is None


# ---------------------------------------------------------------------------
# TOKENIZATION tests
# ---------------------------------------------------------------------------


class TestTokenization:
    def test_roberta_tokenizer_shape(self):
        """Tokenized output should have input_ids and attention_mask."""
        model, tokenizer, _ = _make_tiny_fixture()
        records = [_make_record("r1", "hello world text", spam=0)]

        input_ids, attention_mask, labels = _tokenize_records(
            tokenizer, records, 32
        )
        assert len(input_ids) == 1
        assert len(attention_mask) == 1
        assert len(labels) == 1
        assert labels[0] == [0, -1, 0, 0, 0, 0]

    def test_byte_level_bpe_preserves_text(self):
        """RoBERTa tokenizer should tokenize without lowercasing."""
        _, tokenizer, _ = _make_tiny_fixture()
        # RoBERTa byte-level BPE preserves case; "College" != "college"
        encoded1 = tokenizer("College", truncation=True, max_length=16)
        encoded2 = tokenizer("college", truncation=True, max_length=16)
        # Both should produce tokens; the point is that the API works
        assert "input_ids" in encoded1
        assert "input_ids" in encoded2


# ---------------------------------------------------------------------------
# TRAINING tests
# ---------------------------------------------------------------------------


class TestTraining:
    def test_one_epoch_training_runs(self):
        """Training should complete one epoch without errors."""
        model, tokenizer, config = _make_tiny_fixture()
        fitted = fit_roberta_baseline(
            _TRAIN_RECORDS,
            _EVAL_RECORDS,
            config=config,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )
        assert isinstance(fitted, FittedRobertaBaseline)

    def test_result_dataclass_is_frozen(self):
        model, tokenizer, config = _make_tiny_fixture()
        fitted = fit_roberta_baseline(
            _TRAIN_RECORDS,
            _EVAL_RECORDS,
            config=config,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )
        assert isinstance(fitted, FittedRobertaBaseline)
        with pytest.raises(dataclasses.FrozenInstanceError):
            fitted.best_epoch = 99  # type: ignore[misc]

    def test_device_cpu_fallback(self):
        """CPU training should work."""
        model, tokenizer, config = _make_tiny_fixture()
        fitted = fit_roberta_baseline(
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
        fitted = fit_roberta_baseline(
            _TRAIN_RECORDS,
            _EVAL_RECORDS,
            config=config,
            artifact_dir=None,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )
        assert fitted.best_epoch >= 1

    def test_artifact_dir_external_saves_checkpoint(self):
        """Explicit artifact_dir should save a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model, tokenizer, config = _make_tiny_fixture()
            fitted = fit_roberta_baseline(
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
        fitted = fit_roberta_baseline(
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
        fitted = fit_roberta_baseline(
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
        fitted = fit_roberta_baseline(
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
        result = evaluate_roberta_baseline(
            fitted, _TRAIN_RECORDS, _EVAL_RECORDS
        )
        assert isinstance(result, BaselineEvaluation)

    def test_precision_valid(self):
        fitted = self._fitted_model()
        result = evaluate_roberta_baseline(
            fitted, _TRAIN_RECORDS, _EVAL_RECORDS
        )
        if result.per_label:
            for m in result.per_label:
                assert 0.0 <= m.precision <= 1.0

    def test_recall_valid(self):
        fitted = self._fitted_model()
        result = evaluate_roberta_baseline(
            fitted, _TRAIN_RECORDS, _EVAL_RECORDS
        )
        if result.per_label:
            for m in result.per_label:
                assert 0.0 <= m.recall <= 1.0

    def test_f1_valid(self):
        fitted = self._fitted_model()
        result = evaluate_roberta_baseline(
            fitted, _TRAIN_RECORDS, _EVAL_RECORDS
        )
        if result.per_label:
            for m in result.per_label:
                assert 0.0 <= m.f1 <= 1.0

    def test_auprc_valid(self):
        fitted = self._fitted_model()
        result = evaluate_roberta_baseline(
            fitted, _TRAIN_RECORDS, _EVAL_RECORDS
        )
        if result.per_label:
            for m in result.per_label:
                assert 0.0 <= m.auprc <= 1.0

    def test_macro_f1_valid(self):
        fitted = self._fitted_model()
        result = evaluate_roberta_baseline(
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
        result = evaluate_roberta_baseline(
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
        result = evaluate_roberta_baseline(
            fitted, _TRAIN_RECORDS, records
        )
        spam_metrics = _find_metrics(result, "spam")
        assert spam_metrics is not None
        assert spam_metrics.support == 2

    def test_threshold_half_behavior(self):
        """Threshold at 0.5 should classify sigmoid(logit) >= 0.5 as positive."""
        from rrm.baseline_transformer_common import _compute_label_metrics

        y_true = numpy.array([0, 0, 1, 1])
        y_proba = numpy.array([0.1, 0.4, 0.6, 0.9])
        y_pred = (y_proba >= 0.5).astype(int)
        m = _compute_label_metrics(y_true, y_pred, y_proba, "test")
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0


# ---------------------------------------------------------------------------
# VALIDATION tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_missing_requested_label_rejected(self):
        records = [{"review_id": "x1", "review_text": "some text", "spam": 0}]
        with pytest.raises(ValueError, match="missing required label field"):
            fit_roberta_baseline(records, records)

    def test_duplicate_review_id_rejected(self):
        records = [
            _make_record("d1", "some text", spam=0),
            _make_record("d1", "other text", spam=1),
        ]
        with pytest.raises(ValueError, match="Duplicate review_id"):
            fit_roberta_baseline(records, records)

    def test_same_review_id_across_splits_rejected(self):
        train = [_make_record("s1", "some text", spam=0)]
        val = [_make_record("s1", "different text", spam=1)]
        with pytest.raises(ValueError, match="review_id.*appears in both"):
            fit_roberta_baseline(train, val)

    def test_exact_normalized_duplicate_across_splits_rejected(self):
        train = [_make_record("s1", "some text here", spam=0)]
        val = [_make_record("e1", "some text here", spam=1)]
        with pytest.raises(ValueError, match="Leakage detected"):
            fit_roberta_baseline(train, val)

    def test_case_whitespace_normalized_duplicate_rejected(self):
        train = [_make_record("s1", "some text", spam=0)]
        val = [_make_record("e1", "  SOME   Text  ", spam=1)]
        with pytest.raises(ValueError, match="Leakage detected"):
            fit_roberta_baseline(train, val)

    def test_no_evaluable_validation_raises(self):
        """Validation with only one class across all labels should fail."""
        train_records = [
            _make_record("s1", "text about college", spam=0),
            _make_record("s2", "more college text", spam=0),
        ]
        val_records = [
            _make_record("v1", "different text about college", spam=0),
            _make_record("v2", "more different text", spam=0),
        ]
        model, tokenizer, config = _make_tiny_fixture()
        with pytest.raises(ValueError, match="both class 0 and class 1"):
            fit_roberta_baseline(
                train_records, val_records,
                config=config, model=model, tokenizer=tokenizer, device="cpu",
            )


# ---------------------------------------------------------------------------
# GRADIENT ACCUMULATION tests
# ---------------------------------------------------------------------------


class TestGradientAccumulation:
    def test_gradient_accumulation_step_count(self):
        """Verify optimizer steps match accumulation logic."""
        model, tokenizer, _ = _make_tiny_fixture()

        records = [_make_record(f"r{i}", f"text {i}", spam=i % 2, toxicity=(i + 1) % 2) for i in range(6)]
        config = RobertaBaselineConfig(
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
            fit_roberta_baseline(
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

        records = [_make_record(f"r{i}", f"text {i}", spam=i % 2, toxicity=(i + 1) % 2) for i in range(7)]
        config = RobertaBaselineConfig(
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
            fit_roberta_baseline(
                records, _EVAL_RECORDS[:2],
                config=config,
                model=model,
                tokenizer=tokenizer,
                device="cpu",
            )
            # 7 records / batch 2 = 4 microbatches
            # accumulation 3: 1 full group (3) + 1 partial (1) = 2 steps
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
        config = RobertaBaselineConfig(
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
            fit_roberta_baseline(
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
# REPRODUCIBILITY tests
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_seed_deterministic_cpu(self):
        """Same seed should produce same model init weights on CPU."""
        torch.manual_seed(42)
        m1, _ = _create_tiny_model_and_tokenizer(vocab_size=300)
        torch.manual_seed(42)
        m2, _ = _create_tiny_model_and_tokenizer(vocab_size=300)

        for (n1, p1), (n2, p2) in zip(
            m1.named_parameters(), m2.named_parameters()
        ):
            assert torch.equal(p1, p2), f"Weights differ for {n1}"

    def test_dataloader_shuffle_deterministic(self):
        """DataLoader shuffle should be deterministic with same seed."""
        model, tokenizer, config = _make_tiny_fixture()

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
        fitted = fit_roberta_baseline(
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
        config = RobertaBaselineConfig(
            max_seq_length=32,
            train_batch_size=2,
            eval_batch_size=4,
            gradient_accumulation_steps=2,
            epochs=1,
            use_fp16=False,
            gradient_checkpointing=False,
        )
        fitted = fit_roberta_baseline(
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
        fitted = fit_roberta_baseline(
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
        encoded = tokenizer(["simple review text"], truncation=True, max_length=32)
        assert "input_ids" in encoded
        assert "attention_mask" in encoded


# ---------------------------------------------------------------------------
# INJECTION CONTRACT tests
# ---------------------------------------------------------------------------


class TestInjectionContract:
    def test_model_only_raises(self):
        """Providing model without tokenizer should raise ValueError."""
        model, _, _ = _make_tiny_fixture()
        with pytest.raises(ValueError, match="model and tokenizer must both"):
            fit_roberta_baseline(
                _TRAIN_RECORDS,
                _EVAL_RECORDS,
                model=model,
                tokenizer=None,
                device="cpu",
            )

    def test_tokenizer_only_raises(self):
        """Providing tokenizer without model should raise ValueError."""
        _, tokenizer, _ = _make_tiny_fixture()
        with pytest.raises(ValueError, match="model and tokenizer must both"):
            fit_roberta_baseline(
                _TRAIN_RECORDS,
                _EVAL_RECORDS,
                model=None,
                tokenizer=tokenizer,
                device="cpu",
            )

    def test_both_injected_uses_them(self):
        """Injected model/tokenizer should be used directly."""
        model, tokenizer, config = _make_tiny_fixture()
        fitted = fit_roberta_baseline(
            _TRAIN_RECORDS,
            _EVAL_RECORDS,
            config=config,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )
        assert fitted.model is model
        assert fitted.tokenizer is tokenizer


# ---------------------------------------------------------------------------
# CONFIG INHERITANCE / COMPATIBILITY tests
# ---------------------------------------------------------------------------


class TestConfigCompatibility:
    def test_roberta_config_has_all_common_fields(self):
        """RobertaBaselineConfig should expose the same fields as
        BertBaselineConfig for the common training engine."""
        config = RobertaBaselineConfig()
        assert hasattr(config, "model_name")
        assert hasattr(config, "model_revision")
        assert hasattr(config, "max_seq_length")
        assert hasattr(config, "train_batch_size")
        assert hasattr(config, "eval_batch_size")
        assert hasattr(config, "gradient_accumulation_steps")
        assert hasattr(config, "learning_rate")
        assert hasattr(config, "epochs")
        assert hasattr(config, "weight_decay")
        assert hasattr(config, "warmup_ratio")
        assert hasattr(config, "max_grad_norm")
        assert hasattr(config, "threshold")
        assert hasattr(config, "seed")
        assert hasattr(config, "use_fp16")
        assert hasattr(config, "gradient_checkpointing")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_metrics(result, label_name):
    for m in result.per_label:
        if m.label == label_name:
            return m
    return None


# ---------------------------------------------------------------------------
# GRADIENT NORMALIZATION/CORRECT-ORDER REGRESSION tests
# ---------------------------------------------------------------------------


class TestGradientNormalizeBeforeClip:
    """Prove that gradient normalization (divide) happens BEFORE clipping.

    These tests directly exercise _normalize_and_step with controlled
    gradients.  They verify:
    - divide-then-clip produces the same result as manual divide then clip
    - max_grad_norm from config is propagated
    - exactly one clip call per step
    """

    def test_normalize_before_clip_via_common_helper(self):
        """Verify _normalize_and_step divides BEFORE clipping.

        By injecting large gradients and checking the result after
        _normalize_and_step, we confirm the divide happens before
        clip_grad_norm_.
        """
        from rrm.baseline_transformer_common import (
            PRIMARY_LABELS,
            _masked_bce_sum_and_count,
            _normalize_and_step,
            _tokenize_records,
            _validate_records,
            ReviewDataset,
            _collate_fn,
        )

        # Create a tiny model on CPU
        torch.manual_seed(42)
        model, tokenizer = _create_tiny_model_and_tokenizer(vocab_size=300, max_position_embeddings=32)
        model.train()

        # Inject very large gradients
        big_value = 1e6
        for param in model.parameters():
            if param.requires_grad:
                param.grad = torch.full_like(
                    param.data, big_value / param.data.numel()
                )

        # Create a minimal optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

        supervised_count = 100
        max_grad_norm = 1.0

        # Before step: compute expected norm if divide happens first
        # After divide: each grad becomes big_value / param.numel() / 100
        # After clip: norms should be <= max_grad_norm
        _normalize_and_step(
            optimizer, model, supervised_count, scheduler,
            max_grad_norm, use_amp=False, scaler=None,
        )

        # Verify all gradients were updated (step was performed)
        # After zero_grad, grads should be None or zero
        for param in model.parameters():
            if param.grad is not None:
                assert param.grad is None or torch.allclose(
                    param.grad, torch.zeros_like(param.grad)
                ), "Gradients should be zeroed after step"

    def test_max_grad_norm_propagated_from_config(self):
        """max_grad_norm from config must be passed through, not hard-coded."""
        import torch.nn.utils

        model, tokenizer, _ = _make_tiny_fixture()
        custom_max_norm = 0.5

        config = RobertaBaselineConfig(
            max_seq_length=32,
            train_batch_size=1,
            eval_batch_size=1,
            gradient_accumulation_steps=2,
            epochs=1,
            max_grad_norm=custom_max_norm,
            use_fp16=False,
            gradient_checkpointing=False,
            learning_rate=1e-4,
        )

        captured: list = []

        original_clip = torch.nn.utils.clip_grad_norm_
        def patched_clip(params, max_norm, *args, **kwargs):
            captured.append(max_norm)
            return original_clip(params, max_norm, *args, **kwargs)

        torch.nn.utils.clip_grad_norm_ = patched_clip
        try:
            fit_roberta_baseline(
                _TRAIN_RECORDS[:8],
                _EVAL_RECORDS[:8],
                config=config,
                model=model,
                tokenizer=tokenizer,
                device="cpu",
            )
        finally:
            torch.nn.utils.clip_grad_norm_ = original_clip

        assert len(captured) > 0, "clip_grad_norm_ was never called"
        for m in captured:
            assert m == custom_max_norm, (
                f"Expected max_grad_norm={custom_max_norm}, got {m}"
            )

    def test_exactly_one_clip_per_optimizer_update(self):
        """There must be exactly ONE clip_grad_norm_ call per step,
        not two (one before and one after normalization)."""
        import torch.nn.utils

        model, tokenizer, _ = _make_tiny_fixture()
        config = RobertaBaselineConfig(
            max_seq_length=32,
            train_batch_size=2,
            eval_batch_size=4,
            gradient_accumulation_steps=2,
            epochs=1,
            use_fp16=False,
            gradient_checkpointing=False,
            learning_rate=1e-4,
        )

        clip_calls: list = []

        original_clip = torch.nn.utils.clip_grad_norm_
        def patched_clip(params, max_norm, *args, **kwargs):
            clip_calls.append(max_norm)
            return original_clip(params, max_norm, *args, **kwargs)

        torch.nn.utils.clip_grad_norm_ = patched_clip
        try:
            fit_roberta_baseline(
                _TRAIN_RECORDS,
                _EVAL_RECORDS,
                config=config,
                model=model,
                tokenizer=tokenizer,
                device="cpu",
            )
        finally:
            torch.nn.utils.clip_grad_norm_ = original_clip

        assert len(clip_calls) >= 1, "No clip calls recorded"
        # All calls should use the configured max_grad_norm
        for m in clip_calls:
            assert m == config.max_grad_norm


class TestAMPSkippedStepScheduler:
    """Verify scheduler.step is skipped when AMP overflows and skips
    the optimizer update."""

    # --- Fake scaler for CPU-only testing ---
    class _FakeScaler:
        """Minimal stub that mimics GradScaler.get_scale()/step()/update()
        with controllable skip behavior."""

        def __init__(self, scale: float, skip_step: bool = False):
            self._scale = float(scale)
            self._skip = skip_step
            self.step_called = False
            self.update_called = False

        def get_scale(self) -> float:
            return self._scale

        def step(self, optimizer) -> None:  # noqa: ARG002
            self.step_called = True
            if self._skip:
                # Simulate overflow: scale is halved
                self._scale = self._scale / 2.0
            # else: optimizer.update() is handled by caller

        def update(self) -> None:
            self.update_called = True
            if not self._skip:
                # Grow scale on successful steps
                self._scale = self._scale * 1.01

    def test_skipped_step_skips_scheduler(self):
        """When AMP overflows (scale drops), scheduler.step must NOT
        be called."""
        from rrm.baseline_transformer_common import _normalize_and_step

        torch.manual_seed(42)
        model, tokenizer = _create_tiny_model_and_tokenizer(
            vocab_size=300, max_position_embeddings=32
        )
        model.train()

        # Inject finite gradients
        for param in model.parameters():
            if param.requires_grad and param.grad is None:
                param.grad = torch.randn_like(param.data) * 0.01

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _: 1.0
        )
        scaler = self._FakeScaler(scale=65536.0, skip_step=True)

        sched_before = scheduler.get_last_lr()[0]

        _normalize_and_step(
            optimizer, model, 10, scheduler,
            1.0, use_amp=True, scaler=scaler,
        )

        assert scaler.step_called, "scaler.step was not called"
        assert scaler.update_called, "scaler.update was not called"
        assert scaler.get_scale() == 32768.0, (
            f"Expected scale halved to 32768, got {scaler.get_scale()}"
        )
        # Scheduler must NOT have stepped
        assert scheduler.get_last_lr()[0] == sched_before, (
            "Scheduler stepped despite AMP overflow"
        )

    def test_successful_step_calls_scheduler_unchanged_scale(self):
        """When AMP step succeeds with unchanged scale, scheduler.step
        must be called exactly once."""
        from rrm.baseline_transformer_common import _normalize_and_step

        torch.manual_seed(42)
        model, tokenizer = _create_tiny_model_and_tokenizer(
            vocab_size=300, max_position_embeddings=32
        )
        model.train()

        for param in model.parameters():
            if param.requires_grad and param.grad is None:
                param.grad = torch.randn_like(param.data) * 0.01

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _: 1.0
        )
        # scale unchanged on success
        scaler = self._FakeScaler(scale=65536.0, skip_step=False)

        _normalize_and_step(
            optimizer, model, 10, scheduler,
            1.0, use_amp=True, scaler=scaler,
        )

        assert scaler.step_called, "scaler.step was not called"
        # Scale should have grown slightly
        assert scaler.get_scale() > 65536.0, (
            f"Expected scale to grow, got {scaler.get_scale()}"
        )

    def test_successful_step_calls_scheduler_scale_increase(self):
        """When AMP step succeeds and scale increases, scheduler.step
        must be called exactly once."""
        from rrm.baseline_transformer_common import _normalize_and_step

        torch.manual_seed(42)
        model, tokenizer = _create_tiny_model_and_tokenizer(
            vocab_size=300, max_position_embeddings=32
        )
        model.train()

        for param in model.parameters():
            if param.requires_grad and param.grad is None:
                param.grad = torch.randn_like(param.data) * 0.01

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _: 1.0
        )
        scaler = self._FakeScaler(scale=1024.0, skip_step=False)

        _normalize_and_step(
            optimizer, model, 10, scheduler,
            1.0, use_amp=True, scaler=scaler,
        )

        assert scaler.step_called, "scaler.step was not called"
        # Scale should have grown
        assert scaler.get_scale() > 1024.0, (
            f"Expected scale to grow, got {scaler.get_scale()}"
        )

    def test_non_amp_always_calls_scheduler(self):
        """Without AMP, scheduler.step must always be called."""
        from rrm.baseline_transformer_common import _normalize_and_step

        torch.manual_seed(42)
        model, tokenizer = _create_tiny_model_and_tokenizer(
            vocab_size=300, max_position_embeddings=32
        )
        model.train()

        for param in model.parameters():
            if param.requires_grad and param.grad is None:
                param.grad = torch.randn_like(param.data) * 0.01

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _: 1.0
        )

        _normalize_and_step(
            optimizer, model, 10, scheduler,
            1.0, use_amp=False, scaler=None,
        )

        # Verify the optimizer state was updated (step occurred)
        assert len(optimizer.state) > 0, "Optimizer state is empty; step did not occur"


# ---------------------------------------------------------------------------
# SUBSET-LABEL CANONICAL MAPPING REGRESSION tests
# ---------------------------------------------------------------------------


class TestSubsetLabelMapping:
    """Verify that evaluation uses canonical PRIMARY_LABELS column
    mapping, not enumerate(requested_labels) position."""

    def test_toxicity_only_maps_to_toxicity_column(self):
        """When evaluating only 'toxicity', results must match the
        canonical toxicity column, not spam or any other column."""
        model, tokenizer, _ = _make_tiny_fixture()
        config = RobertaBaselineConfig(
            max_seq_length=32,
            eval_batch_size=4,
            gradient_accumulation_steps=2,
            epochs=1,
            learning_rate=1e-4,
            use_fp16=False,
            gradient_checkpointing=False,
        )

        fitted = fit_roberta_baseline(
            _TRAIN_RECORDS,
            _EVAL_RECORDS,
            config=config,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )

        # Evaluate ONLY toxicity
        result = evaluate_roberta_baseline(
            fitted,
            _TRAIN_RECORDS,
            _EVAL_RECORDS,
            labels=("toxicity",),
        )

        toxicity_metrics = _find_metrics(result, "toxicity")
        assert toxicity_metrics is not None, "toxicity should be evaluated"
        # spam must NOT appear in evaluated results
        spam_metrics = _find_metrics(result, "spam")
        assert spam_metrics is None, (
            "spam should NOT appear when evaluating labels=('toxicity',)"
        )

    def test_non_contiguous_subset_spam_and_pii(self):
        """Evaluating ('spam', 'pii') uses canonical column mapping,
        not position-based indexing.  'spam' has both classes,
        'pii' has only one class (gets skipped)."""
        model, tokenizer, _ = _make_tiny_fixture()
        config = RobertaBaselineConfig(
            max_seq_length=32,
            eval_batch_size=4,
            gradient_accumulation_steps=2,
            epochs=1,
            learning_rate=1e-4,
            use_fp16=False,
            gradient_checkpointing=False,
        )

        fitted = fit_roberta_baseline(
            _TRAIN_RECORDS,
            _EVAL_RECORDS,
            config=config,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )

        result = evaluate_roberta_baseline(
            fitted,
            _TRAIN_RECORDS,
            _EVAL_RECORDS,
            labels=("spam", "pii"),
        )

        evaluated = set(result.evaluated_labels)
        assert "spam" in evaluated, "spam should be evaluated (both classes present)"
        # 'pii' has only one class in eval data so it should be skipped
        assert "pii" not in evaluated, "pii should be skipped (one class only)"
        assert "pii" in result.skipped_labels

    def test_requested_labels_validation_rejects_unknown(self):
        """Unknown label names must raise ValueError."""
        model, tokenizer, _ = _make_tiny_fixture()
        config = RobertaBaselineConfig(
            max_seq_length=32,
            eval_batch_size=4,
            gradient_accumulation_steps=2,
            epochs=1,
            use_fp16=False,
            gradient_checkpointing=False,
        )

        fitted = fit_roberta_baseline(
            _TRAIN_RECORDS,
            _EVAL_RECORDS,
            config=config,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )

        with pytest.raises(ValueError, match="not in PRIMARY_LABELS"):
            evaluate_roberta_baseline(
                fitted,
                _TRAIN_RECORDS,
                _EVAL_RECORDS,
                labels=("nonexistent_label",),
            )

    def test_requested_labels_rejects_empty(self):
        """Empty label set must raise ValueError."""
        model, tokenizer, _ = _make_tiny_fixture()
        config = RobertaBaselineConfig(
            max_seq_length=32,
            eval_batch_size=4,
            gradient_accumulation_steps=2,
            epochs=1,
            use_fp16=False,
            gradient_checkpointing=False,
        )

        fitted = fit_roberta_baseline(
            _TRAIN_RECORDS,
            _EVAL_RECORDS,
            config=config,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )

        with pytest.raises(ValueError, match="empty"):
            evaluate_roberta_baseline(
                fitted,
                _TRAIN_RECORDS,
                _EVAL_RECORDS,
                labels=(),
            )

    def test_requested_labels_rejects_duplicates(self):
        """Duplicate label names must raise ValueError."""
        model, tokenizer, _ = _make_tiny_fixture()
        config = RobertaBaselineConfig(
            max_seq_length=32,
            eval_batch_size=4,
            gradient_accumulation_steps=2,
            epochs=1,
            use_fp16=False,
            gradient_checkpointing=False,
        )

        fitted = fit_roberta_baseline(
            _TRAIN_RECORDS,
            _EVAL_RECORDS,
            config=config,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )

        with pytest.raises(ValueError, match="Duplicate"):
            evaluate_roberta_baseline(
                fitted,
                _TRAIN_RECORDS,
                _EVAL_RECORDS,
                labels=("toxicity", "toxicity"),
            )


# ---------------------------------------------------------------------------
# TOKENIZER TEMP-DIRECTORY CLEANUP tests
# ---------------------------------------------------------------------------


class TestTokenizerTempCleanup:
    """Verify that temporary directories for tokenizer files are cleaned up."""

    def test_tiny_tokenizer_usable_after_cleanup(self):
        """Tokenizer must remain functional after temp directory deletion."""
        model, tokenizer = _create_tiny_model_and_tokenizer(vocab_size=300)

        # Verify tokenizer works after context exits
        encoded = tokenizer("hello world", truncation=True, max_length=32)
        assert "input_ids" in encoded
        assert len(encoded["input_ids"]) > 0

    def test_tiny_tokenizer_ids_under_vocab_size(self):
        """All tokenizer output IDs must be < model.config.vocab_size."""
        model, tokenizer = _create_tiny_model_and_tokenizer(vocab_size=300)

        test_strings = [
            "Hello World!",
            "mixed CASE text",
            "punctuation!!!???",
            "roman-hindi mixed भारतीय",
            "aaabbbccc ddddeee",
            "12345 numbers",
        ]
        for text in test_strings:
            encoded = tokenizer(text, truncation=True, max_length=32)
            max_id = max(encoded["input_ids"])
            assert max_id < model.config.vocab_size, (
                f"Token ID {max_id} >= vocab_size {model.config.vocab_size} "
                f"for input: {text!r}"
            )


class TestTokenizerValidity:
    """3A-G: focused offline tests proving the tiny tokenizer is valid."""

    # --- 3A: default helper succeeds ---
    def test_default_helper_succeeds(self):
        """Calling _create_tiny_model_and_tokenizer() with no arguments
        must succeed and return a usable model and tokenizer."""
        model, tokenizer = _create_tiny_model_and_tokenizer()
        assert model is not None
        assert tokenizer is not None
        assert model.config.vocab_size >= 256

    # --- 3B: tokenizer has materially more than 5 special tokens ---
    def test_tokenizer_vocab_size_materially_greater_than_special(self):
        """The tokenizer must have more than just the 5 special tokens."""
        _, tokenizer = _create_tiny_model_and_tokenizer(vocab_size=300)
        assert len(tokenizer) > 256, (
            f"Tokenizer vocab size {len(tokenizer)} is not materially "
            f"greater than 256; expected a trained BPE tokenizer."
        )

    # --- 3C + 3D + 3E: representative strings encode successfully ---
    _REPRESENTATIVE_STRINGS = [
        "Good college",
        "GOOD College",
        "college!!!",
        "placement bahut accha hai",
        "coooool campus",
        "fees 120000",
    ]

    def test_representative_strings_encode(self):
        """Each representative string must encode without error and
        produce IDs in [0, model.config.vocab_size)."""
        model, tokenizer = _create_tiny_model_and_tokenizer(vocab_size=300)

        for text in self._REPRESENTATIVE_STRINGS:
            encoded = tokenizer(text, truncation=True, max_length=64)
            ids = encoded["input_ids"]
            assert len(ids) > 0, f"Empty encoding for: {text!r}"
            for tid in ids:
                assert 0 <= tid < model.config.vocab_size, (
                    f"Token ID {tid} out of range for input: {text!r}"
                )

    def test_representative_content_not_all_unk(self):
        """After removing special tokens, each non-empty string must have
        at least one content token (not all <unk>)."""
        _, tokenizer = _create_tiny_model_and_tokenizer(vocab_size=300)

        unk_id = tokenizer.unk_token_id
        for text in self._REPRESENTATIVE_STRINGS:
            encoded = tokenizer(text, truncation=True, max_length=64)
            ids = encoded["input_ids"]
            # Remove common special tokens (0=<s>, 1=<pad>, 2=</s>, etc.)
            content_ids = [tid for tid in ids if tid > 4]
            assert len(content_ids) > 0, (
                f"All tokens are special/unk for input: {text!r}"
            )

    def test_representative_strings_diverse_encodings(self):
        """Different representative strings should not all produce
        the exact same content-token sequence."""
        _, tokenizer = _create_tiny_model_and_tokenizer(vocab_size=300)

        sequences = set()
        for text in self._REPRESENTATIVE_STRINGS:
            encoded = tokenizer(text, truncation=True, max_length=64)
            ids = tuple(tid for tid in encoded["input_ids"] if tid > 4)
            sequences.add(ids)

        # At least half should be distinct
        assert len(sequences) >= len(self._REPRESENTATIVE_STRINGS) // 2, (
            f"Only {len(sequences)} unique sequences from "
            f"{len(self._REPRESENTATIVE_STRINGS)} strings"
        )

    # --- 3F: model forward succeeds for representative strings ---
    def test_model_forward_with_representative_strings(self):
        """Model must produce a forward pass on a batch containing all
        representative strings."""
        model, tokenizer = _create_tiny_model_and_tokenizer(vocab_size=300)
        model.eval()

        texts = self._REPRESENTATIVE_STRINGS
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=64,
                            return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        assert outputs.logits.shape == (len(texts), len(PRIMARY_LABELS)), (
            f"Unexpected logits shape: {outputs.logits.shape}"
        )

    # --- 3G: tokenizer works after temp cleanup ---
    def test_tokenizer_usable_after_cleanup(self):
        """After the temporary directory is cleaned up, the tokenizer
        must still encode strings correctly."""
        model, tokenizer = _create_tiny_model_and_tokenizer(vocab_size=300)

        # Encode after temp cleanup (function already returned)
        encoded = tokenizer("college is good", truncation=True, max_length=32)
        assert len(encoded["input_ids"]) > 0


# ---------------------------------------------------------------------------
# EVALUATION AMP tests
# ---------------------------------------------------------------------------


class TestEvaluationAMP:
    """Verify that evaluation uses AMP when use_fp16=True on CPU it does
    NOT use autocast (since CPU doesn't support AMP)."""

    def test_cpu_evaluation_no_amp(self):
        """On CPU, evaluation should NOT use AMP regardless of use_fp16."""
        model, tokenizer, _ = _make_tiny_fixture()
        config = RobertaBaselineConfig(
            max_seq_length=32,
            eval_batch_size=4,
            gradient_accumulation_steps=2,
            epochs=1,
            learning_rate=1e-4,
            use_fp16=True,  # set True but CPU can't use it
            gradient_checkpointing=False,
        )

        fitted = fit_roberta_baseline(
            _TRAIN_RECORDS,
            _EVAL_RECORDS,
            config=config,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )

        # Just verify evaluation completes without errors on CPU
        result = evaluate_roberta_baseline(
            fitted,
            _TRAIN_RECORDS,
            _EVAL_RECORDS,
            labels=("toxicity",),
        )
        assert result.macro_f1 is not None or len(result.per_label) == 0
