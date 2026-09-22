"""Tests for rrm.baseline_tfidf_lr.

All tests use tiny deterministic fixtures.  No synthetic-pilot performance
scores are reported as scientific evidence.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict

import numpy
import pytest

from rrm.baseline_tfidf_lr import (
    PRIMARY_LABELS,
    UNKNOWN_LABEL,
    BaselineEvaluation,
    FittedTfidfLogRegBaseline,
    LabelMetrics,
    TfidfLogRegConfig,
    evaluate_tfidf_logreg,
    fit_tfidf_logreg,
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
) -> Dict[str, Any]:
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


# --- Train split: spam and toxicity have both classes; deception is all -1 ---

_TRAIN_RECORDS = [
    _make_record("t1", "great college amazing faculty", spam=0, toxicity=0),
    _make_record("t2", "worst experience ever terrible", spam=1, toxicity=0),
    _make_record("t3", "love this place so much fun", spam=0, toxicity=0),
    _make_record("t4", "scam scam scam fake reviews everywhere", spam=1, toxicity=1),
    _make_record("t5", "okay nothing special", spam=0, toxicity=0),
    _make_record("t6", "hate it worst waste of money", spam=0, toxicity=1),
]


# --- Eval split: same label distribution ---

_EVAL_RECORDS = [
    _make_record("e1", "nice campus good teachers", spam=0, toxicity=0),
    _make_record("e2", "terrible bad do not join", spam=1, toxicity=0),
    _make_record("e3", "fantastic experience loved it", spam=0, toxicity=0),
    _make_record("e4", "spam spam fake bot comments", spam=1, toxicity=1),
    _make_record("e5", "average college decent", spam=0, toxicity=0),
    _make_record("e6", "awful garbage worthless", spam=0, toxicity=1),
]


# --- Train split where deception is UNKNOWN for all ---

_DECEPTION_UNKNOWN_TRAIN = [
    _make_record("d1", "some text here", deception=UNKNOWN_LABEL),
    _make_record("d2", "other text here", deception=UNKNOWN_LABEL),
    _make_record("d3", "more text content", deception=UNKNOWN_LABEL),
]


# --- Train split where off_topic has only one class (all 0) ---

_SINGLE_CLASS_TRAIN = [
    _make_record("s1", "text about college", off_topic=0),
    _make_record("s2", "more college text", off_topic=0),
    _make_record("s3", "campus life review", off_topic=0),
]


# --- Hinglish and Roman-Hindi records ---

_HINGLISH_RECORDS = [
    _make_record("h1", "hostel ka food bahut achha hai", spam=0),
    _make_record("h2", "placement scene thoda weak hai", spam=0),
    _make_record("h3", "scam scam scam fake admission", spam=1),
]


_ROMAN_HINDI_RECORDS = [
    _make_record("r1", "college ka campus accha hai", spam=0),
    _make_record("r2", "boring lectures useless profs", spam=0),
    _make_record("r3", "fraud fraud fraud dont trust", spam=1),
]


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_config_creation(self):
        cfg = TfidfLogRegConfig()
        assert cfg.random_state == 42
        assert cfg.ngram_range == (1, 2)
        assert cfg.min_df == 1
        assert cfg.max_df == 1.0
        assert cfg.max_features is None
        assert cfg.sublinear_tf is False
        assert cfg.C == 1.0
        assert cfg.max_iter == 1000
        assert cfg.solver == "liblinear"

    def test_invalid_ngram_lower_negative(self):
        with pytest.raises(ValueError, match=">= 1"):
            TfidfLogRegConfig(ngram_range=(0, 2))

    def test_invalid_ngram_upper_negative(self):
        with pytest.raises(ValueError, match=">= 1"):
            TfidfLogRegConfig(ngram_range=(1, 0))

    def test_invalid_ngram_lower_above_upper(self):
        with pytest.raises(ValueError, match="lower ngram"):
            TfidfLogRegConfig(ngram_range=(3, 2))

    def test_invalid_min_df(self):
        with pytest.raises(ValueError, match="min_df"):
            TfidfLogRegConfig(min_df=0)

    def test_invalid_max_df_zero(self):
        with pytest.raises(ValueError, match="max_df"):
            TfidfLogRegConfig(max_df=0.0)

    def test_invalid_max_df_over_one(self):
        with pytest.raises(ValueError, match="max_df"):
            TfidfLogRegConfig(max_df=1.5)

    def test_invalid_max_features(self):
        with pytest.raises(ValueError, match="max_features"):
            TfidfLogRegConfig(max_features=0)

    def test_invalid_C_zero(self):
        with pytest.raises(ValueError, match="C"):
            TfidfLogRegConfig(C=0.0)

    def test_invalid_C_negative(self):
        with pytest.raises(ValueError, match="C"):
            TfidfLogRegConfig(C=-1.0)

    def test_invalid_max_iter(self):
        with pytest.raises(ValueError, match="max_iter"):
            TfidfLogRegConfig(max_iter=0)

    def test_invalid_random_state_type(self):
        with pytest.raises(TypeError, match="random_state"):
            TfidfLogRegConfig(random_state="42")


# ---------------------------------------------------------------------------
# Record validation tests
# ---------------------------------------------------------------------------


class TestRecordValidation:
    def test_empty_review_id_rejected(self):
        with pytest.raises(ValueError, match="review_id"):
            fit_tfidf_logreg([{"review_id": "", "review_text": "text"}])

    def test_non_string_review_text_rejected(self):
        with pytest.raises(TypeError, match="review_text"):
            fit_tfidf_logreg([{"review_id": "r1", "review_text": 123}])

    def test_invalid_label_value_rejected(self):
        with pytest.raises(ValueError, match="label"):
            fit_tfidf_logreg([{"review_id": "r1", "review_text": "text", "spam": 2}])

    def test_string_label_value_not_coerced(self):
        with pytest.raises(ValueError, match="label"):
            fit_tfidf_logreg(
                [{"review_id": "r1", "review_text": "text", "spam": "1"}]
            )

    def test_duplicate_review_id_within_split_rejected(self):
        records = [
            _make_record("r1", "text one", spam=0),
            _make_record("r1", "text two", spam=1),
        ]
        with pytest.raises(ValueError, match="Duplicate review_id"):
            fit_tfidf_logreg(records)

    def test_non_mapping_record_rejected(self):
        with pytest.raises(TypeError, match="not a mapping"):
            fit_tfidf_logreg(["not a dict"])  # type: ignore[arg-type]

    # --- Missing label field tests (audit correction 1) ---

    def test_missing_requested_label_rejected_in_training(self):
        """Missing label key must raise ValueError, NOT silently become -1."""
        # Build a record missing the 'pii' label field entirely
        bad_record = {
            "review_id": "r1",
            "review_text": "some text",
            "spam": 0,
            "deception": UNKNOWN_LABEL,
            "toxicity": 0,
            "advertising": 0,
            "off_topic": 0,
        }
        with pytest.raises(ValueError, match="missing required label field 'pii'"):
            fit_tfidf_logreg([bad_record])

    def test_missing_non_requested_label_allowed(self):
        """Labels not in the requested `labels` tuple may be absent."""
        records = [
            {"review_id": "r1", "review_text": "some text", "spam": 0},
        ]
        # Request only spam; toxicity/others may be absent from records
        fitted = fit_tfidf_logreg(records, labels=("spam",))
        assert fitted is not None

    def test_explicit_unknown_label_minus_one_valid(self):
        """Explicit -1 for a label must be accepted and masked."""
        fitted = fit_tfidf_logreg(_DECEPTION_UNKNOWN_TRAIN)
        assert "deception" not in fitted.trained_labels
        assert "deception" in fitted.skipped_labels

    def test_missing_label_in_eval_rejected(self):
        """Evaluation records with missing label keys are also rejected."""
        train = [
            _make_record("t1", "some text", spam=0),
            _make_record("t2", "other text", spam=1),
        ]
        # Build eval record missing the 'pii' label field
        eval_rec = [
            {
                "review_id": "e1",
                "review_text": "more text",
                "spam": 0,
                "deception": UNKNOWN_LABEL,
                "toxicity": 0,
                "advertising": 0,
                "off_topic": 0,
            },
        ]
        fitted = fit_tfidf_logreg(train)
        with pytest.raises(ValueError, match="missing required label field 'pii'"):
            evaluate_tfidf_logreg(fitted, train, eval_rec)


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------


class TestTraining:
    def test_vectorizer_fits_successfully(self):
        fitted = fit_tfidf_logreg(_TRAIN_RECORDS)
        assert fitted.vectorizer is not None

    def test_word_analyzer_used(self):
        fitted = fit_tfidf_logreg(_TRAIN_RECORDS)
        # The vectorizer should have a vocabulary with word-level entries
        vocab = fitted.vectorizer.vocabulary_
        assert len(vocab) > 0
        # Should contain multi-word ngrams (bigrams)
        bigrams = [g for g in vocab if " " in g]
        assert len(bigrams) > 0

    def test_configured_ngram_range_respected(self):
        # Use (1, 1) only — no bigrams should appear
        cfg = TfidfLogRegConfig(ngram_range=(1, 1))
        fitted = fit_tfidf_logreg(_TRAIN_RECORDS, config=cfg)
        vocab = fitted.vectorizer.vocabulary_
        bigrams = [g for g in vocab if " " in g]
        assert len(bigrams) == 0

    def test_train_produces_models_for_valid_binary_labels(self):
        fitted = fit_tfidf_logreg(_TRAIN_RECORDS)
        # spam and toxicity both have 0 and 1 -> should be trained
        assert "spam" in fitted.trained_labels
        assert "toxicity" in fitted.trained_labels
        assert "deception" not in fitted.trained_labels

    def test_unknown_rows_masked(self):
        fitted = fit_tfidf_logreg(_DECEPTION_UNKNOWN_TRAIN)
        assert "deception" not in fitted.trained_labels
        assert "deception" in fitted.skipped_labels
        assert "both supervised classes" in fitted.skipped_labels["deception"]

    def test_unknown_never_converted_to_zero(self):
        # deception has only -1 -> skipped, never trained as all-0
        fitted = fit_tfidf_logreg(_DECEPTION_UNKNOWN_TRAIN)
        assert "deception" not in fitted.models
        assert "deception" not in fitted.trained_labels

    def test_label_with_one_class_is_skipped(self):
        fitted = fit_tfidf_logreg(_SINGLE_CLASS_TRAIN)
        assert "off_topic" in fitted.skipped_labels
        assert "only one class" not in fitted.skipped_labels["off_topic"]
        assert "both supervised classes" in fitted.skipped_labels["off_topic"]

    def test_skipped_label_contains_reason(self):
        fitted = fit_tfidf_logreg(_DECEPTION_UNKNOWN_TRAIN)
        reason = fitted.skipped_labels["deception"]
        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_deception_skipped_safely(self):
        fitted = fit_tfidf_logreg(_DECEPTION_UNKNOWN_TRAIN)
        # Should not crash; deception should be skipped
        assert fitted is not None
        assert "deception" in fitted.skipped_labels

    def test_result_dataclass_is_frozen(self):
        fitted = fit_tfidf_logreg(_TRAIN_RECORDS)
        with pytest.raises(dataclasses.FrozenInstanceError):
            fitted.trained_labels = ("spam",)  # type: ignore[misc]

    def test_hinglish_text_accepted(self):
        fitted = fit_tfidf_logreg(_HINGLISH_RECORDS)
        assert "spam" in fitted.trained_labels

    def test_roman_hindi_text_accepted(self):
        fitted = fit_tfidf_logreg(_ROMAN_HINDI_RECORDS)
        assert "spam" in fitted.trained_labels

    def test_review_text_is_only_feature_source(self):
        # Train two identical models with different config values for
        # non-text-irrelevant fields (e.g. random_state doesn't affect
        # TF-IDF vocabulary).  Verify the vectorizer vocabulary is driven
        # by review_text only.
        fitted = fit_tfidf_logreg(_TRAIN_RECORDS)
        vocab = fitted.vectorizer.vocabulary_
        # All vocabulary terms should come from the review_text strings
        all_text = " ".join(r["review_text"] for r in _TRAIN_RECORDS)
        for term in vocab:
            # Each term should be findable in the normalized text
            # (we check the raw token, not the normalized, because
            # the vectorizer lowercases internally)
            assert term.lower() in all_text.lower() or " " in term


# ---------------------------------------------------------------------------
# Leakage tests
# ---------------------------------------------------------------------------


class TestLeakage:
    def test_same_review_id_across_splits_rejected(self):
        train = [_make_record("shared", "unique train text", spam=0)]
        eval_rec = [_make_record("shared", "completely different eval text", spam=1)]
        with pytest.raises(ValueError, match="review_id.*appears in both"):
            validate_no_exact_leakage(train, eval_rec)

    def test_exact_normalized_duplicate_across_splits_rejected(self):
        train = [_make_record("t1", "some review text here", spam=0)]
        eval_rec = [_make_record("e1", "some review text here", spam=1)]
        with pytest.raises(ValueError, match="normalized text.*matches a train"):
            validate_no_exact_leakage(train, eval_rec)

    def test_case_whitespace_normalized_duplicate_rejected(self):
        train = [_make_record("t1", "some review text here", spam=0)]
        eval_rec = [_make_record("e1", "  Some   Review   Text   Here  ", spam=1)]
        with pytest.raises(ValueError, match="normalized text.*matches a train"):
            validate_no_exact_leakage(train, eval_rec)

    def test_different_nonduplicate_texts_allowed(self):
        # Should NOT raise
        validate_no_exact_leakage(
            [_make_record("t1", "alpha beta gamma", spam=0)],
            [_make_record("e1", "delta epsilon zeta", spam=1)],
        )


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluation:
    def test_train_only_vocabulary_behavior(self):
        """Eval-only words should not appear in the vocabulary."""
        train = [_make_record("t1", "alpha beta gamma", spam=0)]
        eval_rec = [_make_record("e1", "delta epsilon zeta", spam=1)]
        fitted = fit_tfidf_logreg(train)
        eval_result = evaluate_tfidf_logreg(fitted, train, eval_rec)
        vocab = fitted.vectorizer.vocabulary_
        # eval-only words should not be in vocab
        for word in ["delta", "epsilon", "zeta"]:
            assert word not in vocab

    def test_evaluation_transforms_not_fits(self):
        """The vectorizer must not change after transform on eval data."""
        fitted = fit_tfidf_logreg(_TRAIN_RECORDS)
        vocab_before = dict(fitted.vectorizer.vocabulary_)
        evaluate_tfidf_logreg(fitted, _TRAIN_RECORDS, _EVAL_RECORDS)
        vocab_after = dict(fitted.vectorizer.vocabulary_)
        assert vocab_before == vocab_after

    def test_precision_computed(self):
        fitted = fit_tfidf_logreg(_TRAIN_RECORDS)
        result = evaluate_tfidf_logreg(fitted, _TRAIN_RECORDS, _EVAL_RECORDS)
        spam_metrics = _find_metrics(result, "spam")
        assert spam_metrics is not None
        assert 0.0 <= spam_metrics.precision <= 1.0

    def test_recall_computed(self):
        fitted = fit_tfidf_logreg(_TRAIN_RECORDS)
        result = evaluate_tfidf_logreg(fitted, _TRAIN_RECORDS, _EVAL_RECORDS)
        spam_metrics = _find_metrics(result, "spam")
        assert spam_metrics is not None
        assert 0.0 <= spam_metrics.recall <= 1.0

    def test_f1_computed(self):
        fitted = fit_tfidf_logreg(_TRAIN_RECORDS)
        result = evaluate_tfidf_logreg(fitted, _TRAIN_RECORDS, _EVAL_RECORDS)
        spam_metrics = _find_metrics(result, "spam")
        assert spam_metrics is not None
        assert 0.0 <= spam_metrics.f1 <= 1.0

    def test_auprc_computed(self):
        fitted = fit_tfidf_logreg(_TRAIN_RECORDS)
        result = evaluate_tfidf_logreg(fitted, _TRAIN_RECORDS, _EVAL_RECORDS)
        spam_metrics = _find_metrics(result, "spam")
        assert spam_metrics is not None
        assert 0.0 <= spam_metrics.auprc <= 1.0

    def test_macro_f1_uses_valid_labels(self):
        fitted = fit_tfidf_logreg(_TRAIN_RECORDS)
        result = evaluate_tfidf_logreg(fitted, _TRAIN_RECORDS, _EVAL_RECORDS)
        # Only spam and toxicity should be evaluated
        assert "spam" in result.evaluated_labels
        assert "toxicity" in result.evaluated_labels
        assert result.macro_f1 is not None
        # macro_f1 should be the mean of the evaluated labels' F1
        f1s = [m.f1 for m in result.per_label]
        assert numpy.isclose(result.macro_f1, float(numpy.mean(f1s)))

    def test_single_class_eval_label_skipped(self):
        # off_topic is all 0 in both train and eval
        train = [
            _make_record("s1", "text one", off_topic=0),
            _make_record("s2", "text two", off_topic=0),
        ]
        eval_rec = [
            _make_record("e1", "text three", off_topic=0),
            _make_record("e2", "text four", off_topic=0),
        ]
        fitted = fit_tfidf_logreg(train)
        result = evaluate_tfidf_logreg(fitted, train, eval_rec)
        assert "off_topic" in result.skipped_labels

    def test_unknown_eval_rows_masked(self):
        train = [
            _make_record("t1", "great college good teachers", spam=0),
            _make_record("t2", "terrible bad awful spam", spam=1),
            _make_record("t3", "nice campus clean rooms", spam=0),
            _make_record("t4", "worst fake scam bot", spam=1),
        ]
        eval_rec = [
            _make_record("e1", "decent college okay", spam=0),
            _make_record("e2", "spam fake bot post", spam=1),
            _make_record("e3", "unknown label row here", spam=UNKNOWN_LABEL),
        ]
        fitted = fit_tfidf_logreg(train)
        result = evaluate_tfidf_logreg(fitted, train, eval_rec)
        spam_metrics = _find_metrics(result, "spam")
        assert spam_metrics is not None
        # e3 is masked, so support should be 2 (e1 and e2)
        assert spam_metrics.support == 2

    def test_repeated_evaluation_deterministic(self):
        fitted = fit_tfidf_logreg(_TRAIN_RECORDS)
        result1 = evaluate_tfidf_logreg(fitted, _TRAIN_RECORDS, _EVAL_RECORDS)
        result2 = evaluate_tfidf_logreg(fitted, _TRAIN_RECORDS, _EVAL_RECORDS)
        assert result1 == result2

    def test_result_dataclasses_are_frozen(self):
        fitted = fit_tfidf_logreg(_TRAIN_RECORDS)
        with pytest.raises(dataclasses.FrozenInstanceError):
            fitted.config = fitted.config  # type: ignore[misc]

        result = evaluate_tfidf_logreg(fitted, _TRAIN_RECORDS, _EVAL_RECORDS)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.macro_f1 = 0.0  # type: ignore[misc]

    def test_no_moderation_action_fields_exist(self):
        fitted = fit_tfidf_logreg(_TRAIN_RECORDS)
        result = evaluate_tfidf_logreg(fitted, _TRAIN_RECORDS, _EVAL_RECORDS)

        # Verify no moderation-related fields in result types
        for field in dataclasses.fields(BaselineEvaluation):
            assert "ban" not in field.name
            assert "delete" not in field.name
            assert "action" not in field.name
            assert "moderate" not in field.name

        for field in dataclasses.fields(LabelMetrics):
            assert "ban" not in field.name
            assert "delete" not in field.name
            assert "action" not in field.name

        for field in dataclasses.fields(FittedTfidfLogRegBaseline):
            assert "ban" not in field.name
            assert "delete" not in field.name
            assert "action" not in field.name

    def test_leakage_guard_blocks_eval_call(self):
        train = [_make_record("r1", "shared text", spam=0)]
        eval_rec = [_make_record("r2", "shared text", spam=1)]
        fitted = fit_tfidf_logreg(train)
        with pytest.raises(ValueError, match="Leakage"):
            evaluate_tfidf_logreg(fitted, train, eval_rec)

    def test_eval_empty_eval_split(self):
        fitted = fit_tfidf_logreg(_TRAIN_RECORDS)
        result = evaluate_tfidf_logreg(fitted, _TRAIN_RECORDS, [])
        # All labels should be skipped (no supervised eval data)
        assert len(result.evaluated_labels) == 0
        assert result.macro_f1 is None


# ---------------------------------------------------------------------------
# Generator input tests (audit correction 2)
# ---------------------------------------------------------------------------


def _record_generator(records):
    """Yield records one at a time (simulates streaming input)."""
    for r in records:
        yield r


class TestGeneratorInputs:
    def test_generator_train_records_work(self):
        """Evaluation should accept generator train_records."""
        train_list = [
            _make_record("t1", "great college", spam=0),
            _make_record("t2", "terrible bad", spam=1),
        ]
        eval_rec = [
            _make_record("e1", "nice campus", spam=0),
            _make_record("e2", "awful worst", spam=1),
        ]
        fitted = fit_tfidf_logreg(train_list)
        # train_records as generator
        result = evaluate_tfidf_logreg(
            fitted, _record_generator(train_list), eval_rec
        )
        assert "spam" in result.evaluated_labels

    def test_generator_eval_records_work(self):
        """Evaluation should accept generator eval_records."""
        train_list = [
            _make_record("t1", "great college", spam=0),
            _make_record("t2", "terrible bad", spam=1),
        ]
        eval_list = [
            _make_record("e1", "nice campus", spam=0),
            _make_record("e2", "awful worst", spam=1),
        ]
        fitted = fit_tfidf_logreg(train_list)
        # eval_records as generator
        result = evaluate_tfidf_logreg(
            fitted, train_list, _record_generator(eval_list)
        )
        assert "spam" in result.evaluated_labels

    def test_generator_produces_same_result_as_list(self):
        """Generator input must produce identical results to list input."""
        train_list = [
            _make_record("t1", "alpha beta gamma", spam=0),
            _make_record("t2", "delta epsilon zeta", spam=1),
        ]
        eval_list = [
            _make_record("e1", "alpha beta delta", spam=0),
            _make_record("e2", "zeta epsilon gamma", spam=1),
        ]

        fitted = fit_tfidf_logreg(train_list)

        # List inputs
        result_list = evaluate_tfidf_logreg(fitted, train_list, eval_list)

        # Generator inputs
        result_gen = evaluate_tfidf_logreg(
            fitted,
            _record_generator(train_list),
            _record_generator(eval_list),
        )

        assert result_list == result_gen

    def test_both_generators_and_malformed_eval_rejected(self):
        """Malformed eval records are rejected regardless of input type."""
        train = [
            _make_record("t1", "some text", spam=0),
            _make_record("t2", "other text", spam=1),
        ]
        # Build eval record missing the 'pii' label field
        bad_eval = [
            {
                "review_id": "e1",
                "review_text": "more text",
                "spam": 0,
                "deception": UNKNOWN_LABEL,
                "toxicity": 0,
                "advertising": 0,
                "off_topic": 0,
            },
        ]
        fitted = fit_tfidf_logreg(train)
        with pytest.raises(ValueError, match="missing required label field"):
            evaluate_tfidf_logreg(
                fitted, _record_generator(train), _record_generator(bad_eval)
            )


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_repeated_fit_produces_equal_vocabulary(self):
        fitted1 = fit_tfidf_logreg(_TRAIN_RECORDS)
        fitted2 = fit_tfidf_logreg(_TRAIN_RECORDS)
        # Vocabulary from TF-IDF vectorizer is deterministic
        assert fitted1.vectorizer.vocabulary_ == fitted2.vectorizer.vocabulary_

    def test_repeated_evaluation_is_deterministic(self):
        fitted = fit_tfidf_logreg(_TRAIN_RECORDS)
        result1 = evaluate_tfidf_logreg(
            fitted, _TRAIN_RECORDS, _EVAL_RECORDS
        )
        result2 = evaluate_tfidf_logreg(
            fitted, _TRAIN_RECORDS, _EVAL_RECORDS
        )
        assert result1 == result2


# ---------------------------------------------------------------------------
# Metrics edge cases
# ---------------------------------------------------------------------------


class TestMetricsEdgeCases:
    def test_metrics_validity_with_clean_separation(self):
        """On a tiny dataset with clear class separation, metrics should be
        well-formed and deterministic across repeated calls."""
        # Train: spam=1 contains "spam", spam=0 does not
        train = [
            _make_record("p1", "spam post here", spam=1),
            _make_record("p2", "spam content here", spam=1),
            _make_record("p3", "nice review text", spam=0),
            _make_record("p4", "great review content", spam=0),
        ]
        # Eval: same vocabulary, different IDs, no exact duplicates
        eval_rec = [
            _make_record("p1b", "spam junk text", spam=1),
            _make_record("p2b", "spam bot link", spam=1),
            _make_record("p3b", "nice review post", spam=0),
            _make_record("p4b", "great review entry", spam=0),
        ]
        fitted = fit_tfidf_logreg(train)
        result = evaluate_tfidf_logreg(fitted, train, eval_rec)
        spam_metrics = _find_metrics(result, "spam")
        assert spam_metrics is not None
        # Metrics are valid (deterministic quality, not arbitrary threshold)
        assert 0.0 <= spam_metrics.precision <= 1.0
        assert 0.0 <= spam_metrics.recall <= 1.0
        assert 0.0 <= spam_metrics.f1 <= 1.0
        assert 0.0 <= spam_metrics.auprc <= 1.0
        assert spam_metrics.support == 4
        # Deterministic: repeat evaluation gives same result
        result2 = evaluate_tfidf_logreg(fitted, train, eval_rec)
        assert result == result2

    def test_label_with_only_positive_in_eval_skipped(self):
        train = [
            _make_record("t1", "spammy text content", spam=0),
            _make_record("t2", "spammy bot content", spam=1),
        ]
        eval_rec = [
            _make_record("e1", "spam keyword xyz", spam=1),
            _make_record("e2", "spam keyword abc", spam=1),
        ]
        fitted = fit_tfidf_logreg(train)
        result = evaluate_tfidf_logreg(fitted, train, eval_rec)
        # spam has only class 1 in eval -> should be skipped
        assert "spam" in result.skipped_labels

    def test_label_with_only_negative_in_eval_skipped(self):
        train = [
            _make_record("t1", "spammy text content", spam=0),
            _make_record("t2", "spammy bot content", spam=1),
        ]
        eval_rec = [
            _make_record("e1", "normal review text", spam=0),
            _make_record("e2", "genuine feedback here", spam=0),
        ]
        fitted = fit_tfidf_logreg(train)
        result = evaluate_tfidf_logreg(fitted, train, eval_rec)
        # spam has only class 0 in eval -> should be skipped
        assert "spam" in result.skipped_labels
        assert "only one class" in result.skipped_labels["spam"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_metrics(
    result: BaselineEvaluation, label: str
) -> LabelMetrics | None:
    for m in result.per_label:
        if m.label == label:
            return m
    return None
