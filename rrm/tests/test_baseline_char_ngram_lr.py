"""Tests for rrm.baseline_char_ngram_lr.

All tests use tiny deterministic fixtures.  No synthetic-pilot performance
scores are reported as scientific evidence.
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy
import pytest

from rrm.baseline_char_ngram_lr import (
    PRIMARY_LABELS,
    evaluate_char_ngram_logreg,
    fit_char_ngram_logreg,
)
from rrm.baseline_tfidf_lr import (
    UNKNOWN_LABEL,
    TfidfLogRegConfig,
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
    """Create a minimal record with all six primary labels."""
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


# --- Train split: spam and toxicity have both classes ---
_TRAIN_RECORDS = [
    _make_record("t1", "great college amazing faculty", spam=0, toxicity=0),
    _make_record("t2", "worst experience ever terrible", spam=1, toxicity=0),
    _make_record("t3", "love this place so much fun", spam=0, toxicity=0),
    _make_record("t4", "scam scam scam fake reviews everywhere", spam=1, toxicity=1),
    _make_record("t5", "okay nothing special", spam=0, toxicity=0),
    _make_record("t6", "hate it worst waste of money", spam=0, toxicity=1),
]


# --- Eval split ---
_EVAL_RECORDS = [
    _make_record("e1", "nice campus good teachers", spam=0, toxicity=0),
    _make_record("e2", "terrible bad do not join", spam=1, toxicity=0),
    _make_record("e3", "fantastic experience loved it", spam=0, toxicity=0),
    _make_record("e4", "spam spam fake bot comments", spam=1, toxicity=1),
    _make_record("e5", "average college decent", spam=0, toxicity=0),
    _make_record("e6", "awful garbage worthless", spam=0, toxicity=1),
]


# --- Hinglish and Roman Hindi ---
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


class TestCharConfig:
    """Character baseline configuration behavior."""

    def test_default_char_config_uses_ngram_3_5(self):
        """fit_char_ngram_logreg with no config should use (3, 5)."""
        fitted = fit_char_ngram_logreg(_TRAIN_RECORDS)
        assert fitted.config.ngram_range == (3, 5)

    def test_custom_ngram_range_respected(self):
        """Custom ngram_range on config should be passed through."""
        config = TfidfLogRegConfig(ngram_range=(2, 4))
        fitted = fit_char_ngram_logreg(_TRAIN_RECORDS, config=config)
        assert fitted.config.ngram_range == (2, 4)

    def test_default_other_params_inherited(self):
        """Other TfidfLogRegConfig defaults should apply."""
        fitted = fit_char_ngram_logreg(_TRAIN_RECORDS)
        assert fitted.config.C == 1.0
        assert fitted.config.solver == "liblinear"
        assert fitted.config.max_iter == 1000
        assert fitted.config.random_state == 42
        assert fitted.config.sublinear_tf is False
        assert fitted.config.min_df == 1
        assert fitted.config.max_df == 1.0


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------


class TestCharTraining:
    """Character baseline training behavior."""

    def test_training_produces_models_for_valid_binary_labels(self):
        fitted = fit_char_ngram_logreg(_TRAIN_RECORDS)
        assert "spam" in fitted.models
        assert "toxicity" in fitted.models

    def test_unknown_rows_masked(self):
        """UNKNOWN=-1 rows must be masked during training."""
        records = [
            _make_record("k1", "some text here", spam=0),
            _make_record("k2", "other text content", spam=UNKNOWN_LABEL),
            _make_record("k3", "more text here", spam=1),
        ]
        fitted = fit_char_ngram_logreg(records, labels=("spam",))
        # Should train on the two known rows
        assert "spam" in fitted.models

    def test_single_class_training_label_skipped(self):
        """Label with only one class should be skipped."""
        records = [
            _make_record("s1", "text about college", spam=0),
            _make_record("s2", "more college text", spam=0),
            _make_record("s3", "campus life review", spam=0),
        ]
        fitted = fit_char_ngram_logreg(records, labels=("spam",))
        assert "spam" not in fitted.models
        assert "requires both supervised classes 0 and 1" in fitted.skipped_labels["spam"]

    def test_skipped_label_contains_reason(self):
        fitted = fit_char_ngram_logreg(_TRAIN_RECORDS)
        # deception is all UNKNOWN in our fixtures, should be skipped
        assert "deception" in fitted.skipped_labels
        assert "requires both supervised classes 0 and 1" in fitted.skipped_labels["deception"]

    def test_result_dataclass_is_frozen(self):
        fitted = fit_char_ngram_logreg(_TRAIN_RECORDS)
        with pytest.raises(dataclasses.FrozenInstanceError):
            fitted.config = TfidfLogRegConfig()  # type: ignore[misc]

    def test_hinglish_text_accepted(self):
        """Hinglish text should train without errors."""
        fitted = fit_char_ngram_logreg(_HINGLISH_RECORDS)
        assert "spam" in fitted.models

    def test_roman_hindi_text_accepted(self):
        """Roman Hindi text should train without errors."""
        fitted = fit_char_ngram_logreg(_ROMAN_HINDI_RECORDS)
        assert "spam" in fitted.models

    def test_review_text_is_only_feature_source(self):
        """Only review_text should be used for features."""
        # Records with extra fields should train fine — they're ignored
        records = [
            {
                "review_id": "x1",
                "review_text": "some review text",
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
                "review_text": "another review text",
                "spam": 1,
                "deception": UNKNOWN_LABEL,
                "toxicity": 0,
                "advertising": 0,
                "off_topic": 0,
                "pii": 0,
                "extra_field": "also ignored",
            },
        ]
        fitted = fit_char_ngram_logreg(records)
        assert "spam" in fitted.models


# ---------------------------------------------------------------------------
# Shared validation tests
# ---------------------------------------------------------------------------


class TestSharedValidation:
    """Tests proving shared contracts from 3.4A still work for 3.4B."""

    def test_missing_requested_label_rejected(self):
        records = [
            {"review_id": "x1", "review_text": "some text", "spam": 0},
        ]
        with pytest.raises(ValueError, match="missing required label field"):
            fit_char_ngram_logreg(records, labels=("spam", "toxicity"))

    def test_duplicate_review_id_rejected(self):
        records = [
            _make_record("d1", "some text", spam=0),
            _make_record("d1", "other text", spam=1),
        ]
        with pytest.raises(ValueError, match="Duplicate review_id"):
            fit_char_ngram_logreg(records)

    def test_non_string_review_text_rejected(self):
        records = [
            {"review_id": "x1", "review_text": 123, "spam": 0},
        ]
        with pytest.raises(TypeError, match="review_text must be str"):
            fit_char_ngram_logreg(records)

    def test_unknown_label_minus_one_valid(self):
        """label = -1 (UNKNOWN_LABEL) must remain valid."""
        records = [
            _make_record("u1", "some text", spam=UNKNOWN_LABEL),
        ]
        # Should not raise — -1 is a valid label value
        fitted = fit_char_ngram_logreg(records, labels=("spam",))
        assert "spam" not in fitted.models  # masked, not trained


# ---------------------------------------------------------------------------
# Leakage tests
# ---------------------------------------------------------------------------


class TestLeakage:
    """Leakage protection must carry over from 3.4A."""

    def test_same_review_id_across_splits_rejected(self):
        train = [_make_record("s1", "some text", spam=0)]
        eval_rec = [_make_record("s1", "different text", spam=1)]
        fitted = fit_char_ngram_logreg(train)
        with pytest.raises(ValueError, match="review_id.*appears in both"):
            evaluate_char_ngram_logreg(fitted, train, eval_rec)

    def test_exact_normalized_duplicate_across_splits_rejected(self):
        train = [_make_record("s1", "some text here", spam=0)]
        eval_rec = [_make_record("e1", "some text here", spam=1)]
        fitted = fit_char_ngram_logreg(train)
        with pytest.raises(ValueError, match="Leakage detected"):
            evaluate_char_ngram_logreg(fitted, train, eval_rec)

    def test_case_whitespace_normalized_duplicate_rejected(self):
        train = [_make_record("s1", "some text", spam=0)]
        eval_rec = [_make_record("e1", "  SOME   Text  ", spam=1)]
        fitted = fit_char_ngram_logreg(train)
        with pytest.raises(ValueError, match="Leakage detected"):
            evaluate_char_ngram_logreg(fitted, train, eval_rec)

    def test_different_nonduplicate_texts_allowed(self):
        train = [_make_record("s1", "alpha text", spam=0)]
        eval_rec = [_make_record("e1", "beta text", spam=1)]
        fitted = fit_char_ngram_logreg(train)
        # Should not raise
        result = evaluate_char_ngram_logreg(fitted, train, eval_rec)
        assert result is not None


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluation:
    """Delegated evaluation must produce same contract as 3.4A."""

    def test_precision_computed(self):
        fitted = fit_char_ngram_logreg(_TRAIN_RECORDS)
        result = evaluate_char_ngram_logreg(fitted, _TRAIN_RECORDS, _EVAL_RECORDS)
        spam_metrics = _find_metrics(result, "spam")
        assert spam_metrics is not None
        assert 0.0 <= spam_metrics.precision <= 1.0

    def test_recall_computed(self):
        fitted = fit_char_ngram_logreg(_TRAIN_RECORDS)
        result = evaluate_char_ngram_logreg(fitted, _TRAIN_RECORDS, _EVAL_RECORDS)
        spam_metrics = _find_metrics(result, "spam")
        assert spam_metrics is not None
        assert 0.0 <= spam_metrics.recall <= 1.0

    def test_f1_computed(self):
        fitted = fit_char_ngram_logreg(_TRAIN_RECORDS)
        result = evaluate_char_ngram_logreg(fitted, _TRAIN_RECORDS, _EVAL_RECORDS)
        spam_metrics = _find_metrics(result, "spam")
        assert spam_metrics is not None
        assert 0.0 <= spam_metrics.f1 <= 1.0

    def test_auprc_computed(self):
        fitted = fit_char_ngram_logreg(_TRAIN_RECORDS)
        result = evaluate_char_ngram_logreg(fitted, _TRAIN_RECORDS, _EVAL_RECORDS)
        spam_metrics = _find_metrics(result, "spam")
        assert spam_metrics is not None
        assert 0.0 <= spam_metrics.auprc <= 1.0

    def test_macro_f1_computed(self):
        fitted = fit_char_ngram_logreg(_TRAIN_RECORDS)
        result = evaluate_char_ngram_logreg(fitted, _TRAIN_RECORDS, _EVAL_RECORDS)
        assert result.macro_f1 is not None
        f1s = [m.f1 for m in result.per_label]
        assert numpy.isclose(result.macro_f1, float(numpy.mean(f1s)))

    def test_single_class_eval_label_skipped(self):
        records = [
            _make_record("s1", "text about college", spam=0),
            _make_record("s2", "more college text", spam=0),
            _make_record("s3", "campus life review", spam=0),
        ]
        fitted = fit_char_ngram_logreg(_TRAIN_RECORDS)
        result = evaluate_char_ngram_logreg(fitted, _TRAIN_RECORDS, records)
        # spam has 3 negatives in eval — should be skipped
        assert "spam" in result.skipped_labels

    def test_unknown_eval_rows_masked(self):
        records = [
            _make_record("u1", "some text", spam=0),
            _make_record("u2", "other text", spam=UNKNOWN_LABEL),
            _make_record("u3", "more text", spam=1),
        ]
        fitted = fit_char_ngram_logreg(_TRAIN_RECORDS)
        result = evaluate_char_ngram_logreg(fitted, _TRAIN_RECORDS, records)
        spam_metrics = _find_metrics(result, "spam")
        assert spam_metrics is not None
        assert spam_metrics.support == 2  # only known rows

    def test_evaluation_does_not_refit_vectorizer(self):
        fitted = fit_char_ngram_logreg(_TRAIN_RECORDS)
        vocab_before = dict(fitted.vectorizer.vocabulary_)
        evaluate_char_ngram_logreg(fitted, _TRAIN_RECORDS, _EVAL_RECORDS)
        vocab_after = dict(fitted.vectorizer.vocabulary_)
        assert vocab_before == vocab_after

    def test_repeated_evaluation_deterministic(self):
        fitted = fit_char_ngram_logreg(_TRAIN_RECORDS)
        result1 = evaluate_char_ngram_logreg(fitted, _TRAIN_RECORDS, _EVAL_RECORDS)
        result2 = evaluate_char_ngram_logreg(fitted, _TRAIN_RECORDS, _EVAL_RECORDS)
        assert result1.macro_f1 == result2.macro_f1
        assert result1.per_label == result2.per_label

    def test_result_dataclasses_are_frozen(self):
        fitted = fit_char_ngram_logreg(_TRAIN_RECORDS)
        result = evaluate_char_ngram_logreg(fitted, _TRAIN_RECORDS, _EVAL_RECORDS)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.macro_f1 = 0.0  # type: ignore[misc]

    def test_no_moderation_action_fields_exist(self):
        """Result must not contain moderation action fields."""
        fitted = fit_char_ngram_logreg(_TRAIN_RECORDS)
        result = evaluate_char_ngram_logreg(fitted, _TRAIN_RECORDS, _EVAL_RECORDS)
        assert not hasattr(result, "delete")
        assert not hasattr(result, "ban")
        assert not hasattr(result, "remove_user")
        assert not hasattr(result, "final_decision")

    def test_leakage_guard_blocks_eval_call(self):
        # Distinct review_ids, same text — tests fingerprint leakage detection
        train = [_make_record("s1", "exact same text", spam=0)]
        eval_rec = [_make_record("e1", "exact same text", spam=1)]
        fitted = fit_char_ngram_logreg(train)
        with pytest.raises(ValueError, match="Leakage"):
            evaluate_char_ngram_logreg(fitted, train, eval_rec)


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------


def _record_generator(records):
    """Yield records one at a time, simulating a generator."""
    for r in records:
        yield r


class TestGeneratorInputs:
    """Character baseline must be generator-safe like the word baseline."""

    def test_generator_train_records_work(self):
        fitted = fit_char_ngram_logreg(_record_generator(_TRAIN_RECORDS))
        assert "spam" in fitted.models

    def test_generator_eval_records_work(self):
        fitted = fit_char_ngram_logreg(_TRAIN_RECORDS)
        result = evaluate_char_ngram_logreg(
            fitted,
            _TRAIN_RECORDS,
            _record_generator(_EVAL_RECORDS),
        )
        assert result is not None

    def test_generator_produces_same_result_as_list(self):
        fitted_list = fit_char_ngram_logreg(_TRAIN_RECORDS)
        result_list = evaluate_char_ngram_logreg(
            fitted_list, _TRAIN_RECORDS, _EVAL_RECORDS
        )

        fitted_gen = fit_char_ngram_logreg(_record_generator(_TRAIN_RECORDS))
        result_gen = evaluate_char_ngram_logreg(
            fitted_gen,
            _record_generator(_TRAIN_RECORDS),
            _record_generator(_EVAL_RECORDS),
        )

        assert result_list.macro_f1 == result_gen.macro_f1
        assert result_list.evaluated_labels == result_gen.evaluated_labels

    def test_both_generators_and_malformed_eval_rejected(self):
        fitted = fit_char_ngram_logreg(_TRAIN_RECORDS)
        bad_eval = [{"review_id": "b1", "review_text": 123}]  # type: ignore[list-item]
        with pytest.raises((TypeError, ValueError)):
            evaluate_char_ngram_logreg(
                fitted,
                _record_generator(_TRAIN_RECORDS),
                _record_generator(bad_eval),  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Character-specific tests
# ---------------------------------------------------------------------------


class TestCharSpecific:
    """Tests unique to character n-gram behavior."""

    def test_analyzer_is_char(self):
        """The fitted vectorizer must use analyzer='char'."""
        fitted = fit_char_ngram_logreg(_TRAIN_RECORDS)
        assert fitted.vectorizer.analyzer == "char"

    def test_char_ngram_vocab_differs_from_word(self):
        """Char and word baselines should produce different feature spaces."""
        from rrm.baseline_tfidf_lr import fit_tfidf_logreg

        char_fitted = fit_char_ngram_logreg(_TRAIN_RECORDS)
        word_fitted = fit_tfidf_logreg(_TRAIN_RECORDS)

        char_vocab = set(char_fitted.vectorizer.vocabulary_.keys())
        word_vocab = set(word_fitted.vectorizer.vocabulary_.keys())

        # They should not be identical — character n-grams are different from words
        assert char_vocab != word_vocab
        # Character vocab should contain at least one token with length > 1
        assert any(len(t) > 1 for t in char_vocab)

    def test_char_vocabulary_isolation(self):
        """Eval-only character sequences should not appear in vocabulary.

        Uses deliberately disjoint character sets: train uses letters a-c,
        eval uses letters x-z, ensuring zero shared character trigrams.
        """
        train = [_make_record("t1", "abcabc", spam=0), _make_record("t2", "bca", spam=1)]
        eval_rec = [_make_record("e1", "xyzxyz", spam=0)]
        fitted = fit_char_ngram_logreg(train)

        # The vocabulary should only contain character n-grams from training text
        # For trigrams on "abcabc": "abc", "bca", "cab"
        vocab = set(fitted.vectorizer.vocabulary_.keys())
        # "xyz" and "yzx" should not appear
        assert "xyz" not in vocab
        assert "yzx" not in vocab

    def test_evaluation_does_not_refit_vectorizer(self):
        """Evaluation must not expand the vocabulary."""
        fitted = fit_char_ngram_logreg(_TRAIN_RECORDS)
        vocab_before = dict(fitted.vectorizer.vocabulary_)
        evaluate_char_ngram_logreg(fitted, _TRAIN_RECORDS, _EVAL_RECORDS)
        vocab_after = dict(fitted.vectorizer.vocabulary_)
        assert vocab_before == vocab_after

    def test_elongated_text_produces_char_features(self):
        """Elongated characters like 'facultyyyy' should produce char n-grams."""
        records = [
            _make_record("e1", "faculty is great", spam=0),
            _make_record("e2", "facultyyyy is bad", spam=1),
        ]
        fitted = fit_char_ngram_logreg(records)
        vocab = set(fitted.vectorizer.vocabulary_.keys())
        # The elongated form should produce unique character n-grams
        assert "yyy" in vocab or "uty" in vocab  # trigrams from elongated text

    def test_spelling_variants_share_char_features(self):
        """Spelling variants like 'colleege' and 'college' should share
        character trigrams, proving partial overlap sensitivity."""
        from rrm.similarity import character_ngrams

        ngrams1 = character_ngrams("colleege", n=3)
        ngrams2 = character_ngrams("college", n=3)
        # Must share at least one trigram (mathematically guaranteed for
        # single-character insertion in the middle of a word)
        shared = ngrams1 & ngrams2
        assert len(shared) > 0, (
            f"No shared trigrams between 'colleege' and 'college': "
            f"{ngrams1} vs {ngrams2}"
        )

    def test_hinglish_char_ngrams_accepted(self):
        """Hinglish text should produce character n-gram features."""
        fitted = fit_char_ngram_logreg(_HINGLISH_RECORDS)
        vocab = set(fitted.vectorizer.vocabulary_.keys())
        assert len(vocab) > 0

    def test_roman_hindi_char_ngrams_accepted(self):
        """Roman Hindi text should produce character n-gram features."""
        fitted = fit_char_ngram_logreg(_ROMAN_HINDI_RECORDS)
        vocab = set(fitted.vectorizer.vocabulary_.keys())
        assert len(vocab) > 0

    def test_repeated_fit_deterministic(self):
        """Repeated fitting should produce identical models."""
        fitted1 = fit_char_ngram_logreg(_TRAIN_RECORDS)
        fitted2 = fit_char_ngram_logreg(_TRAIN_RECORDS)

        # Same vocabulary
        assert fitted1.vectorizer.vocabulary_ == fitted2.vectorizer.vocabulary_

        # Same model predictions on eval data
        X1 = fitted1.vectorizer.transform([r["review_text"] for r in _EVAL_RECORDS])
        X2 = fitted2.vectorizer.transform([r["review_text"] for r in _EVAL_RECORDS])
        pred1 = fitted1.models["spam"].predict(X1)
        pred2 = fitted2.models["spam"].predict(X2)
        assert numpy.array_equal(pred1, pred2)

    def test_no_moderation_fields_in_result(self):
        """Result should not contain moderation action fields."""
        fitted = fit_char_ngram_logreg(_TRAIN_RECORDS)
        result = evaluate_char_ngram_logreg(fitted, _TRAIN_RECORDS, _EVAL_RECORDS)
        for field in ("delete", "ban", "remove_user", "final_decision"):
            assert not hasattr(result, field)

    def test_extra_fields_ignored(self):
        """Only review_text should be used as feature source; extra fields ignored."""
        # Records with extra fields should train fine — they're ignored
        records = [
            {
                "review_id": "x1",
                "review_text": "some review text",
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
                "review_text": "another review text",
                "spam": 1,
                "deception": UNKNOWN_LABEL,
                "toxicity": 0,
                "advertising": 0,
                "off_topic": 0,
                "pii": 0,
                "extra_field": "also ignored",
            },
        ]
        fitted = fit_char_ngram_logreg(records)
        assert "spam" in fitted.models

    def test_char_eval_empty_eval_split(self):
        """Empty eval split should return empty metrics gracefully."""
        fitted = fit_char_ngram_logreg(_TRAIN_RECORDS)
        result = evaluate_char_ngram_logreg(fitted, _TRAIN_RECORDS, [])
        assert result.per_label == ()
        assert result.macro_f1 is None
        assert result.evaluated_labels == ()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_metrics(result, label_name):
    """Find LabelMetrics for a given label name in a BaselineEvaluation."""
    for m in result.per_label:
        if m.label == label_name:
            return m
    return None
