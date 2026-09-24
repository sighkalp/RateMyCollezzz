"""Tests for rrm/labels.py — the neutral canonical label contract."""

from __future__ import annotations

import pytest

from rrm.labels import (
    NUM_PRIMARY_LABELS,
    PRIMARY_LABELS,
    UNKNOWN_LABEL,
    _VALID_LABEL_VALUES,
)


class TestPrimaryLabels:
    """Tests for the PRIMARY_LABELS constant."""

    def test_is_tuple(self):
        assert isinstance(PRIMARY_LABELS, tuple)

    def test_length(self):
        assert len(PRIMARY_LABELS) == 6

    def test_canonical_order(self):
        assert PRIMARY_LABELS == (
            "spam",
            "deception",
            "toxicity",
            "advertising",
            "off_topic",
            "pii",
        )

    def test_immutable(self):
        """PRIMARY_LABELS must be a tuple (immutable), not a list."""
        with pytest.raises(AttributeError):
            PRIMARY_LABELS.append("extra")

    def test_no_duplicates(self):
        assert len(set(PRIMARY_LABELS)) == len(PRIMARY_LABELS)

    def test_all_strings(self):
        for label in PRIMARY_LABELS:
            assert isinstance(label, str)
            assert len(label) > 0


class TestUnknownLabel:
    """Tests for the UNKNOWN_LABEL constant."""

    def test_value(self):
        assert UNKNOWN_LABEL == -1

    def test_is_int(self):
        assert isinstance(UNKNOWN_LABEL, int)

    def test_not_zero(self):
        assert UNKNOWN_LABEL != 0

    def test_not_one(self):
        assert UNKNOWN_LABEL != 1

    def test_negative(self):
        assert UNKNOWN_LABEL < 0


class TestNumPrimaryLabels:
    """Tests for NUM_PRIMARY_LABELS."""

    def test_matches_length(self):
        assert NUM_PRIMARY_LABELS == len(PRIMARY_LABELS)

    def test_value(self):
        assert NUM_PRIMARY_LABELS == 6

    def test_is_int(self):
        assert isinstance(NUM_PRIMARY_LABELS, int)


class TestValidLabelValues:
    """Tests for _VALID_LABEL_VALUES."""

    def test_contains_zero(self):
        assert 0 in _VALID_LABEL_VALUES

    def test_contains_one(self):
        assert 1 in _VALID_LABEL_VALUES

    def test_contains_unknown(self):
        assert UNKNOWN_LABEL in _VALID_LABEL_VALUES

    def test_no_other_values(self):
        assert len(_VALID_LABEL_VALUES) == 3


class TestCrossModuleImport:
    """Verify label constants are importable and consistent."""

    def test_baseline_imports_consistent(self):
        """Baseline modules import from rrm.labels."""
        from rrm.baseline_transformer_common import PRIMARY_LABELS as PL
        assert PL is PRIMARY_LABELS

    def test_baseline_unknown_consistent(self):
        from rrm.baseline_transformer_common import UNKNOWN_LABEL as UL
        assert UL is UNKNOWN_LABEL

    def test_tfidf_imports_consistent(self):
        from rrm.baseline_tfidf_lr import PRIMARY_LABELS as PL
        assert PL is PRIMARY_LABELS

    def test_char_ngram_imports_consistent(self):
        from rrm.baseline_char_ngram_lr import PRIMARY_LABELS as PL
        assert PL is PRIMARY_LABELS
