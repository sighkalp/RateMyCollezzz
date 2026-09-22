"""Tests for RRM deterministic pre-checks: text normalization and similarity.

Uses pytest.  All tests use only the Python standard library (no external
dependencies beyond pytest itself).

Architectural note:
    These tests verify evidence-generation correctness.  They do NOT
    validate moderation thresholds or Trust decisions, which belong to
    Layer 4.
"""

from __future__ import annotations

import pytest

from rrm.text_normalization import exact_fingerprint, normalize_for_similarity
from rrm.similarity import (
    SimilarityMatch,
    character_ngrams,
    find_best_match,
    jaccard_similarity,
)


# ---------------------------------------------------------------------------
# 1. Whitespace normalization
# ---------------------------------------------------------------------------


class TestWhitespaceNormalization:
    def test_leading_whitespace_removed(self):
        assert normalize_for_similarity("  hello") == "hello"

    def test_trailing_whitespace_removed(self):
        assert normalize_for_similarity("hello  ") == "hello"

    def test_consecutive_spaces_collapsed(self):
        assert normalize_for_similarity("hello   world") == "hello world"

    def test_tabs_and_newlines_collapsed(self):
        assert normalize_for_similarity("hello\tworld\nfoo") == "hello world foo"

    def test_multiple_whitespace_types(self):
        result = normalize_for_similarity("  College   Is GOOD  ")
        assert result == "college is good"


# ---------------------------------------------------------------------------
# 2. Unicode NFKC behavior
# ---------------------------------------------------------------------------


class TestUnicodeNFKC:
    def test_fullwidth_latin_normalized(self):
        # U+FF21 is fullwidth 'A'; NFKC decomposes to 'A'
        text = "Ａ"  # Ａ
        result = normalize_for_similarity(text)
        assert result == "a"  # after casefold

    def test_superscript_digits_normalized(self):
        # U+00B2 is superscript 2; NFKC maps to '2'
        text = "x\00b2"  # x²
        result = normalize_for_similarity(text)
        assert "2" in result

    def test_combining_characters_composed(self):
        # e + combining acute -> é after NFKC
        text = "é"  # e + ◌́
        result = normalize_for_similarity(text)
        assert result == "é"  # é


# ---------------------------------------------------------------------------
# 3. Case folding
# ---------------------------------------------------------------------------


class TestCaseFolding:
    def test_uppercase_to_lowercase(self):
        assert normalize_for_similarity("HELLO") == "hello"

    def test_mixed_case(self):
        assert normalize_for_similarity("Hello World") == "hello world"

    def test_esset_ss(self):
        # German ß -> ss in casefold
        assert normalize_for_similarity("groß") == "gross"

    def test_preserves_case_after_fold(self):
        assert normalize_for_similarity("ABC") == "abc"


# ---------------------------------------------------------------------------
# 4. Punctuation preserved
# ---------------------------------------------------------------------------


class TestPunctuationPreserved:
    def test_exclamation_preserved(self):
        assert normalize_for_similarity("BEST!!!") == "best!!!"

    def test_question_mark_preserved(self):
        result = normalize_for_similarity("Is this good?")
        assert "?" in result

    def test_period_preserved(self):
        result = normalize_for_similarity("Good.")
        assert "." in result

    def test_comma_preserved(self):
        result = normalize_for_similarity("Good, bad")
        assert "," in result

    def test_punctuation_variation_not_exact_duplicate(self):
        # Adding punctuation should change the normalized output
        text_a = normalize_for_similarity("best")
        text_b = normalize_for_similarity("best!")
        assert text_a != text_b


# ---------------------------------------------------------------------------
# 5. Digits preserved
# ---------------------------------------------------------------------------


class TestDigitsPreserved:
    def test_digits_unchanged(self):
        assert "123" in normalize_for_similarity("room 123")

    def test_digits_not_removed(self):
        result = normalize_for_similarity("year 2024")
        assert "2024" in result

    def test_mixed_digits_and_letters(self):
        result = normalize_for_similarity("3rd year")
        assert "3rd" in result


# ---------------------------------------------------------------------------
# 6. Fingerprint determinism
# ---------------------------------------------------------------------------


class TestFingerprintDeterminism:
    def test_same_input_same_fingerprint(self):
        fp1 = exact_fingerprint("hello world")
        fp2 = exact_fingerprint("hello world")
        assert fp1 == fp2

    def test_different_texts_different_fingerprints(self):
        fp1 = exact_fingerprint("hello")
        fp2 = exact_fingerprint("world")
        assert fp1 != fp2

    def test_fingerprint_is_hex(self):
        fp = exact_fingerprint("test")
        assert all(c in "0123456789abcdef" for c in fp)

    def test_fingerprint_length(self):
        fp = exact_fingerprint("test")
        assert len(fp) == 64  # SHA-256 hex digest length


# ---------------------------------------------------------------------------
# 7. Equivalent normalized texts have same fingerprint
# ---------------------------------------------------------------------------


class TestEquivalentFingerprints:
    def test_whitespace_variations_same_fingerprint(self):
        fp1 = exact_fingerprint("hello world")
        fp2 = exact_fingerprint("  hello   world  ")
        assert fp1 == fp2

    def test_case_variations_same_fingerprint(self):
        fp1 = exact_fingerprint("Hello World")
        fp2 = exact_fingerprint("hello world")
        assert fp1 == fp2

    def test_combined_whitespace_and_case(self):
        fp1 = exact_fingerprint("  College Is GOOD  ")
        fp2 = exact_fingerprint("college is good")
        assert fp1 == fp2


# ---------------------------------------------------------------------------
# 8. Punctuation-changed text is not exact duplicate
# ---------------------------------------------------------------------------


class TestPunctuationNotExactDuplicate:
    def test_punctuation_changes_fingerprint(self):
        fp1 = exact_fingerprint("best")
        fp2 = exact_fingerprint("best!")
        assert fp1 != fp2

    def test_space_changes_fingerprint(self):
        fp1 = exact_fingerprint("hello world")
        fp2 = exact_fingerprint("helloworld")
        assert fp1 != fp2


# ---------------------------------------------------------------------------
# 9-12. Jaccard similarity basics
# ---------------------------------------------------------------------------


class TestJaccardBasics:
    def test_identical_text_similarity(self):
        score = jaccard_similarity("hello world", "hello world")
        assert score == 1.0

    def test_both_empty_similarity(self):
        score = jaccard_similarity("", "")
        assert score == 1.0

    def test_empty_nonempty_similarity(self):
        score = jaccard_similarity("", "hello")
        assert score == 0.0

    def test_nonempty_empty_similarity(self):
        score = jaccard_similarity("hello", "")
        assert score == 0.0

    def test_score_in_valid_range(self):
        score = jaccard_similarity("hello", "world")
        assert 0.0 <= score <= 1.0

    def test_unrelated_texts_lower_than_near_copy(self):
        unrelated = jaccard_similarity(
            "the hostel food is terrible and the wifi never works",
            "the library has great study spaces and comfortable seating",
        )
        near_copy = jaccard_similarity(
            "the hostel food is terrible and the wifi never works",
            "the hostel food is terrible and the wifi does not work",
        )
        assert near_copy > unrelated


# ---------------------------------------------------------------------------
# 13. Small spelling variation creates non-zero similarity
# ---------------------------------------------------------------------------


class TestSpellingVariation:
    def test_minor_edit_nonzero_similarity(self):
        score = jaccard_similarity(
            "beautiful campus",
            "beautifull campus",
        )
        assert score > 0.0

    def test_minor_edit_not_identical(self):
        score = jaccard_similarity(
            "beautiful campus",
            "beautifull campus",
        )
        assert score < 1.0


# ---------------------------------------------------------------------------
# 14-16. Hinglish and Roman Hindi examples
# ---------------------------------------------------------------------------


class TestHinglishRomanHindi:
    def test_hinglish_same_phrase(self):
        text_a = "college ka food bahut achha hai"
        text_b = "college ka food bahut achha hai"
        assert jaccard_similarity(text_a, text_b) == 1.0

    def test_roman_hindi_same_phrase(self):
        text_a = "padhai acchi hai par faculty bekar hai"
        text_b = "padhai acchi hai par faculty bekar hai"
        assert jaccard_similarity(text_a, text_b) == 1.0

    def test_hinglish_slight_variation(self):
        text_a = "hostel me wifi nahi hai"
        text_b = "hostel mein wifi nahi hai"
        score = jaccard_similarity(text_a, text_b)
        assert 0.0 < score < 1.0

    def test_repeated_characters(self):
        # "goooood" with repeated 'o'
        text_a = "the faculty is goooood"
        text_b = "the faculty is good"
        score = jaccard_similarity(text_a, text_b)
        assert score > 0.0


# ---------------------------------------------------------------------------
# 17. Best-match selection
# ---------------------------------------------------------------------------


class TestBestMatch:
    def test_best_match_selected(self):
        candidates = [
            ("r1", "the hostel is clean and nice"),
            ("r2", "the hostel is clean and good"),
            ("r3", "completely unrelated text about astronomy"),
        ]
        result = find_best_match(
            "the hostel is clean and nice", candidates
        )
        assert result is not None
        assert result.review_id == "r1"
        assert result.exact_duplicate is True
        assert result.score == 1.0

    def test_near_match_selected_over_unrelated(self):
        candidates = [
            ("a", "the wifi is terrible"),
            ("b", "the wifi is great"),
            ("c", "astronomy stars planets galaxies"),
        ]
        result = find_best_match(
            "the wifi is bad", candidates
        )
        assert result is not None
        assert result.review_id in ("a", "b")
        assert result.review_id != "c"


# ---------------------------------------------------------------------------
# 18. exclude_review_id behavior
# ---------------------------------------------------------------------------


class TestExcludeReviewId:
    def test_excluded_review_not_returned(self):
        candidates = [
            ("r1", "hello world"),
            ("r2", "hello world"),
        ]
        result = find_best_match(
            "hello world", candidates, exclude_review_id="r1"
        )
        assert result is not None
        assert result.review_id == "r2"

    def test_all_excluded_returns_none(self):
        candidates = [
            ("r1", "hello world"),
        ]
        result = find_best_match(
            "hello world", candidates, exclude_review_id="r1"
        )
        assert result is None


# ---------------------------------------------------------------------------
# 19. Deterministic tie breaking
# ---------------------------------------------------------------------------


class TestDeterministicTieBreaking:
    def test_tie_broken_by_review_id(self):
        # Two candidates with identical text (guaranteed same score),
        # so the tie-break rule picks the lexicographically smallest review_id.
        candidates = [
            ("bbb", "identical text here"),
            ("aaa", "identical text here"),
        ]
        result = find_best_match(
            "some other text", candidates
        )
        assert result is not None
        assert result.review_id == "aaa"

    def test_same_score_repeated(self):
        """Same result on repeated calls confirms determinism."""
        candidates = [
            ("z1", "text one"),
            ("a1", "text two"),
        ]
        r1 = find_best_match("text three", candidates)
        r2 = find_best_match("text three", candidates)
        assert r1 is not None and r2 is not None
        assert r1.review_id == r2.review_id


# ---------------------------------------------------------------------------
# 20. exact_duplicate field correctness
# ---------------------------------------------------------------------------


class TestExactDuplicateField:
    def test_exact_match_sets_flag(self):
        candidates = [
            ("r1", "hello world test"),
        ]
        result = find_best_match("hello world test", candidates)
        assert result is not None
        assert result.exact_duplicate is True

    def test_near_but_not_exact_clears_flag(self):
        candidates = [
            ("r1", "hello world test!"),
        ]
        result = find_best_match("hello world test", candidates)
        assert result is not None
        assert result.exact_duplicate is False

    def test_no_candidates_returns_none(self):
        result = find_best_match("hello", [])
        assert result is None


# ---------------------------------------------------------------------------
# 21. Invalid n rejected
# ---------------------------------------------------------------------------


class TestInvalidN:
    def test_n_zero_raises(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            character_ngrams("hello", 0)

    def test_negative_n_raises(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            character_ngrams("hello", -1)

    def test_jaccard_n_zero_raises(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            jaccard_similarity("hello", "world", 0)

    def test_find_best_match_n_zero_raises(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            find_best_match("hello", [("r1", "world")], n=0)


# ---------------------------------------------------------------------------
# 22. Non-string normalization rejected
# ---------------------------------------------------------------------------


class TestNonStringRejected:
    def test_none_raises_type_error(self):
        with pytest.raises(TypeError, match="expects str"):
            normalize_for_similarity(None)

    def test_int_raises_type_error(self):
        with pytest.raises(TypeError, match="expects str"):
            normalize_for_similarity(123)

    def test_list_raises_type_error(self):
        with pytest.raises(TypeError, match="expects str"):
            normalize_for_similarity(["hello"])

    def test_character_ngrams_non_string_raises(self):
        with pytest.raises(TypeError, match="expects str"):
            character_ngrams(42, 4)


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_short_string_ngrams(self):
        # String shorter than n: should return set with full string
        result = character_ngrams("hi", n=5)
        assert result == {"hi"}

    def test_single_char_ngrams(self):
        result = character_ngrams("a", n=1)
        assert result == {"a"}

    def test_return_type_is_set(self):
        # Approved API: character_ngrams returns set[str], not frozenset
        result = character_ngrams("hello", 4)
        assert isinstance(result, set)

    def test_unicode_preserved_after_normalization(self):
        # Devanagari text should be preserved (not transliterated)
        text = "नमस्ते"  # नमस्ते
        result = normalize_for_similarity(text)
        # Should still contain Devanagari characters
        assert any(ord(c) > 0x0900 for c in result)

    def test_hinglish_normalized_same(self):
        # "mein" and "MEIN" should normalize identically
        n1 = normalize_for_similarity("hostel mein accha hai")
        n2 = normalize_for_similarity("HOSTEL MEIN ACCHA HAI")
        assert n1 == n2
