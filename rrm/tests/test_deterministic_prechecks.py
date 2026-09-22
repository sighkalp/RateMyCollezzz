"""Tests for RRM 3.3C — Unified Deterministic Precheck Output.

Tests verify that the orchestration layer correctly delegates to existing
modules and produces a coherent immutable result.

Architectural note:
    These tests verify evidence-aggregation correctness.  They do NOT
    validate moderation thresholds or Trust decisions, which belong to
    Layer 4.

    exact_duplicate != spam automatically.
    high similarity != malicious behavior automatically.
    PII pattern != confirmed privacy violation.
"""

from __future__ import annotations

import dataclasses

import pytest

from rrm.deterministic_prechecks import DeterministicPrecheckResult, run_deterministic_prechecks
from rrm.pii_detection import PIIMatch, detect_pii
from rrm.similarity import SimilarityMatch
from rrm.text_evidence import TextEvidence, extract_text_evidence
from rrm.text_normalization import exact_fingerprint


# ---------------------------------------------------------------------------
# 1-5. Basic result structure
# ---------------------------------------------------------------------------


class TestBasicResult:
    def test_result_has_expected_fields(self):
        result = run_deterministic_prechecks("r1", "hello world")
        assert result.review_id == "r1"
        assert isinstance(result.fingerprint, str)
        assert isinstance(result.text_evidence, TextEvidence)
        assert isinstance(result.pii_matches, tuple)
        assert result.best_similarity_match is None

    def test_result_is_frozen(self):
        result = run_deterministic_prechecks("r1", "hello")
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.review_id = "r2"  # type: ignore[misc]

    def test_fingerprint_matches_exact_fingerprint(self):
        text = "hello world test"
        result = run_deterministic_prechecks("r1", text)
        assert result.fingerprint == exact_fingerprint(text)

    def test_text_evidence_matches_extract_text_evidence(self):
        text = "hello world test"
        result = run_deterministic_prechecks("r1", text)
        assert result.text_evidence == extract_text_evidence(text)

    def test_pii_matches_match_detect_pii(self):
        text = "email: admin@test.com call 9876543210"
        result = run_deterministic_prechecks("r1", text)
        assert result.pii_matches == detect_pii(text)


# ---------------------------------------------------------------------------
# 6-9. Similarity candidates
# ---------------------------------------------------------------------------


class TestSimilarityCandidates:
    def test_no_candidates_returns_none_match(self):
        result = run_deterministic_prechecks("r1", "hello world")
        assert result.best_similarity_match is None

    def test_exact_duplicate_candidate(self):
        candidates = [("r2", "hello world")]
        result = run_deterministic_prechecks("r1", "hello world", candidates)
        assert result.best_similarity_match is not None
        assert result.best_similarity_match.exact_duplicate is True
        assert result.best_similarity_match.score == 1.0

    def test_near_duplicate_candidate(self):
        candidates = [("r2", "the hostel is clean and good")]
        result = run_deterministic_prechecks(
            "r1", "the hostel is clean and nice", candidates
        )
        assert result.best_similarity_match is not None
        assert result.best_similarity_match.exact_duplicate is False
        assert result.best_similarity_match.score > 0.0

    def test_unrelated_candidate(self):
        candidates = [("r2", "astronomy stars planets galaxies")]
        result = run_deterministic_prechecks(
            "r1", "the hostel food is terrible", candidates
        )
        assert result.best_similarity_match is not None
        # Score should be low for completely unrelated text
        assert result.best_similarity_match.score < 0.5


# ---------------------------------------------------------------------------
# 10-11. Exclusion and tie-breaking
# ---------------------------------------------------------------------------


class TestExclusionAndTieBreaking:
    def test_current_review_excluded(self):
        candidates = [
            ("r1", "hello world"),
            ("r2", "hello world"),
        ]
        result = run_deterministic_prechecks(
            "r1", "hello world", candidates
        )
        assert result.best_similarity_match is not None
        assert result.best_similarity_match.review_id == "r2"

    def test_deterministic_tie_behavior_preserved(self):
        candidates = [
            ("bbb", "identical text here"),
            ("aaa", "identical text here"),
        ]
        result = run_deterministic_prechecks(
            "r1", "some other text", candidates
        )
        assert result.best_similarity_match is not None
        assert result.best_similarity_match.review_id == "aaa"


# ---------------------------------------------------------------------------
# 12. Generator candidates
# ---------------------------------------------------------------------------


class TestGeneratorCandidates:
    def test_generator_candidates_supported(self):
        def gen():
            yield ("r2", "hello world test")
            yield ("r3", "unrelated content here")

        result = run_deterministic_prechecks("r1", "hello world test", gen())
        assert result.best_similarity_match is not None
        assert result.best_similarity_match.review_id == "r2"
        assert result.best_similarity_match.exact_duplicate is True


# ---------------------------------------------------------------------------
# 13-16. Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_review_id_rejected(self):
        with pytest.raises(ValueError, match="review_id"):
            run_deterministic_prechecks("", "hello")

    def test_whitespace_only_review_id_rejected(self):
        with pytest.raises(ValueError, match="review_id"):
            run_deterministic_prechecks("   ", "hello")

    def test_non_string_review_id_rejected(self):
        with pytest.raises(TypeError, match="review_id"):
            run_deterministic_prechecks(123, "hello")  # type: ignore[arg-type]

    def test_non_string_review_text_rejected(self):
        with pytest.raises(TypeError, match="review_text"):
            run_deterministic_prechecks("r1", 456)  # type: ignore[arg-type]

    def test_n_zero_rejected(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            run_deterministic_prechecks("r1", "hello", n=0)


# ---------------------------------------------------------------------------
# 17-19. Candidate validation
# ---------------------------------------------------------------------------


class TestCandidateValidation:
    def test_malformed_candidate_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            run_deterministic_prechecks(
                "r1", "hello", [("r2",)]  # type: ignore[list-item]
            )

    def test_candidate_non_string_id_rejected(self):
        with pytest.raises(TypeError, match="Candidate review_id must be str"):
            run_deterministic_prechecks(
                "r1", "hello", [(123, "text")]  # type: ignore[list-item]
            )

    def test_candidate_non_string_text_rejected(self):
        with pytest.raises(TypeError, match="Candidate review_text must be str"):
            run_deterministic_prechecks(
                "r1", "hello", [("r2", 456)]  # type: ignore[list-item]
            )


# ---------------------------------------------------------------------------
# 20-22. PII evidence consistency
# ---------------------------------------------------------------------------


class TestPIIEvidenceConsistency:
    def test_email_evidence_consistency(self):
        text = "contact: admin@test.com and user@example.org"
        result = run_deterministic_prechecks("r1", text)
        assert result.text_evidence.email_count == 2
        assert len(result.pii_matches) == 2
        email_matches = [m for m in result.pii_matches if m.kind == "EMAIL"]
        assert len(email_matches) == 2

    def test_phone_evidence_consistency(self):
        text = "call me at 9876543210 or 9123456789"
        result = run_deterministic_prechecks("r1", text)
        assert result.text_evidence.phone_count == 2
        phone_matches = [m for m in result.pii_matches if m.kind == "PHONE"]
        assert len(phone_matches) == 2

    def test_url_evidence_consistency(self):
        text = "see https://a.com and http://b.org"
        result = run_deterministic_prechecks("r1", text)
        assert result.text_evidence.url_count == 2
        url_matches = [m for m in result.pii_matches if m.kind == "URL"]
        assert len(url_matches) == 2


# ---------------------------------------------------------------------------
# 23-25. Language and noisy examples
# ---------------------------------------------------------------------------


class TestLanguageAndNoisyExamples:
    def test_hinglish_example(self):
        text = "hostel ka food bahut achha hai!!! call 9876543210"
        result = run_deterministic_prechecks("r1", text)
        # split(): hostel, ka, food, bahut, achha, hai!!!, call, 9876543210
        assert result.text_evidence.word_count == 8
        assert result.text_evidence.exclamation_count == 3
        assert result.text_evidence.phone_count == 1
        assert result.pii_matches[0].kind == "PHONE"

    def test_roman_hindi_example(self):
        text = "padhai acchi hai par faculty bekar hai"
        result = run_deterministic_prechecks("r1", text)
        assert result.text_evidence.word_count == 7
        assert result.text_evidence.digit_count == 0

    def test_noisy_repeated_characters(self):
        text = "the faculty is goooood and campus is amaaaazing"
        result = run_deterministic_prechecks("r1", text)
        assert result.text_evidence.repeated_character_run_count == 2
        assert result.text_evidence.max_character_run_length == 5


# ---------------------------------------------------------------------------
# 26-27. Determinism and no moderation fields
# ---------------------------------------------------------------------------


class TestDeterminismAndNoModeration:
    def test_repeated_calls_produce_equal_result(self):
        text = "review text here"
        candidates = [("r2", "other review")]
        r1 = run_deterministic_prechecks("r1", text, candidates)
        r2 = run_deterministic_prechecks("r1", text, candidates)
        assert r1 == r2

    def test_no_moderation_fields_exist(self):
        """The result must not contain spam, toxicity, or risk fields."""
        field_names = {f.name for f in dataclasses.fields(DeterministicPrecheckResult)}
        forbidden = {
            "spam_score", "toxicity_score", "advertising_score",
            "risk_score", "moderation_action", "pii_label",
            "off_topic_score", "deception_score",
        }
        assert forbidden.isdisjoint(field_names), (
            f"Forbidden moderation fields found: {forbidden & field_names}"
        )
