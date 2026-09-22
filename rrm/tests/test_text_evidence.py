"""Tests for RRM deterministic PII detection and text-evidence extraction.

Uses pytest.  All tests use only the Python standard library (no external
dependencies beyond pytest itself).

Architectural note:
    These tests verify evidence-generation correctness.  They do NOT
    validate moderation thresholds, privacy policies, or Trust decisions,
    which belong to Layer 4.

    PII pattern match != confirmed harmful disclosure.
    For example, admissions@college.example may match the EMAIL pattern,
    but whether it is personal/private PII is a later policy/context decision.
"""

from __future__ import annotations

import dataclasses
import pytest

from rrm.pii_detection import PIIMatch, detect_emails, detect_phones, detect_pii, detect_urls
from rrm.text_evidence import TextEvidence, extract_text_evidence


# ---------------------------------------------------------------------------
# 1. Email detection
# ---------------------------------------------------------------------------


class TestEmailDetection:
    def test_simple_email_detected(self):
        matches = detect_emails("contact me at user@example.com")
        assert len(matches) == 1
        assert matches[0].kind == "EMAIL"
        assert matches[0].text == "user@example.com"

    def test_multiple_emails(self):
        text = "a@b.com and c@d.org"
        matches = detect_emails(text)
        assert len(matches) == 2
        assert matches[0].text == "a@b.com"
        assert matches[1].text == "c@d.org"

    def test_no_email_returns_empty(self):
        matches = detect_emails("hello world")
        assert matches == ()

    def test_email_with_plus(self):
        matches = detect_emails("first.last+tag@sub.domain.com")
        assert len(matches) == 1
        assert matches[0].text == "first.last+tag@sub.domain.com"


# ---------------------------------------------------------------------------
# 2. Email original offsets
# ---------------------------------------------------------------------------


class TestEmailOffsets:
    def test_email_start_offset(self):
        text = "xx user@example.com yy"
        matches = detect_emails(text)
        assert len(matches) == 1
        assert text[matches[0].start:matches[0].end] == "user@example.com"
        assert matches[0].start == 3

    def test_email_end_offset(self):
        text = "user@example.com end"
        matches = detect_emails(text)
        assert len(matches) == 1
        assert text[matches[0].start:matches[0].end] == "user@example.com"
        assert matches[0].end == 16


# ---------------------------------------------------------------------------
# 3. URL detection
# ---------------------------------------------------------------------------


class TestURLDetection:
    def test_http_url(self):
        matches = detect_urls("visit http://example.com for more")
        assert len(matches) == 1
        assert matches[0].kind == "URL"
        assert matches[0].text == "http://example.com"

    def test_https_url(self):
        matches = detect_urls("see https://secure.example.org/path")
        assert len(matches) == 1
        assert matches[0].text == "https://secure.example.org/path"

    def test_no_url(self):
        matches = detect_urls("just some text here")
        assert matches == ()

    def test_url_with_query(self):
        matches = detect_urls("link https://example.com?a=1&b=2 end")
        assert len(matches) == 1


# ---------------------------------------------------------------------------
# 4. URL offsets
# ---------------------------------------------------------------------------


class TestURLOffsets:
    def test_url_start_offset(self):
        text = "xx https://example.com yy"
        matches = detect_urls(text)
        assert len(matches) == 1
        assert text[matches[0].start:matches[0].end] == "https://example.com"
        assert matches[0].start == 3

    def test_url_end_offset(self):
        text = "see https://example.com now"
        matches = detect_urls(text)
        assert text[matches[0].start:matches[0].end] == "https://example.com"
        # end should be right after the URL
        assert text[matches[0].end - 1] == "m"


# ---------------------------------------------------------------------------
# 5. Simple Indian-style phone detection
# ---------------------------------------------------------------------------


class TestIndianPhoneDetection:
    def test_10_digit_indian_phone(self):
        text = "call me at 9876543210"
        matches = detect_phones(text)
        assert len(matches) == 1
        assert matches[0].kind == "PHONE"
        assert matches[0].text == "9876543210"

    def test_10_digit_starting_with_6(self):
        text = "my number is 6123456789"
        matches = detect_phones(text)
        assert len(matches) == 1

    def test_11_digit_phone(self):
        text = "18001234567 is toll-free"
        matches = detect_phones(text)
        assert len(matches) == 1


# ---------------------------------------------------------------------------
# 6. Formatted phone detection
# ---------------------------------------------------------------------------


class TestFormattedPhoneDetection:
    def test_dash_separated(self):
        text = "call 98765-43210"
        matches = detect_phones(text)
        assert len(matches) == 1
        assert matches[0].kind == "PHONE"

    def test_space_separated(self):
        text = "number 98765 43210"
        matches = detect_phones(text)
        assert len(matches) == 1

    def test_parenthesized_area_code(self):
        text = "(022) 12345678"
        matches = detect_phones(text)
        assert len(matches) == 1

    def test_international_prefix_plus(self):
        text = "+91 98765 43210"
        matches = detect_phones(text)
        assert len(matches) == 1

    def test_international_prefix_00(self):
        text = "00 91 98765 43210"
        matches = detect_phones(text)
        assert len(matches) == 1


# ---------------------------------------------------------------------------
# 7-9. Short numbers, years, empty text
# ---------------------------------------------------------------------------


class TestShortNumbersNotPhones:
    def test_year_not_phone(self):
        matches = detect_phones("admitted in 2026")
        assert len(matches) == 0

    def test_three_digit_not_phone(self):
        matches = detect_phones("room 404")
        assert len(matches) == 0

    def test_single_digit_not_phone(self):
        matches = detect_phones("semester 4")
        assert len(matches) == 0

    def test_123_not_phone(self):
        matches = detect_phones("123")
        assert len(matches) == 0

    def test_empty_text(self):
        assert detect_emails("") == ()
        assert detect_phones("") == ()
        assert detect_urls("") == ()
        assert detect_pii("") == ()


# ---------------------------------------------------------------------------
# 10. Deterministic ordering
# ---------------------------------------------------------------------------


class TestDeterministicOrdering:
    def test_ordering_by_start_then_end_then_kind(self):
        text = "email a@b.com phone 9876543210 url https://x.com"
        matches = detect_pii(text)
        assert len(matches) == 3
        # Sorted by start position
        assert matches[0].start < matches[1].start < matches[2].start

    def test_repeated_calls_same_order(self):
        text = "see https://a.com and call 9876543210 or email x@y.com"
        r1 = detect_pii(text)
        r2 = detect_pii(text)
        assert r1 == r2


# ---------------------------------------------------------------------------
# 11. Multiple evidence types
# ---------------------------------------------------------------------------


class TestMultipleEvidenceTypes:
    def test_all_three_types_in_one_text(self):
        text = "email: admin@test.com | phone: 9876543210 | url: https://test.com"
        matches = detect_pii(text)
        kinds = {m.kind for m in matches}
        assert kinds == {"EMAIL", "PHONE", "URL"}


# ---------------------------------------------------------------------------
# 12. Immutable PIIMatch
# ---------------------------------------------------------------------------


class TestImmutablePIIMatch:
    def test_pii_match_is_frozen(self):
        m = PIIMatch(kind="EMAIL", start=0, end=5, text="a@b.c")
        with pytest.raises(dataclasses.FrozenInstanceError):
            m.kind = "URL"  # type: ignore[misc]

    def test_pii_match_fields_accessible(self):
        m = PIIMatch(kind="URL", start=3, end=20, text="https://example.com")
        assert m.kind == "URL"
        assert m.start == 3
        assert m.end == 20


# ---------------------------------------------------------------------------
# 13. Text input type validation
# ---------------------------------------------------------------------------


class TestTypeValidation:
    def test_none_rejected(self):
        with pytest.raises(TypeError, match="expects str"):
            detect_emails(None)

    def test_int_rejected(self):
        with pytest.raises(TypeError, match="expects str"):
            detect_phones(12345)

    def test_list_rejected(self):
        with pytest.raises(TypeError, match="expects str"):
            detect_urls(["https://test.com"])

    def test_detect_pii_none_rejected(self):
        with pytest.raises(TypeError, match="expects str"):
            detect_pii(None)

    def test_extract_evidence_none_rejected(self):
        with pytest.raises(TypeError, match="expects str"):
            extract_text_evidence(None)


# ---------------------------------------------------------------------------
# 14-20. Text evidence basic counts
# ---------------------------------------------------------------------------


class TestTextEvidenceBasicCounts:
    def test_character_count(self):
        e = extract_text_evidence("hello world")
        assert e.character_count == 11

    def test_non_whitespace_count(self):
        e = extract_text_evidence("a b c")
        assert e.non_whitespace_character_count == 3

    def test_word_count(self):
        e = extract_text_evidence("one two three")
        assert e.word_count == 3

    def test_word_count_ignores_empty_tokens(self):
        e = extract_text_evidence("a  b")  # double space
        assert e.word_count == 2

    def test_digit_count(self):
        e = extract_text_evidence("year 2024 batch")
        assert e.digit_count == 4

    def test_uppercase_count(self):
        e = extract_text_evidence("Hello World")
        assert e.uppercase_character_count == 2

    def test_exclamation_count(self):
        e = extract_text_evidence("wow!!! amazing")
        assert e.exclamation_count == 3

    def test_question_count(self):
        e = extract_text_evidence("is it good? really?")
        assert e.question_count == 2


# ---------------------------------------------------------------------------
# 21-24. Repeated character runs
# ---------------------------------------------------------------------------


class TestRepeatedRuns:
    def test_no_repeated_run(self):
        e = extract_text_evidence("good college")
        assert e.repeated_character_run_count == 0
        assert e.max_character_run_length == 0

    def test_single_repeated_run(self):
        e = extract_text_evidence("goood")
        assert e.repeated_character_run_count == 1
        assert e.max_character_run_length == 3

    def test_multiple_repeated_runs(self):
        e = extract_text_evidence("soooo coool")
        assert e.repeated_character_run_count == 2
        assert e.max_character_run_length == 4

    def test_whitespace_runs_ignored(self):
        # "a  b" has a whitespace run of length 2, not >= 3
        e = extract_text_evidence("a  b")
        assert e.repeated_character_run_count == 0

    def test_case_sensitive_repeated_run(self):
        # "AAa": 'A' x2 then 'a' -- run of 'A' is length 2 (< 3),
        # so no qualifying repeated run under case-sensitive matching.
        e = extract_text_evidence("AAa")
        assert e.repeated_character_run_count == 0
        assert e.max_character_run_length == 0


# ---------------------------------------------------------------------------
# 25-27. Hinglish / Roman Hindi / noisy examples
# ---------------------------------------------------------------------------


class TestLanguageExamples:
    def test_hinglish_text_evidence(self):
        text = "hostel ka food bahut achha hai!!!"
        e = extract_text_evidence(text)
        # split() on whitespace: ["hostel","ka","food","bahut","achha","hai!!!"]
        assert e.word_count == 6
        assert e.exclamation_count == 3

    def test_roman_hindi_text_evidence(self):
        text = "padhai acchi hai par faculty bekar hai"
        e = extract_text_evidence(text)
        # split() on whitespace: 7 tokens
        assert e.word_count == 7
        assert e.digit_count == 0

    def test_noisy_review_with_repeats(self):
        text = "the faculty is goooood and the campus is amaaaazing"
        e = extract_text_evidence(text)
        assert e.repeated_character_run_count == 2
        assert e.max_character_run_length == 5


# ---------------------------------------------------------------------------
# 28-30. PII counts and immutability
# ---------------------------------------------------------------------------


class TestPIICountsInEvidence:
    def test_email_count_in_evidence(self):
        e = extract_text_evidence("contact: admin@test.com")
        assert e.email_count == 1
        assert e.phone_count == 0
        assert e.url_count == 0

    def test_phone_count_in_evidence(self):
        e = extract_text_evidence("call 9876543210 now")
        assert e.phone_count == 1

    def test_url_count_in_evidence(self):
        e = extract_text_evidence("see https://example.com")
        assert e.url_count == 1

    def test_text_evidence_is_frozen(self):
        e = extract_text_evidence("hello")
        with pytest.raises(dataclasses.FrozenInstanceError):
            e.word_count = 99  # type: ignore[misc]

    def test_determinism(self):
        text = "email: a@b.com phone: 9876543210 url: https://x.com"
        e1 = extract_text_evidence(text)
        e2 = extract_text_evidence(text)
        assert e1 == e2

    def test_no_pii_in_simple_text(self):
        e = extract_text_evidence("the college has good wifi")
        assert e.email_count == 0
        assert e.phone_count == 0
        assert e.url_count == 0
