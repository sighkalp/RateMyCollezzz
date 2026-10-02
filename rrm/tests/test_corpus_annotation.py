"""Tests for rrm.corpus.annotation module.

Public API tested:
    AnnotationSubmissionStore
    ControlledProtocolTruthStore
    DispositionRegister
    ReannotationRegister
    compute_eight_dimension_disagreement
    needs_adjudication
    adjudicator_decision
"""

from __future__ import annotations

import pytest

from rrm.corpus.annotation import (
    AnnotationSubmissionStore,
    ControlledProtocolTruthStore,
    DispositionRegister,
    ReannotationRegister,
    adjudicator_decision,
    compute_eight_dimension_disagreement,
    needs_adjudication,
)
from rrm.corpus.models import (
    AnnotationStatus,
    AnnotationSubmission,
    CollegeCategory,
    ControlledProtocolTruth,
    Disposition,
    LanguageMix,
    RecordDisposition,
    ReannotationRequirement,
    SourceType,
    INVALID_CONSENT,
    ANNOTATION_INCOMPLETE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sub(
    review_id: str = "r-1",
    annotator_id: str = "ann-a",
    spam: int = 0,
    deception: int = -1,
    toxicity: int = 0,
    advertising: int = 0,
    off_topic: int = 0,
    pii: int = 0,
    language_mix: LanguageMix = LanguageMix.ENGLISH,
    college_category: CollegeCategory = CollegeCategory.ACADEMICS,
    annotation_guide_version: str = "1.0",
    submitted_at: str = "2025-01-01T00:00:00Z",
    annotation_note: str | None = None,
) -> AnnotationSubmission:
    return AnnotationSubmission(
        review_id=review_id,
        annotator_id=annotator_id,
        annotation_guide_version=annotation_guide_version,
        spam=spam,
        deception=deception,
        toxicity=toxicity,
        advertising=advertising,
        off_topic=off_topic,
        pii=pii,
        language_mix=language_mix,
        college_category=college_category,
        annotation_note=annotation_note,
        submitted_at=submitted_at,
    )


# ---------------------------------------------------------------------------
# AnnotationSubmissionStore
# ---------------------------------------------------------------------------


class TestAnnotationSubmissionStore:
    def test_initial_empty(self):
        store = AnnotationSubmissionStore()
        assert store.get_for_review("r-1") == ()
        assert store.get_a("r-1") is None
        assert store.get_b("r-1") is None
        assert not store.has_both("r-1")

    def test_submit_and_retrieve(self):
        store = AnnotationSubmissionStore()
        sub = _make_sub()
        store.submit(sub)
        assert store.get_for_review("r-1") == (sub,)
        assert store.get_a("r-1") == sub
        assert store.get_b("r-1") is None
        assert not store.has_both("r-1")

    def test_submit_two_different_annotators(self):
        store = AnnotationSubmissionStore()
        sub_a = _make_sub(annotator_id="ann-a")
        sub_b = _make_sub(annotator_id="ann-b")
        store.submit(sub_a)
        store.submit(sub_b)
        assert store.get_a("r-1") == sub_a
        assert store.get_b("r-1") == sub_b
        assert store.has_both("r-1")

    def test_get_for_review_empty_when_no_subs(self):
        store = AnnotationSubmissionStore()
        assert store.get_for_review("nonexistent") == ()

    def test_type_error_on_wrong_type(self):
        store = AnnotationSubmissionStore()
        with pytest.raises(TypeError):
            store.submit("not a submission")  # type: ignore


# ---------------------------------------------------------------------------
# ControlledProtocolTruthStore
# ---------------------------------------------------------------------------


class TestControlledProtocolTruthStore:
    def test_initial_empty(self):
        store = ControlledProtocolTruthStore()
        assert store.get("r-1") is None
        assert store.get_target_value("r-1", "spam") is None

    def test_set_and_get(self):
        store = ControlledProtocolTruthStore()
        truth = ControlledProtocolTruth(
            review_id="r-1",
            control_protocol_id="proto-1",
            target_values={"spam": 0, "deception": 1},
        )
        store.set_truth(truth)
        assert store.get("r-1") == truth
        assert store.get_target_value("r-1", "spam") == 0
        assert store.get_target_value("r-1", "deception") == 1
        assert store.get_target_value("r-1", "nonexistent") is None

    def test_type_error_on_wrong_type(self):
        store = ControlledProtocolTruthStore()
        with pytest.raises(TypeError):
            store.set_truth("not a truth")  # type: ignore


# ---------------------------------------------------------------------------
# DispositionRegister
# ---------------------------------------------------------------------------


class TestDispositionRegister:
    def test_initial_empty(self):
        reg = DispositionRegister()
        assert reg.get("r-1") is None
        assert not reg.is_excluded("r-1")
        assert not reg.is_held("r-1")

    def test_register_and_query(self):
        reg = DispositionRegister()
        disp = RecordDisposition(
            review_id="r-1",
            disposition=Disposition.EXCLUDED,
            reason_code=INVALID_CONSENT,
            recorded_at="2025-01-01T00:00:00Z",
        )
        reg.register(disp)
        assert reg.get("r-1") == disp
        assert reg.is_excluded("r-1")
        assert not reg.is_held("r-1")

    def test_hold_query(self):
        reg = DispositionRegister()
        disp = RecordDisposition(
            review_id="r-1",
            disposition=Disposition.HOLD,
            reason_code=ANNOTATION_INCOMPLETE,
            recorded_at="2025-01-01T00:00:00Z",
        )
        reg.register(disp)
        assert not reg.is_excluded("r-1")
        assert reg.is_held("r-1")

    def test_type_error_on_wrong_type(self):
        reg = DispositionRegister()
        with pytest.raises(TypeError):
            reg.register("not a disposition")  # type: ignore


# ---------------------------------------------------------------------------
# ReannotationRegister
# ---------------------------------------------------------------------------


class TestReannotationRegister:
    def test_initial_empty(self):
        reg = ReannotationRegister()
        assert reg.get_pending() == ()
        assert reg.get_for_review("r-1") is None

    def test_require_and_query(self):
        reg = ReannotationRegister()
        req = ReannotationRequirement(
            review_id="r-1",
            old_guide_version="1.0",
            required_guide_version="2.0",
            reason="Guide revised",
            recorded_at="2025-01-01T00:00:00Z",
        )
        reg.require(req)
        assert reg.get_for_review("r-1") == req
        assert reg.get_pending() == (req,)

    def test_overwrite_on_same_review(self):
        reg = ReannotationRegister()
        req1 = ReannotationRequirement(
            review_id="r-1",
            old_guide_version="1.0",
            required_guide_version="2.0",
            reason="First",
            recorded_at="2025-01-01T00:00:00Z",
        )
        req2 = ReannotationRequirement(
            review_id="r-1",
            old_guide_version="2.0",
            required_guide_version="3.0",
            reason="Second",
            recorded_at="2025-01-02T00:00:00Z",
        )
        reg.require(req1)
        reg.require(req2)
        assert reg.get_for_review("r-1") == req2
        assert reg.get_pending() == (req2,)

    def test_type_error_on_wrong_type(self):
        reg = ReannotationRegister()
        with pytest.raises(TypeError):
            reg.require("not a requirement")  # type: ignore


# ---------------------------------------------------------------------------
# compute_eight_dimension_disagreement
# ---------------------------------------------------------------------------


class TestComputeEightDimensionDisagreement:
    def test_identical_submissions_no_disagreement(self):
        a = _make_sub()
        b = _make_sub()
        result = compute_eight_dimension_disagreement(a, b)
        assert all(v is False for v in result.values())

    def test_label_disagreement_detected(self):
        a = _make_sub(spam=0)
        b = _make_sub(spam=1)
        result = compute_eight_dimension_disagreement(a, b)
        assert result["spam"] is True
        # All other dimensions should agree
        assert result["toxicity"] is False
        assert result["deception"] is False
        assert result["advertising"] is False
        assert result["off_topic"] is False
        assert result["pii"] is False

    def test_language_mix_disagreement(self):
        a = _make_sub(language_mix=LanguageMix.ENGLISH)
        b = _make_sub(language_mix=LanguageMix.HINGLISH)
        result = compute_eight_dimension_disagreement(a, b)
        assert result["language_mix"] is True

    def test_college_category_disagreement(self):
        a = _make_sub(college_category=CollegeCategory.ACADEMICS)
        b = _make_sub(college_category=CollegeCategory.PLACEMENTS)
        result = compute_eight_dimension_disagreement(a, b)
        assert result["college_category"] is True

    def test_both_none_language_mix_agrees(self):
        a = _make_sub(language_mix=None)
        b = _make_sub(language_mix=None)
        result = compute_eight_dimension_disagreement(a, b)
        assert result["language_mix"] is False

    def test_all_dimensions_returned(self):
        a = _make_sub()
        b = _make_sub()
        result = compute_eight_dimension_disagreement(a, b)
        assert set(result.keys()) == {
            "spam", "deception", "toxicity", "advertising",
            "off_topic", "pii", "language_mix", "college_category",
        }


# ---------------------------------------------------------------------------
# needs_adjudication
# ---------------------------------------------------------------------------


class TestNeedsAdjudication:
    def test_no_disagreement(self):
        dims = {
            "spam": False, "deception": False, "toxicity": False,
            "advertising": False, "off_topic": False, "pii": False,
            "language_mix": False, "college_category": False,
        }
        assert needs_adjudication(dims) is False

    def test_single_disagreement(self):
        dims = {
            "spam": True, "deception": False, "toxicity": False,
            "advertising": False, "off_topic": False, "pii": False,
            "language_mix": False, "college_category": False,
        }
        assert needs_adjudication(dims) is True

    def test_all_disagree(self):
        dims = {k: True for k in (
            "spam", "deception", "toxicity", "advertising",
            "off_topic", "pii", "language_mix", "college_category",
        )}
        assert needs_adjudication(dims) is True


# ---------------------------------------------------------------------------
# adjudicator_decision
# ---------------------------------------------------------------------------


class TestAdjudicatorDecision:
    def test_empty_choices_inherits_from_a(self):
        a = _make_sub(spam=0, deception=-1, toxicity=1)
        b = _make_sub(spam=0, deception=-1, toxicity=1)
        result = adjudicator_decision(a, b, {})
        assert result["spam"] == 0
        assert result["deception"] == -1
        assert result["toxicity"] == 1
        assert result["advertising"] == 0
        assert result["off_topic"] == 0
        assert result["pii"] == 0

    def test_choices_override_a(self):
        a = _make_sub(spam=0, deception=-1)
        b = _make_sub(spam=1, deception=0)
        result = adjudicator_decision(
            a, b, {"spam": 0, "deception": 0}
        )
        assert result["spam"] == 0  # chosen by adjudicator
        assert result["deception"] == 0  # chosen by adjudicator

    def test_unknown_dimension_ignored(self):
        a = _make_sub()
        b = _make_sub()
        result = adjudicator_decision(a, b, {"nonexistent_dim": 1})
        # Should not raise; unknown dimension is ignored
        assert "nonexistent_dim" not in result

    def test_none_choice_inherits_from_a(self):
        a = _make_sub(spam=1)
        b = _make_sub(spam=0)
        result = adjudicator_decision(a, b, {"spam": None})
        # None choice means inherit from A
        assert result["spam"] == 1

    def test_type_error_non_dict(self):
        a = _make_sub()
        b = _make_sub()
        with pytest.raises(TypeError):
            adjudicator_decision(a, b, "not a dict")  # type: ignore

    def test_enum_dimensions_serialized_as_values(self):
        a = _make_sub(language_mix=LanguageMix.ENGLISH, college_category=CollegeCategory.ACADEMICS)
        b = _make_sub(language_mix=LanguageMix.ENGLISH, college_category=CollegeCategory.ACADEMICS)
        result = adjudicator_decision(a, b, {})
        assert result["language_mix"] == LanguageMix.ENGLISH.value
        assert result["college_category"] == CollegeCategory.ACADEMICS.value
