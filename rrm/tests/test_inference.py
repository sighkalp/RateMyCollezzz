"""Pytest test suite for RRM 3.11 inference contract — FINAL HARDENING.

Tests the public runtime boundary: RrmInferenceRequest, RrmInferenceResult,
RrmInferenceEngine, NeuralAvailability, and related types.

All neural-dependent tests use fake/stub predictors.
No production checkpoint is required.
No production tokenizer is required.
No network calls are made.
"""

from __future__ import annotations

import json
import math
import types

import pytest

from rrm import (
    NeuralAvailability,
    RrmConfigurationError,
    RrmInferenceEngine,
    RrmInferenceError,
    RrmInferenceRequest,
    RrmInferenceResult,
    RRM_RUNTIME_SCHEMA_VERSION,
)
from rrm.labels import PRIMARY_LABELS
from rrm.deterministic_prechecks import DeterministicPrecheckResult, run_deterministic_prechecks


# ---------------------------------------------------------------------------
# Stub neural predictors
# ---------------------------------------------------------------------------

def _make_stub_predictor(logits):
    """Create a callable that returns fixed logits."""
    def predictor(_text: str):
        return list(logits)
    return predictor


# Six zeros → sigmoid(0) = 0.5 each
_STUB_ZERO_LOGITS = [0.0] * 6

# Positive/negative logits → varied sigmoid values
_STUB_MIXED_LOGITS = [3.0, -1.0, 2.0, -2.0, 1.0, 0.5]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def valid_request():
    return RrmInferenceRequest(
        review_text="This is a test review.",
        review_id="r-001",
    )


@pytest.fixture()
def valid_request_with_candidates():
    return RrmInferenceRequest(
        review_text="This is a test review.",
        review_id="r-001",
        similarity_candidates=(
            ("r1", "This is a similar review."),
            ("r2", "Completely unrelated text here."),
        ),
    )


@pytest.fixture()
def engine_deterministic_only():
    """Engine with no neural predictor (deterministic-only mode)."""
    return RrmInferenceEngine()


@pytest.fixture()
def engine_with_stub():
    """Engine with a zero-logit stub predictor."""
    return RrmInferenceEngine(
        neural_predictor=_make_stub_predictor(_STUB_ZERO_LOGITS),
        model_identity="stub-model-1",
        tokenizer_identity="stub-tokenizer-1",
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _assert_deterministic_present(result: RrmInferenceResult):
    assert isinstance(result.deterministic_prechecks, DeterministicPrecheckResult)


def _make_available_result(model="m", tokenizer="t"):
    """Helper to construct a valid AVAILABLE result."""
    predictor = _make_stub_predictor(_STUB_ZERO_LOGITS)
    engine = RrmInferenceEngine(
        neural_predictor=predictor,
        model_identity=model,
        tokenizer_identity=tokenizer,
    )
    req = RrmInferenceRequest(review_text="test", review_id="r1")
    return engine.analyze(req)


def _make_real_precheck(review_id="r1", review_text="test text"):
    """Helper to build a real DeterministicPrecheckResult via the locked module."""
    return run_deterministic_prechecks(
        review_id=review_id,
        review_text=review_text,
    )


# ===========================================================================
# Section A: RrmInferenceRequest — review_id is truly required
# ===========================================================================

class TestReviewIdRequired:
    """Section A: review_id is a required field — no default, strict validation."""

    def test_a01_review_id_required_no_default(self):
        """Omitting review_id must fail at construction (no default)."""
        with pytest.raises(TypeError):
            RrmInferenceRequest(review_text="hello world")

    def test_a02_empty_review_id_raises(self):
        with pytest.raises(ValueError):
            RrmInferenceRequest(review_text="hello", review_id="")

    def test_a03_whitespace_only_review_id_raises(self):
        with pytest.raises(ValueError):
            RrmInferenceRequest(review_text="hello", review_id="   \t  ")

    def test_a04_non_string_review_id_raises(self):
        with pytest.raises(TypeError):
            RrmInferenceRequest(review_text="hello", review_id=123)

    def test_a05_valid_review_id_accepted(self):
        req = RrmInferenceRequest(review_text="hello", review_id="r-42")
        assert req.review_id == "r-42"

    def test_a06_analyze_passes_review_id_to_prechecks(self, engine_deterministic_only):
        """engine.analyze must pass request.review_id to run_deterministic_prechecks."""
        req = RrmInferenceRequest(review_text="college review", review_id="exact-id")
        result = engine_deterministic_only.analyze(req)
        assert result.deterministic_prechecks.review_id == "exact-id"


# ===========================================================================
# Section B: RrmInferenceRequest — deep immutability of similarity_candidates
# ===========================================================================

class TestDeepImmutability:
    """Section B: caller mutations must not leak into the request."""

    def test_b01_list_input_deep_copied(self):
        original = [["r1", "hello world"]]
        req = RrmInferenceRequest(
            review_text="review",
            review_id="current",
            similarity_candidates=original,
        )
        # Mutate original
        original[0][1] = "changed text"
        # Request must retain original value
        assert req.similarity_candidates[0][1] == "hello world"

    def test_b02_nested_list_mutation_blocked(self):
        original = [["r1", "text A"], ["r2", "text B"]]
        req = RrmInferenceRequest(
            review_text="review",
            review_id="current",
            similarity_candidates=original,
        )
        original[0][0] = "mutated"
        original[1] = ["r3", "mutated"]
        assert req.similarity_candidates[0][0] == "r1"
        assert req.similarity_candidates[1] == ("r2", "text B")

    def test_b03_result_is_tuple_of_tuples(self):
        req = RrmInferenceRequest(
            review_text="review",
            review_id="current",
            similarity_candidates=[["r1", "hello"]],
        )
        assert isinstance(req.similarity_candidates, tuple)
        assert isinstance(req.similarity_candidates[0], tuple)

    def test_b04_tuple_input_accepted(self):
        req = RrmInferenceRequest(
            review_text="review",
            review_id="current",
            similarity_candidates=(("r1", "hello"),),
        )
        assert req.similarity_candidates == (("r1", "hello"),)

    def test_b05_empty_candidates_defaults_to_empty_tuple(self):
        req = RrmInferenceRequest(review_text="review", review_id="r1")
        assert req.similarity_candidates == ()

    def test_b06_outer_list_mutation_blocked(self):
        original = [["r1", "hello"]]
        req = RrmInferenceRequest(
            review_text="review",
            review_id="current",
            similarity_candidates=original,
        )
        original.append(["r2", "extra"])
        assert len(req.similarity_candidates) == 1


# ===========================================================================
# Section C: Engine configuration validation
# ===========================================================================

class TestEngineConfiguration:
    """Section C: __init__ must validate configuration explicitly."""

    # --- neural_predictor must be callable ---

    def test_c01_non_callable_predictor_raises(self):
        with pytest.raises(RrmConfigurationError):
            RrmInferenceEngine(neural_predictor=123)

    def test_c02_int_predictor_raises(self):
        with pytest.raises(RrmConfigurationError):
            RrmInferenceEngine(neural_predictor=123)

    def test_c03_string_predictor_raises(self):
        with pytest.raises(RrmConfigurationError):
            RrmInferenceEngine(neural_predictor="not callable")

    def test_c04_none_predictor_accepted(self):
        engine = RrmInferenceEngine()
        assert engine._neural_predictor is None

    def test_c05_callable_predictor_accepted(self):
        engine = RrmInferenceEngine(
            neural_predictor=lambda x: [0.0] * 6,
            model_identity="m",
            tokenizer_identity="t",
        )
        assert callable(engine._neural_predictor)

    # --- model_identity validation ---

    def test_c06_non_string_model_identity_raises(self):
        with pytest.raises(RrmConfigurationError):
            RrmInferenceEngine(
                neural_predictor=lambda x: [0.0] * 6,
                model_identity=123,
                tokenizer_identity="t",
            )

    def test_c07_empty_model_identity_raises(self):
        with pytest.raises(RrmConfigurationError):
            RrmInferenceEngine(
                neural_predictor=lambda x: [0.0] * 6,
                model_identity="",
                tokenizer_identity="t",
            )

    def test_c08_whitespace_model_identity_raises(self):
        with pytest.raises(RrmConfigurationError):
            RrmInferenceEngine(
                neural_predictor=lambda x: [0.0] * 6,
                model_identity="   ",
                tokenizer_identity="t",
            )

    def test_c09_valid_model_identity_stripped(self):
        engine = RrmInferenceEngine(
            neural_predictor=lambda x: [0.0] * 6,
            model_identity="  my-model  ",
            tokenizer_identity="t",
        )
        assert engine._model_identity == "my-model"

    # --- tokenizer_identity validation ---

    def test_c10_non_string_tokenizer_identity_raises(self):
        with pytest.raises(RrmConfigurationError):
            RrmInferenceEngine(
                neural_predictor=lambda x: [0.0] * 6,
                model_identity="m",
                tokenizer_identity=456,
            )

    def test_c11_empty_tokenizer_identity_raises(self):
        with pytest.raises(RrmConfigurationError):
            RrmInferenceEngine(
                neural_predictor=lambda x: [0.0] * 6,
                model_identity="m",
                tokenizer_identity="",
            )

    def test_c12_whitespace_tokenizer_identity_raises(self):
        with pytest.raises(RrmConfigurationError):
            RrmInferenceEngine(
                neural_predictor=lambda x: [0.0] * 6,
                model_identity="m",
                tokenizer_identity="\t\n",
            )

    # --- deterministic-only mode rejects identities ---

    def test_c13_model_identity_without_predictor_raises(self):
        with pytest.raises(RrmConfigurationError):
            RrmInferenceEngine(model_identity="some-model")

    def test_c14_tokenizer_identity_without_predictor_raises(self):
        with pytest.raises(RrmConfigurationError):
            RrmInferenceEngine(tokenizer_identity="some-tokenizer")

    def test_c15_both_identities_without_predictor_raise(self):
        with pytest.raises(RrmConfigurationError):
            RrmInferenceEngine(
                model_identity="m",
                tokenizer_identity="t",
            )


# ===========================================================================
# Section D: Neural output materialization
# ===========================================================================

class TestNeuralOutputMaterialization:
    """Section D: non-iterable predictor output must raise RrmInferenceError."""

    def test_d01_none_output_raises(self):
        engine = RrmInferenceEngine(
            neural_predictor=lambda x: None,  # type: ignore[return-value]
            model_identity="m",
            tokenizer_identity="t",
        )
        req = RrmInferenceRequest(review_text="test", review_id="r1")
        with pytest.raises(RrmInferenceError):
            engine.analyze(req)

    def test_d02_int_output_raises(self):
        engine = RrmInferenceEngine(
            neural_predictor=lambda x: 1,  # type: ignore[return-value]
            model_identity="m",
            tokenizer_identity="t",
        )
        req = RrmInferenceRequest(review_text="test", review_id="r1")
        with pytest.raises(RrmInferenceError):
            engine.analyze(req)

    def test_d03_float_output_raises(self):
        engine = RrmInferenceEngine(
            neural_predictor=lambda x: 1.0,  # type: ignore[return-value]
            model_identity="m",
            tokenizer_identity="t",
        )
        req = RrmInferenceRequest(review_text="test", review_id="r1")
        with pytest.raises(RrmInferenceError):
            engine.analyze(req)

    def test_d04_cause_preserved_on_materialization_error(self):
        engine = RrmInferenceEngine(
            neural_predictor=lambda x: None,  # type: ignore[return-value]
            model_identity="m",
            tokenizer_identity="t",
        )
        req = RrmInferenceRequest(review_text="test", review_id="r1")
        with pytest.raises(RrmInferenceError) as exc_info:
            engine.analyze(req)
        assert exc_info.value.__cause__ is not None


# ===========================================================================
# Section E: Neural logit type validation
# ===========================================================================

class TestNeuralLogitValidation:
    """Section E: each logit must be a real numeric, not bool, finite."""

    def _make_failing_predictor(self, logits):
        return RrmInferenceEngine(
            neural_predictor=_make_stub_predictor(logits),
            model_identity="m",
            tokenizer_identity="t",
        )

    def test_e01_bool_logit_rejected(self):
        """bool must be rejected even though isinstance(True, int) is True."""
        engine = self._make_failing_predictor([True, 0.0, 0.0, 0.0, 0.0, 0.0])
        req = RrmInferenceRequest(review_text="test", review_id="r1")
        with pytest.raises(RrmInferenceError):
            engine.analyze(req)

    def test_e02_string_logit_rejected(self):
        engine = self._make_failing_predictor(["bad", 0.0, 0.0, 0.0, 0.0, 0.0])
        req = RrmInferenceRequest(review_text="test", review_id="r1")
        with pytest.raises(RrmInferenceError):
            engine.analyze(req)

    def test_e03_none_logit_rejected(self):
        engine = self._make_failing_predictor([None, 0.0, 0.0, 0.0, 0.0, 0.0])
        req = RrmInferenceRequest(review_text="test", review_id="r1")
        with pytest.raises(RrmInferenceError):
            engine.analyze(req)

    def test_e04_complex_logit_rejected(self):
        engine = self._make_failing_predictor([1.0 + 2j, 0.0, 0.0, 0.0, 0.0, 0.0])
        req = RrmInferenceRequest(review_text="test", review_id="r1")
        with pytest.raises(RrmInferenceError):
            engine.analyze(req)

    def test_e05_nan_logit_rejected(self):
        engine = self._make_failing_predictor(
            [float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        req = RrmInferenceRequest(review_text="test", review_id="r1")
        with pytest.raises(RrmInferenceError):
            engine.analyze(req)

    def test_e06_positive_inf_logit_rejected(self):
        engine = self._make_failing_predictor(
            [float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        req = RrmInferenceRequest(review_text="test", review_id="r1")
        with pytest.raises(RrmInferenceError):
            engine.analyze(req)

    def test_e07_negative_inf_logit_rejected(self):
        engine = self._make_failing_predictor(
            [float("-inf"), 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        req = RrmInferenceRequest(review_text="test", review_id="r1")
        with pytest.raises(RrmInferenceError):
            engine.analyze(req)


# ===========================================================================
# Section F: Exact six logits
# ===========================================================================

class TestExactSixLogits:
    """Section F: predictor must return exactly len(PRIMARY_LABELS) logits."""

    def test_f01_too_few_logits_raises(self):
        engine = RrmInferenceEngine(
            neural_predictor=_make_stub_predictor([0.0, 1.0]),
            model_identity="m",
            tokenizer_identity="t",
        )
        req = RrmInferenceRequest(review_text="test", review_id="r1")
        with pytest.raises(RrmInferenceError):
            engine.analyze(req)

    def test_f02_too_many_logits_raises(self):
        engine = RrmInferenceEngine(
            neural_predictor=_make_stub_predictor([0.0] * 8),
            model_identity="m",
            tokenizer_identity="t",
        )
        req = RrmInferenceRequest(review_text="test", review_id="r1")
        with pytest.raises(RrmInferenceError):
            engine.analyze(req)

    def test_f03_exact_six_logits_accepted(self):
        engine = RrmInferenceEngine(
            neural_predictor=_make_stub_predictor([0.0] * 6),
            model_identity="m",
            tokenizer_identity="t",
        )
        req = RrmInferenceRequest(review_text="test", review_id="r1")
        result = engine.analyze(req)
        assert len(result.task_scores) == 6


# ===========================================================================
# Section G: RrmInferenceResult — deterministic_prechecks type
# ===========================================================================

class TestResultDeterministicPrechecks:
    """Section G: deterministic_prechecks must be DeterministicPrecheckResult."""

    def test_g01_wrong_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            RrmInferenceResult(
                deterministic_prechecks="not a result",  # type: ignore[arg-type]
                task_scores=None,
                neural_availability=NeuralAvailability.UNAVAILABLE,
                neural_unavailable_reason="reason",
                model_identity=None,
                tokenizer_identity=None,
            )

    def test_g02_valid_prechecks_accepted(self, engine_deterministic_only, valid_request):
        result = engine_deterministic_only.analyze(valid_request)
        assert isinstance(
            result.deterministic_prechecks, DeterministicPrecheckResult
        )


# ===========================================================================
# Section H: RrmInferenceResult — runtime_schema_version
# ===========================================================================

class TestResultSchemaVersion:
    """Section H: runtime_schema_version must equal RRM_RUNTIME_SCHEMA_VERSION."""

    def test_h01_wrong_version_raises(self):
        with pytest.raises(ValueError):
            RrmInferenceResult(
                deterministic_prechecks=_make_real_precheck(),
                task_scores=None,
                neural_availability=NeuralAvailability.UNAVAILABLE,
                neural_unavailable_reason="reason",
                model_identity=None,
                tokenizer_identity=None,
                runtime_schema_version="2.0",
            )

    def test_h02_default_is_correct_version(self, engine_deterministic_only, valid_request):
        result = engine_deterministic_only.analyze(valid_request)
        assert result.runtime_schema_version == RRM_RUNTIME_SCHEMA_VERSION

    def test_h03_correct_version_accepted(self):
        result = RrmInferenceResult(
            deterministic_prechecks=_make_real_precheck(),
            task_scores=None,
            neural_availability=NeuralAvailability.UNAVAILABLE,
            neural_unavailable_reason="reason",
            model_identity=None,
            tokenizer_identity=None,
            runtime_schema_version=RRM_RUNTIME_SCHEMA_VERSION,
        )
        assert result.runtime_schema_version == "1.0"


# ===========================================================================
# Section I: AVAILABLE result contract
# ===========================================================================

class TestAvailableResultContract:
    """Section I: when neural_availability == AVAILABLE, all invariants hold."""

    def test_i01_task_scores_is_mapping(self):
        result = _make_available_result()
        assert isinstance(result.task_scores, types.MappingProxyType)

    def test_i02_task_scores_keys_match_primary_labels(self):
        result = _make_available_result()
        assert list(result.task_scores.keys()) == list(PRIMARY_LABELS)

    def test_i03_missing_label_raises(self):
        """Constructing a result with missing label must raise."""
        scores = {label: 0.5 for label in PRIMARY_LABELS[:-1]}
        with pytest.raises(ValueError):
            RrmInferenceResult(
                deterministic_prechecks=_make_real_precheck(),
                task_scores=scores,
                neural_availability=NeuralAvailability.AVAILABLE,
                neural_unavailable_reason=None,
                model_identity="m",
                tokenizer_identity="t",
            )

    def test_i04_extra_label_raises(self):
        scores = dict(zip(PRIMARY_LABELS, [0.5] * 6))
        scores["extra_label"] = 0.5
        with pytest.raises(ValueError):
            RrmInferenceResult(
                deterministic_prechecks=_make_real_precheck(),
                task_scores=scores,
                neural_availability=NeuralAvailability.AVAILABLE,
                neural_unavailable_reason=None,
                model_identity="m",
                tokenizer_identity="t",
            )

    def test_i05_wrong_label_raises(self):
        scores = {"wrong": 0.5, "labels": 0.5, "here": 0.5, "a": 0.5, "b": 0.5, "c": 0.5}
        with pytest.raises(ValueError):
            RrmInferenceResult(
                deterministic_prechecks=_make_real_precheck(),
                task_scores=scores,
                neural_availability=NeuralAvailability.AVAILABLE,
                neural_unavailable_reason=None,
                model_identity="m",
                tokenizer_identity="t",
            )

    # --- each score: real numeric, not bool, finite, in [0, 1] ---

    def test_i06_bool_score_rejected(self):
        scores = dict(zip(PRIMARY_LABELS, [True] + [0.5] * 5))
        with pytest.raises((TypeError, ValueError)):
            RrmInferenceResult(
                deterministic_prechecks=_make_real_precheck(),
                task_scores=scores,
                neural_availability=NeuralAvailability.AVAILABLE,
                neural_unavailable_reason=None,
                model_identity="m",
                tokenizer_identity="t",
            )

    def test_i07_nan_score_rejected(self):
        scores = dict(zip(PRIMARY_LABELS, [float("nan")] + [0.5] * 5))
        with pytest.raises(ValueError):
            RrmInferenceResult(
                deterministic_prechecks=_make_real_precheck(),
                task_scores=scores,
                neural_availability=NeuralAvailability.AVAILABLE,
                neural_unavailable_reason=None,
                model_identity="m",
                tokenizer_identity="t",
            )

    def test_i08_inf_score_rejected(self):
        scores = dict(zip(PRIMARY_LABELS, [float("inf")] + [0.5] * 5))
        with pytest.raises(ValueError):
            RrmInferenceResult(
                deterministic_prechecks=_make_real_precheck(),
                task_scores=scores,
                neural_availability=NeuralAvailability.AVAILABLE,
                neural_unavailable_reason=None,
                model_identity="m",
                tokenizer_identity="t",
            )

    def test_i09_negative_score_rejected(self):
        scores = dict(zip(PRIMARY_LABELS, [-0.1] + [0.5] * 5))
        with pytest.raises(ValueError):
            RrmInferenceResult(
                deterministic_prechecks=_make_real_precheck(),
                task_scores=scores,
                neural_availability=NeuralAvailability.AVAILABLE,
                neural_unavailable_reason=None,
                model_identity="m",
                tokenizer_identity="t",
            )

    def test_i10_score_above_one_rejected(self):
        scores = dict(zip(PRIMARY_LABELS, [1.1] + [0.5] * 5))
        with pytest.raises(ValueError):
            RrmInferenceResult(
                deterministic_prechecks=_make_real_precheck(),
                task_scores=scores,
                neural_availability=NeuralAvailability.AVAILABLE,
                neural_unavailable_reason=None,
                model_identity="m",
                tokenizer_identity="t",
            )

    def test_i11_string_score_rejected(self):
        scores = dict(zip(PRIMARY_LABELS, ["0.5"] + [0.5] * 5))
        with pytest.raises(TypeError):
            RrmInferenceResult(
                deterministic_prechecks=_make_real_precheck(),
                task_scores=scores,
                neural_availability=NeuralAvailability.AVAILABLE,
                neural_unavailable_reason=None,
                model_identity="m",
                tokenizer_identity="t",
            )

    def test_i12_none_score_rejected(self):
        scores = dict(zip(PRIMARY_LABELS, [None] + [0.5] * 5))
        with pytest.raises(TypeError):
            RrmInferenceResult(
                deterministic_prechecks=_make_real_precheck(),
                task_scores=scores,
                neural_availability=NeuralAvailability.AVAILABLE,
                neural_unavailable_reason=None,
                model_identity="m",
                tokenizer_identity="t",
            )

    # --- unavailable_reason, model_identity, tokenizer_identity ---

    def test_i13_unavailable_reason_must_be_none(self):
        with pytest.raises(ValueError):
            RrmInferenceResult(
                deterministic_prechecks=_make_real_precheck(),
                task_scores=dict(zip(PRIMARY_LABELS, [0.5] * 6)),
                neural_availability=NeuralAvailability.AVAILABLE,
                neural_unavailable_reason="should be None",
                model_identity="m",
                tokenizer_identity="t",
            )

    def test_i14_model_identity_non_empty(self):
        with pytest.raises(ValueError):
            RrmInferenceResult(
                deterministic_prechecks=_make_real_precheck(),
                task_scores=dict(zip(PRIMARY_LABELS, [0.5] * 6)),
                neural_availability=NeuralAvailability.AVAILABLE,
                neural_unavailable_reason=None,
                model_identity="",
                tokenizer_identity="t",
            )

    def test_i15_model_identity_non_empty_whitespace(self):
        with pytest.raises(ValueError):
            RrmInferenceResult(
                deterministic_prechecks=_make_real_precheck(),
                task_scores=dict(zip(PRIMARY_LABELS, [0.5] * 6)),
                neural_availability=NeuralAvailability.AVAILABLE,
                neural_unavailable_reason=None,
                model_identity="   ",
                tokenizer_identity="t",
            )

    def test_i16_tokenizer_identity_non_empty(self):
        with pytest.raises(ValueError):
            RrmInferenceResult(
                deterministic_prechecks=_make_real_precheck(),
                task_scores=dict(zip(PRIMARY_LABELS, [0.5] * 6)),
                neural_availability=NeuralAvailability.AVAILABLE,
                neural_unavailable_reason=None,
                model_identity="m",
                tokenizer_identity="",
            )

    def test_i17_model_identity_type_error(self):
        with pytest.raises(TypeError):
            RrmInferenceResult(
                deterministic_prechecks=_make_real_precheck(),
                task_scores=dict(zip(PRIMARY_LABELS, [0.5] * 6)),
                neural_availability=NeuralAvailability.AVAILABLE,
                neural_unavailable_reason=None,
                model_identity=123,  # type: ignore[arg-type]
                tokenizer_identity="t",
            )

    def test_i18_tokenizer_identity_type_error(self):
        with pytest.raises(TypeError):
            RrmInferenceResult(
                deterministic_prechecks=_make_real_precheck(),
                task_scores=dict(zip(PRIMARY_LABELS, [0.5] * 6)),
                neural_availability=NeuralAvailability.AVAILABLE,
                neural_unavailable_reason=None,
                model_identity="m",
                tokenizer_identity=456,  # type: ignore[arg-type]
            )


# ===========================================================================
# Section J: UNAVAILABLE result contract
# ===========================================================================

class TestUnavailableResultContract:
    """Section J: when neural_availability == UNAVAILABLE, all invariants hold."""

    def test_j01_valid_unavailable_result(self):
        result = RrmInferenceResult(
            deterministic_prechecks=_make_real_precheck(),
            task_scores=None,
            neural_availability=NeuralAvailability.UNAVAILABLE,
            neural_unavailable_reason="production_assets_unavailable",
            model_identity=None,
            tokenizer_identity=None,
        )
        assert result.task_scores is None
        assert result.neural_unavailable_reason == "production_assets_unavailable"
        assert result.model_identity is None
        assert result.tokenizer_identity is None

    def test_j02_task_scores_with_unavailable_raises(self):
        with pytest.raises(ValueError):
            RrmInferenceResult(
                deterministic_prechecks=_make_real_precheck(),
                task_scores=dict(zip(PRIMARY_LABELS, [0.5] * 6)),
                neural_availability=NeuralAvailability.UNAVAILABLE,
                neural_unavailable_reason="reason",
                model_identity=None,
                tokenizer_identity=None,
            )

    def test_j03_empty_unavailable_reason_raises(self):
        with pytest.raises(ValueError):
            RrmInferenceResult(
                deterministic_prechecks=_make_real_precheck(),
                task_scores=None,
                neural_availability=NeuralAvailability.UNAVAILABLE,
                neural_unavailable_reason="",
                model_identity=None,
                tokenizer_identity=None,
            )

    def test_j04_none_unavailable_reason_raises(self):
        with pytest.raises(TypeError):
            RrmInferenceResult(
                deterministic_prechecks=_make_real_precheck(),
                task_scores=None,
                neural_availability=NeuralAvailability.UNAVAILABLE,
                neural_unavailable_reason=None,
                model_identity=None,
                tokenizer_identity=None,
            )

    def test_j05_whitespace_unavailable_reason_raises(self):
        with pytest.raises(ValueError):
            RrmInferenceResult(
                deterministic_prechecks=_make_real_precheck(),
                task_scores=None,
                neural_availability=NeuralAvailability.UNAVAILABLE,
                neural_unavailable_reason="   ",
                model_identity=None,
                tokenizer_identity=None,
            )

    def test_j06_model_identity_with_unavailable_raises(self):
        with pytest.raises(ValueError):
            RrmInferenceResult(
                deterministic_prechecks=_make_real_precheck(),
                task_scores=None,
                neural_availability=NeuralAvailability.UNAVAILABLE,
                neural_unavailable_reason="reason",
                model_identity="m",
                tokenizer_identity=None,
            )

    def test_j07_tokenizer_identity_with_unavailable_raises(self):
        with pytest.raises(ValueError):
            RrmInferenceResult(
                deterministic_prechecks=_make_real_precheck(),
                task_scores=None,
                neural_availability=NeuralAvailability.UNAVAILABLE,
                neural_unavailable_reason="reason",
                model_identity=None,
                tokenizer_identity="t",
            )


# ===========================================================================
# Section K: Invalid neural_availability value
# ===========================================================================

class TestInvalidNeuralAvailability:
    """Section K: neural_availability must be one of the two known values."""

    def test_k01_random_string_raises(self):
        with pytest.raises(ValueError):
            RrmInferenceResult(
                deterministic_prechecks=_make_real_precheck(),
                task_scores=None,
                neural_availability="maybe",
                neural_unavailable_reason="reason",
                model_identity=None,
                tokenizer_identity=None,
            )

    def test_k02_integer_raises(self):
        with pytest.raises(ValueError):
            RrmInferenceResult(
                deterministic_prechecks=_make_real_precheck(),
                task_scores=None,
                neural_availability=1,
                neural_unavailable_reason="reason",
                model_identity=None,
                tokenizer_identity=None,
            )


# ===========================================================================
# Section L: Engine output semantics (AVAILABLE)
# ===========================================================================

class TestEngineAvailableSemantics:
    """Section L: neural-enabled engine output must match the contract."""

    def test_l01_task_scores_is_mapping_proxy(self):
        result = _make_available_result()
        assert isinstance(result.task_scores, types.MappingProxyType)

    def test_l02_task_scores_not_plain_dict(self):
        result = _make_available_result()
        assert not isinstance(result.task_scores, dict)

    def test_l03_task_scores_keys_in_primary_labels_order(self):
        result = _make_available_result()
        assert list(result.task_scores.keys()) == list(PRIMARY_LABELS)

    def test_l04_neural_availability_available(self):
        result = _make_available_result()
        assert result.neural_availability == NeuralAvailability.AVAILABLE

    def test_l05_unavailable_reason_none(self):
        result = _make_available_result()
        assert result.neural_unavailable_reason is None

    def test_l06_model_identity_set(self):
        result = _make_available_result(model="my-model-v1")
        assert result.model_identity == "my-model-v1"

    def test_l07_tokenizer_identity_set(self):
        result = _make_available_result(tokenizer="my-tok-v1")
        assert result.tokenizer_identity == "my-tok-v1"


# ===========================================================================
# Section M: Engine output semantics (UNAVAILABLE)
# ===========================================================================

class TestEngineUnavailableSemantics:
    """Section M: deterministic-only engine output must match the contract."""

    def _unavailable_result(self):
        engine = RrmInferenceEngine()
        req = RrmInferenceRequest(review_text="test", review_id="r1")
        return engine.analyze(req)

    def test_m01_task_scores_none(self):
        result = self._unavailable_result()
        assert result.task_scores is None

    def test_m02_neural_availability_unavailable(self):
        result = self._unavailable_result()
        assert result.neural_availability == NeuralAvailability.UNAVAILABLE

    def test_m03_unavailable_reason_exact(self):
        result = self._unavailable_result()
        assert result.neural_unavailable_reason == "production_assets_unavailable"

    def test_m04_model_identity_none(self):
        result = self._unavailable_result()
        assert result.model_identity is None

    def test_m05_tokenizer_identity_none(self):
        result = self._unavailable_result()
        assert result.tokenizer_identity is None


# ===========================================================================
# Section N: Direct-construction deep immutability
# ===========================================================================

class TestDirectConstructionImmutability:
    """Section N: Direct RrmInferenceResult construction must defensively copy."""

    def test_n01_caller_mutation_does_not_leak(self):
        """Mutating the original dict after construction must not affect result."""
        scores = {
            label: 0.5
            for label in PRIMARY_LABELS
        }
        result = RrmInferenceResult(
            deterministic_prechecks=_make_real_precheck(),
            task_scores=scores,
            neural_availability=NeuralAvailability.AVAILABLE,
            neural_unavailable_reason=None,
            model_identity="m",
            tokenizer_identity="t",
        )
        # Mutate the original
        scores["spam"] = 0.99
        # Result must retain original value
        assert result.task_scores["spam"] == 0.5

    def test_n02_result_immutable_against_assignment(self):
        """Assignment to result.task_scores must raise TypeError."""
        scores = {
            label: 0.5
            for label in PRIMARY_LABELS
        }
        result = RrmInferenceResult(
            deterministic_prechecks=_make_real_precheck(),
            task_scores=scores,
            neural_availability=NeuralAvailability.AVAILABLE,
            neural_unavailable_reason=None,
            model_identity="m",
            tokenizer_identity="t",
        )
        with pytest.raises(TypeError):
            result.task_scores["spam"] = 0.1  # type: ignore[index]

    def test_n03_result_is_mapping_proxy_type(self):
        scores = {
            label: 0.5
            for label in PRIMARY_LABELS
        }
        result = RrmInferenceResult(
            deterministic_prechecks=_make_real_precheck(),
            task_scores=scores,
            neural_availability=NeuralAvailability.AVAILABLE,
            neural_unavailable_reason=None,
            model_identity="m",
            tokenizer_identity="t",
        )
        assert isinstance(result.task_scores, types.MappingProxyType)


# ===========================================================================
# Section O: Sigmoid contract
# ===========================================================================

class TestSigmoidContract:
    """Section O: sigmoid produces values in (0, 1) for finite inputs."""

    def test_o01_sigmoid_zero_is_half(self):
        from rrm.inference import _sigmoid
        assert math.isclose(_sigmoid(0.0), 0.5, abs_tol=1e-9)

    def test_o02_sigmoid_large_positive_near_one(self):
        from rrm.inference import _sigmoid
        assert _sigmoid(100.0) > 0.9999

    def test_o03_sigmoid_large_negative_near_zero(self):
        from rrm.inference import _sigmoid
        assert _sigmoid(-100.0) < 0.0001

    def test_o04_sigmoid_monotonic(self):
        from rrm.inference import _sigmoid
        for x in [-10, -5, -1, 0, 1, 5, 10]:
            assert _sigmoid(x) < _sigmoid(x + 1)

    def test_o05_all_scores_in_open_interval(self, engine_with_stub):
        result = engine_with_stub.analyze(
            RrmInferenceRequest(review_text="test", review_id="r1")
        )
        for label in PRIMARY_LABELS:
            score = result.task_scores[label]
            assert 0.0 < score < 1.0


# ===========================================================================
# Section P: Strict JSON serialization
# ===========================================================================

class TestStrictJsonSerialization:
    """Section P: to_dict produces JSON-compatible output with NO fallback."""

    @staticmethod
    def _assert_json_only(value, path=""):
        """Recursively verify value contains only JSON-native types."""
        if value is None or isinstance(value, (bool, int, float)):
            return
        if isinstance(value, str):
            return
        if isinstance(value, dict):
            for k, v in value.items():
                TestStrictJsonSerialization._assert_json_only(v, f"{path}.{k}")
            return
        if isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                TestStrictJsonSerialization._assert_json_only(v, f"{path}[{i}]")
            return
        raise AssertionError(
            f"Non-JSON-native type {type(value).__name__} at {path}: {value!r}"
        )

    def test_p01_unavailable_result_strict_json(self, engine_deterministic_only, valid_request):
        result = engine_deterministic_only.analyze(valid_request)
        data = result.to_dict()
        # Must serialize with NO custom fallback
        json.dumps(data)
        # Must contain only JSON-native types
        self._assert_json_only(data)

    def test_p02_available_result_strict_json(self, engine_with_stub, valid_request):
        result = engine_with_stub.analyze(valid_request)
        data = result.to_dict()
        json.dumps(data)
        self._assert_json_only(data)

    def test_p03_no_mapping_proxy_leaks(self, engine_with_stub, valid_request):
        result = engine_with_stub.analyze(valid_request)
        data = result.to_dict()
        # task_scores must be a plain dict, not MappingProxyType
        assert isinstance(data["task_scores"], dict)
        assert not isinstance(data["task_scores"], types.MappingProxyType)

    def test_p04_no_dataclass_leaks(self, engine_deterministic_only, valid_request):
        result = engine_deterministic_only.analyze(valid_request)
        data = result.to_dict()
        from dataclasses import is_dataclass
        for key, value in data.items():
            if is_dataclass(value):
                raise AssertionError(
                    f"Dataclass leaked into to_dict() at key {key!r}"
                )

    def test_p05_no_enum_leaks(self, engine_deterministic_only, valid_request):
        result = engine_deterministic_only.analyze(valid_request)
        data = result.to_dict()
        from enum import Enum
        def _check_no_enum(obj, path):
            if isinstance(obj, Enum):
                raise AssertionError(f"Enum leaked at {path}: {obj!r}")
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _check_no_enum(v, f"{path}.{k}")
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    _check_no_enum(v, f"{path}[{i}]")
        _check_no_enum(data, "root")

    def test_p06_unavailable_to_dict_structure(self, engine_deterministic_only, valid_request):
        result = engine_deterministic_only.analyze(valid_request)
        data = result.to_dict()
        assert data["task_scores"] is None
        assert data["neural_availability"] == "unavailable"
        assert data["model_identity"] is None
        assert data["tokenizer_identity"] is None


# ===========================================================================
# Section Q: Public package surface
# ===========================================================================

class TestPackageSurface:
    """Section Q: __all__ contains exactly the expected names."""

    def test_q01_all_exports_exact(self):
        import rrm
        expected = {
            "RRM_RUNTIME_SCHEMA_VERSION",
            "NeuralAvailability",
            "RrmConfigurationError",
            "RrmInferenceEngine",
            "RrmInferenceError",
            "RrmInferenceRequest",
            "RrmInferenceResult",
        }
        assert set(rrm.__all__) == expected

    def test_q02_no_duplicates(self):
        import rrm
        assert len(rrm.__all__) == len(set(rrm.__all__))


# ===========================================================================
# Section R: Runtime schema version constant
# ===========================================================================

class TestSchemaVersion:
    """Section R: RRM_RUNTIME_SCHEMA_VERSION is exactly '1.0'."""

    def test_r01_schema_version_is_string(self):
        assert isinstance(RRM_RUNTIME_SCHEMA_VERSION, str)

    def test_r02_schema_version_exact_value(self):
        assert RRM_RUNTIME_SCHEMA_VERSION == "1.0"


# ===========================================================================
# Section S: Deterministic prechecks integration
# ===========================================================================

class TestDeterministicPrechecksIntegration:
    """Section S: inference layer correctly uses locked deterministic modules."""

    def test_s01_deterministic_prechecks_run_on_analyze(
        self, engine_deterministic_only, valid_request
    ):
        result = engine_deterministic_only.analyze(valid_request)
        assert result.deterministic_prechecks is not None

    def test_s02_fingerprint_computed(
        self, engine_deterministic_only, valid_request
    ):
        result = engine_deterministic_only.analyze(valid_request)
        assert result.deterministic_prechecks.fingerprint is not None
        assert isinstance(result.deterministic_prechecks.fingerprint, str)
        assert len(result.deterministic_prechecks.fingerprint) == 64  # SHA-256 hex

    def test_s03_review_id_passed_correctly(self, engine_deterministic_only):
        req = RrmInferenceRequest(review_text="text", review_id="my-review-id")
        result = engine_deterministic_only.analyze(req)
        assert result.deterministic_prechecks.review_id == "my-review-id"

    def test_s04_non_empty_text_produces_evidence(self, engine_deterministic_only):
        req = RrmInferenceRequest(review_text="Some college review text.", review_id="r1")
        result = engine_deterministic_only.analyze(req)
        evidence = result.deterministic_prechecks.text_evidence
        assert evidence.character_count > 0
        assert evidence.word_count > 0

    def test_s05_candidates_reach_prechecks(
        self, engine_deterministic_only, valid_request_with_candidates
    ):
        result = engine_deterministic_only.analyze(valid_request_with_candidates)
        assert result.deterministic_prechecks.best_similarity_match is not None


# ===========================================================================
# Section T: Original preserved tests
# ===========================================================================

class TestDeterministicOnly:
    """Tests for deterministic-only engine operation."""

    def test_valid_request_returns_result(
        self, engine_deterministic_only, valid_request
    ):
        result = engine_deterministic_only.analyze(valid_request)
        assert isinstance(result, RrmInferenceResult)

    def test_deterministic_evidence_present(
        self, engine_deterministic_only, valid_request
    ):
        result = engine_deterministic_only.analyze(valid_request)
        _assert_deterministic_present(result)

    def test_neural_availability_unavailable(
        self, engine_deterministic_only, valid_request
    ):
        result = engine_deterministic_only.analyze(valid_request)
        assert result.neural_availability == NeuralAvailability.UNAVAILABLE

    def test_task_scores_is_none(
        self, engine_deterministic_only, valid_request
    ):
        result = engine_deterministic_only.analyze(valid_request)
        assert result.task_scores is None

    def test_unavailable_reason_exact(
        self, engine_deterministic_only, valid_request
    ):
        result = engine_deterministic_only.analyze(valid_request)
        assert result.neural_unavailable_reason == "production_assets_unavailable"

    def test_model_identity_none(
        self, engine_deterministic_only, valid_request
    ):
        result = engine_deterministic_only.analyze(valid_request)
        assert result.model_identity is None

    def test_tokenizer_identity_none(
        self, engine_deterministic_only, valid_request
    ):
        result = engine_deterministic_only.analyze(valid_request)
        assert result.tokenizer_identity is None


class TestNeuralAvailable:
    """Tests for neural-enabled engine operation."""

    def test_fake_neural_predictor_accepted(self, engine_with_stub, valid_request):
        result = engine_with_stub.analyze(valid_request)
        assert result.neural_availability == NeuralAvailability.AVAILABLE

    def test_exactly_primary_labels_in_scores(self, engine_with_stub, valid_request):
        result = engine_with_stub.analyze(valid_request)
        assert list(result.task_scores.keys()) == list(PRIMARY_LABELS)

    def test_score_key_order_equals_primary_labels(
        self, engine_with_stub, valid_request
    ):
        result = engine_with_stub.analyze(valid_request)
        assert list(result.task_scores.keys()) == list(PRIMARY_LABELS)

    def test_zero_logits_produce_half_sigmoid(
        self, engine_with_stub, valid_request
    ):
        result = engine_with_stub.analyze(valid_request)
        for label in PRIMARY_LABELS:
            assert math.isclose(result.task_scores[label], 0.5, abs_tol=1e-9)

    def test_positive_negative_logits_correct_sigmoid(self):
        engine = RrmInferenceEngine(
            neural_predictor=_make_stub_predictor(_STUB_MIXED_LOGITS),
            model_identity="m",
            tokenizer_identity="t",
        )
        result = engine.analyze(RrmInferenceRequest(review_text="test", review_id="r-test"))
        # sigmoid(3.0) > 0.9
        assert result.task_scores["spam"] > 0.9
        # sigmoid(-1.0) < 0.35
        assert result.task_scores["deception"] < 0.35

    def test_all_scores_in_zero_one(self, engine_with_stub, valid_request):
        result = engine_with_stub.analyze(valid_request)
        for label in PRIMARY_LABELS:
            score = result.task_scores[label]
            assert 0.0 <= score <= 1.0


class TestTaskScoresImmutability:
    """Tests for task_scores immutability from engine-produced results."""

    def test_task_scores_is_mapping_proxy(self, engine_with_stub, valid_request):
        result = engine_with_stub.analyze(valid_request)
        assert isinstance(result.task_scores, types.MappingProxyType)

    def test_task_scores_rejects_assignment(self, engine_with_stub, valid_request):
        result = engine_with_stub.analyze(valid_request)
        with pytest.raises(TypeError):
            result.task_scores["spam"] = 0.99  # type: ignore[index]


class TestInputValidationOriginal:
    """Tests for request input validation."""

    def test_non_string_review_text_raises_typeerror(self):
        with pytest.raises(TypeError):
            RrmInferenceRequest(review_text=123, review_id="r1")

    def test_empty_review_text_raises_valueerror(self):
        with pytest.raises(ValueError):
            RrmInferenceRequest(review_text="", review_id="r1")

    def test_whitespace_only_raises_valueerror(self):
        with pytest.raises(ValueError):
            RrmInferenceRequest(review_text="   \t\n  ", review_id="r1")


class TestSimilarityCandidatesOriginal:
    """Tests for similarity candidate handling."""

    def test_malformed_container_rejected(self):
        with pytest.raises(TypeError):
            RrmInferenceRequest(
                review_text="test",
                review_id="r-test",
                similarity_candidates=("not_a_pair",),
            )

    def test_malformed_item_rejected(self):
        with pytest.raises(ValueError):
            RrmInferenceRequest(
                review_text="test",
                review_id="r-test",
                similarity_candidates=(("r1",),),
            )

    def test_valid_candidates_reach_prechecks(
        self, engine_deterministic_only, valid_request_with_candidates
    ):
        result = engine_deterministic_only.analyze(valid_request_with_candidates)
        _assert_deterministic_present(result)
        assert result.deterministic_prechecks.best_similarity_match is not None


class TestSerializationOriginal:
    """Tests for result serialization — strict JSON, no fallback."""

    def test_to_dict_json_serializable(self, engine_with_stub, valid_request):
        result = engine_with_stub.analyze(valid_request)
        data = result.to_dict()
        # Must serialize directly with NO custom fallback
        json.dumps(data)

    def test_to_dict_has_all_keys(self, engine_with_stub, valid_request):
        result = engine_with_stub.analyze(valid_request)
        data = result.to_dict()
        expected_keys = {
            "deterministic_prechecks",
            "task_scores",
            "neural_availability",
            "neural_unavailable_reason",
            "model_identity",
            "tokenizer_identity",
            "runtime_schema_version",
        }
        assert expected_keys.issubset(data.keys())

    def test_unavailable_to_dict(self, engine_deterministic_only, valid_request):
        result = engine_deterministic_only.analyze(valid_request)
        data = result.to_dict()
        assert data["task_scores"] is None
        assert data["neural_availability"] == "unavailable"


class TestForbiddenPatterns:
    """Tests ensuring forbidden API patterns are absent."""

    def test_no_load_checkpoint_import(self):
        import rrm.inference as mod
        assert not hasattr(mod, "from_checkpoint")
        assert not hasattr(mod, "load_checkpoint")

    def test_no_analyze_review_method(self):
        engine = RrmInferenceEngine()
        assert not hasattr(engine, "analyze_review")
        assert not hasattr(engine, "predict")
