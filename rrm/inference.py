"""RRM 3.11 — Runtime inference contract and engine.

This module defines the stable Layer-3 public interface that Layer 2
(Django Platform) and Layer 4 (Trust) consume.  It orchestrates existing
locked RRM components without modifying them.

Architectural rule:
    RRM UNDERSTANDS. TRUST DECIDES.

    This module returns evidence and uncalibrated model scores only.
    It does NOT make moderation decisions, assign risk labels, or
    implement Trust policy.

Public API:
    RRM_RUNTIME_SCHEMA_VERSION
    NeuralAvailability
    RrmConfigurationError
    RrmInferenceError
    RrmInferenceRequest
    RrmInferenceResult
    RrmInferenceEngine

Evidence vs. decision:
    - exact duplicate flag is evidence only; it is NOT spam, NOT a
      moderation decision, and MUST NOT trigger automatic removal.
    - high similarity is a signal; it MUST NOT be treated as confirmed
      malicious behavior without Trust-layer policy evaluation.
    - all similarity scores are RRM evidence; the Trust layer makes
      final policy decisions.
"""

from __future__ import annotations

import math
import numbers
import types
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence, Tuple

from rrm.deterministic_prechecks import DeterministicPrecheckResult, run_deterministic_prechecks
from rrm.labels import PRIMARY_LABELS, NUM_PRIMARY_LABELS

# ---------------------------------------------------------------------------
# Runtime schema version
# ---------------------------------------------------------------------------

#: Stable runtime Layer-3 API schema version.
#: This is independent of:
#:   - RRM project phase number (e.g., 3.11)
#:   - Scientific evaluation protocol_version
#:   - CheckpointMetadata.schema_version
#: The exact string "1.0" is the only valid value for this constant.
RRM_RUNTIME_SCHEMA_VERSION: str = "1.0"

# ---------------------------------------------------------------------------
# Availability type
# ---------------------------------------------------------------------------


class NeuralAvailability:
    """Minimal two-state neural runtime availability indicator.

    Attributes
    ----------
    AVAILABLE : str
        Neural runtime is loaded and inference succeeded.
    UNAVAILABLE : str
        Neural runtime is not configured or not available.
    """

    AVAILABLE: str = "available"
    UNAVAILABLE: str = "unavailable"


# ---------------------------------------------------------------------------
# Public exceptions
# ---------------------------------------------------------------------------


class RrmConfigurationError(Exception):
    """Raised when the RRM inference engine has invalid configuration.

    This covers programmer/initialization errors such as:
    - neural predictor supplied without model_identity
    - neural predictor supplied without tokenizer_identity
    - non-callable neural_predictor
    - non-string model_identity or tokenizer_identity

    These are NOT normal runtime states.  The caller must fix configuration.
    """


class RrmInferenceError(Exception):
    """Raised when an unexpected neural runtime execution failure occurs.

    This covers unexpected failures during neural inference such as:
    - model runtime exceptions
    - invalid predictor output (wrong length, wrong types, NaN/Inf)
    - OOM or other resource failures

    This is distinct from the expected UNAVAILABLE state (no neural
    runtime configured).  The application layer decides how to handle
    this exception.
    """


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RrmInferenceRequest:
    """Immutable request for RRM inference.

    Attributes
    ----------
    review_text : str
        Raw review text to analyze.  Required.  Must be non-empty
        and non-whitespace-only.
    review_id : str
        Identifier for fingerprint identity and similarity self-exclusion.
        Required.  Must be non-empty and non-whitespace-only.
    similarity_candidates : tuple[tuple[str, str], ...]
        Optional collection of (review_id, review_text) pairs for
        near-duplicate similarity evidence.  Default is empty (no
        similarity check).  Caller-owned mutable inputs are deep-copied
        on construction.

    Raises
    ------
    TypeError
        If *review_text* or *review_id* is not str.
    ValueError
        If *review_text* or *review_id* is empty or whitespace-only.
    TypeError
        If a candidate is not a tuple/list of exactly 2 strings.
    """

    review_text: str
    review_id: str
    similarity_candidates: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # --- review_text ---
        if not isinstance(self.review_text, str):
            raise TypeError(
                f"RrmInferenceRequest.review_text must be str, "
                f"got {type(self.review_text).__name__}"
            )
        if not self.review_text.strip():
            raise ValueError(
                "RrmInferenceRequest.review_text must not be empty "
                "or whitespace-only"
            )
        # --- review_id ---
        if not isinstance(self.review_id, str):
            raise TypeError(
                f"RrmInferenceRequest.review_id must be str, "
                f"got {type(self.review_id).__name__}"
            )
        if not self.review_id.strip():
            raise ValueError(
                "RrmInferenceRequest.review_id must not be empty "
                "or whitespace-only"
            )
        # --- similarity_candidates: validate, deep-copy, normalize ---
        normalized: list[Tuple[str, str]] = []
        for idx, candidate in enumerate(self.similarity_candidates):
            if not isinstance(candidate, (tuple, list)):
                raise TypeError(
                    f"Similarity candidate at index {idx} must be a tuple/list, "
                    f"got {type(candidate).__name__}: {candidate!r}"
                )
            if len(candidate) != 2:
                raise ValueError(
                    f"Similarity candidate at index {idx} must have exactly "
                    f"2 items (review_id, review_text), got {len(candidate)}: "
                    f"{candidate!r}"
                )
            c_id, c_text = candidate
            if not isinstance(c_id, str):
                raise TypeError(
                    f"Similarity candidate review_id at index {idx} must be "
                    f"str, got {type(c_id).__name__}: {c_id!r}"
                )
            if not isinstance(c_text, str):
                raise TypeError(
                    f"Similarity candidate review_text at index {idx} must be "
                    f"str, got {type(c_text).__name__}: {c_text!r}"
                )
            # Deep-copy: convert to a NEW tuple so caller mutations
            # of the original list/inner list do not leak into this
            # frozen request.
            normalized.append((c_id, c_text))
        # Replace the caller-owned collection with a new immutable tuple.
        # Must use object.__setattr__ because the dataclass is frozen.
        object.__setattr__(self, "similarity_candidates", tuple(normalized))


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RrmInferenceResult:
    """Immutable RRM inference result.

    This object contains evidence only.  It does NOT contain moderation
    decisions, risk labels, or Trust policy outcomes.

    Attributes
    ----------
    deterministic_prechecks : DeterministicPrecheckResult
        Deterministic evidence (always present regardless of neural state).
    task_scores : Mapping[str, float] or None
        Canonical six label-keyed uncalibrated sigmoid scores in [0, 1],
        or None when neural runtime is unavailable.  Keys match
        ``PRIMARY_LABELS`` order.  Read-only mapping.
    neural_availability : str
        ``NeuralAvailability.AVAILABLE`` or ``NeuralAvailability.UNAVAILABLE``.
    neural_unavailable_reason : str or None
        Machine-readable reason when neural runtime is unavailable.
        None when available.  Required (non-empty) when unavailable.
    model_identity : str or None
        Explicit model/checkpoint identity string, or None.
    tokenizer_identity : str or None
        Explicit tokenizer identity string, or None.
    runtime_schema_version : str
        Runtime schema version.  Must equal ``RRM_RUNTIME_SCHEMA_VERSION``
        ("1.0").

    Raises
    ------
    TypeError
        If *deterministic_prechecks* is not a
        ``DeterministicPrecheckResult``.
    ValueError
        If any field fails the invariant rules described above.
    """

    deterministic_prechecks: DeterministicPrecheckResult
    task_scores: Optional[Mapping[str, float]]
    neural_availability: str
    neural_unavailable_reason: Optional[str]
    model_identity: Optional[str]
    tokenizer_identity: Optional[str]
    runtime_schema_version: str = RRM_RUNTIME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        # --- runtime_schema_version ---
        if self.runtime_schema_version != RRM_RUNTIME_SCHEMA_VERSION:
            raise ValueError(
                f"runtime_schema_version must be {RRM_RUNTIME_SCHEMA_VERSION!r}, "
                f"got {self.runtime_schema_version!r}"
            )
        # --- neural_availability must be a known value ---
        if self.neural_availability not in (
            NeuralAvailability.AVAILABLE,
            NeuralAvailability.UNAVAILABLE,
        ):
            raise ValueError(
                f"neural_availability must be NeuralAvailability.AVAILABLE or "
                f"NeuralAvailability.UNAVAILABLE, got {self.neural_availability!r}"
            )
        # --- deterministic_prechecks type ---
        if not isinstance(self.deterministic_prechecks, DeterministicPrecheckResult):
            raise TypeError(
                f"deterministic_prechecks must be DeterministicPrecheckResult, "
                f"got {type(self.deterministic_prechecks).__name__}"
            )
        # --- branch-specific validation ---
        if self.neural_availability == NeuralAvailability.AVAILABLE:
            _validate_available_result(self)
        else:
            _validate_unavailable_result(self)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert to a JSON-compatible dictionary.

        Returns
        -------
        dict
            JSON-compatible representation.  No PyTorch tensors, no
            MappingProxyType, no custom Enum instances, no custom
            dataclass instances.
        """
        deterministic_prechecks_dict = _deterministic_prechecks_to_dict(
            self.deterministic_prechecks
        )
        task_scores_dict: Optional[dict] = None
        if self.task_scores is not None:
            task_scores_dict = dict(self.task_scores)

        return {
            "deterministic_prechecks": deterministic_prechecks_dict,
            "task_scores": task_scores_dict,
            "neural_availability": self.neural_availability,
            "neural_unavailable_reason": self.neural_unavailable_reason,
            "model_identity": self.model_identity,
            "tokenizer_identity": self.tokenizer_identity,
            "runtime_schema_version": self.runtime_schema_version,
        }


# ---------------------------------------------------------------------------
# Result validation helpers
# ---------------------------------------------------------------------------


def _validate_available_result(result: RrmInferenceResult) -> None:
    """Validate invariants for AVAILABLE (neural) result."""
    # task_scores required
    if result.task_scores is None:
        raise ValueError(
            "task_scores must not be None when neural_availability "
            "is AVAILABLE"
        )
    # task_scores must be a Mapping
    if not isinstance(result.task_scores, Mapping):
        raise TypeError(
            f"task_scores must be a Mapping, "
            f"got {type(result.task_scores).__name__}"
        )
    # Keys must exactly match PRIMARY_LABELS in order
    task_keys = list(result.task_scores.keys())
    if task_keys != list(PRIMARY_LABELS):
        raise ValueError(
            f"task_scores keys must exactly equal PRIMARY_LABELS in order: "
            f"{list(PRIMARY_LABELS)}, got {task_keys}"
        )
    # Every score must be a finite real number, not bool, in [0, 1]
    for key, value in result.task_scores.items():
        _validate_score_value(key, value)
    # Deep-copy into a new dict and wrap in MappingProxyType so caller
    # mutations of the original mapping cannot affect this frozen result.
    copied_scores = {
        label: float(result.task_scores[label])
        for label in PRIMARY_LABELS
    }
    object.__setattr__(
        result,
        "task_scores",
        types.MappingProxyType(copied_scores),
    )
    # unavailable_reason must be None
    if result.neural_unavailable_reason is not None:
        raise ValueError(
            "neural_unavailable_reason must be None when "
            "neural_availability is AVAILABLE"
        )
    # model_identity: str, non-empty after strip
    if not isinstance(result.model_identity, str):
        raise TypeError(
            f"model_identity must be str, "
            f"got {type(result.model_identity).__name__}"
        )
    if not result.model_identity.strip():
        raise ValueError(
            "model_identity must be non-empty when "
            "neural_availability is AVAILABLE"
        )
    # tokenizer_identity: str, non-empty after strip
    if not isinstance(result.tokenizer_identity, str):
        raise TypeError(
            f"tokenizer_identity must be str, "
            f"got {type(result.tokenizer_identity).__name__}"
        )
    if not result.tokenizer_identity.strip():
        raise ValueError(
            "tokenizer_identity must be non-empty when "
            "neural_availability is AVAILABLE"
        )


def _validate_unavailable_result(result: RrmInferenceResult) -> None:
    """Validate invariants for UNAVAILABLE (deterministic-only) result."""
    # task_scores must be None
    if result.task_scores is not None:
        raise ValueError(
            "task_scores must be None when neural_availability "
            "is UNAVAILABLE"
        )
    # neural_unavailable_reason must be non-empty str
    if not isinstance(result.neural_unavailable_reason, str):
        raise TypeError(
            f"neural_unavailable_reason must be str, "
            f"got {type(result.neural_unavailable_reason).__name__}"
        )
    if not result.neural_unavailable_reason.strip():
        raise ValueError(
            "neural_unavailable_reason must be non-empty when "
            "neural_availability is UNAVAILABLE"
        )
    # model_identity must be None
    if result.model_identity is not None:
        raise ValueError(
            "model_identity must be None when neural_availability "
            "is UNAVAILABLE"
        )
    # tokenizer_identity must be None
    if result.tokenizer_identity is not None:
        raise ValueError(
            "tokenizer_identity must be None when neural_availability "
            "is UNAVAILABLE"
        )


def _validate_score_value(label: str, value: object) -> None:
    """Validate that a task score is a finite real number, not bool, in [0, 1]."""
    if isinstance(value, bool):
        raise TypeError(
            f"task score for {label!r} must be a real number, not bool"
        )
    if not isinstance(value, numbers.Real):
        raise TypeError(
            f"task score for {label!r} must be a real number, "
            f"got {type(value).__name__}: {value!r}"
        )
    if math.isnan(float(value)) or math.isinf(float(value)):
        raise ValueError(
            f"task score for {label!r} must be finite, got {value!r}"
        )
    if not (0.0 <= float(value) <= 1.0):
        raise ValueError(
            f"task score for {label!r} must be in [0.0, 1.0], got {value!r}"
        )


# ---------------------------------------------------------------------------
# Neural logit processing
# ---------------------------------------------------------------------------


def _logits_to_scores(raw_logits: object) -> Mapping[str, float]:
    """Validate and convert raw predictor output to label-keyed scores.

    Steps
    -----
    1. Attempt to materialize output into a sequence.
    2. Validate exactly 6 logits.
    3. Validate each logit: real numeric, not bool, finite.
    4. Apply sigmoid.
    5. Return ordered mapping keyed by PRIMARY_LABELS, wrapped in
       MappingProxyType for immutability.

    Parameters
    ----------
    raw_logits : object
        Raw predictor output.

    Returns
    -------
    Mapping[str, float]
        Label-keyed sigmoid scores in canonical PRIMARY_LABELS order.
        Returned as a read-only MappingProxyType.

    Raises
    ------
    RrmInferenceError
        If output cannot be materialized, has wrong length, or contains
        invalid values.
    """
    # Step 1: materialize output into a list
    try:
        logits_list = list(raw_logits)
    except TypeError as exc:
        raise RrmInferenceError(
            "Neural predictor output must be iterable; "
            f"got {type(raw_logits).__name__}"
        ) from exc

    # Step 2: validate exact count
    if len(logits_list) != NUM_PRIMARY_LABELS:
        raise RrmInferenceError(
            f"Neural predictor must return exactly {NUM_PRIMARY_LABELS} logits "
            f"(one per label in PRIMARY_LABELS), got {len(logits_list)}: "
            f"{logits_list}"
        )

    # Step 3: validate each logit
    for idx, logit in enumerate(logits_list):
        if isinstance(logit, bool):
            raise RrmInferenceError(
                f"Neural logit at index {idx} must be a real number, not bool "
                f"(got {logit!r})"
            )
        if not isinstance(logit, numbers.Real):
            raise RrmInferenceError(
                f"Neural logit at index {idx} must be a real number, "
                f"got {type(logit).__name__}: {logit!r}"
            )
        if math.isnan(float(logit)) or math.isinf(float(logit)):
            raise RrmInferenceError(
                f"Neural logit at index {idx} must be finite, got {logit!r}"
            )

    # Step 4: sigmoid and step 5: build mapping
    scores = {}
    for label, logit in zip(PRIMARY_LABELS, logits_list):
        scores[label] = _sigmoid(float(logit))

    return types.MappingProxyType(scores)


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

#: Type for a neural predictor callable.
#: Accepts raw review text, returns exactly 6 raw logits as floats
#: in canonical PRIMARY_LABELS order.
NeuralPredictor = Callable[[str], Sequence[float]]


class RrmInferenceEngine:
    """RRM inference engine orchestrating deterministic and neural analysis.

    The engine owns orchestration only.  It does NOT own checkpoint/model
    loading.  Neural runtime components are supplied via dependency
    injection at construction time.

    Parameters
    ----------
    neural_predictor : callable, optional
        A callable that accepts raw review text (str) and returns exactly
        6 raw logits as floats in canonical PRIMARY_LABELS order.
        When None, the engine operates in deterministic-only mode.
    model_identity : str, optional
        Explicit model/checkpoint identity string.  Required when
        *neural_predictor* is supplied.  Must be non-empty.
    tokenizer_identity : str, optional
        Explicit tokenizer identity string.  Required when
        *neural_predictor* is supplied.  Must be non-empty.

    Raises
    ------
    RrmConfigurationError
        If *neural_predictor* is supplied but *model_identity* or
        *tokenizer_identity* is missing, empty, or wrong type.
        If *neural_predictor* is not callable.
        If *model_identity* or *tokenizer_identity* is supplied without
        *neural_predictor*.
    """

    def __init__(
        self,
        neural_predictor: Optional[NeuralPredictor] = None,
        model_identity: Optional[str] = None,
        tokenizer_identity: Optional[str] = None,
    ) -> None:
        if neural_predictor is not None:
            # Validate predictor is callable
            if not callable(neural_predictor):
                raise RrmConfigurationError(
                    f"neural_predictor must be callable, "
                    f"got {type(neural_predictor).__name__}"
                )
            # Validate model_identity type and non-empty
            if not isinstance(model_identity, str) or not model_identity.strip():
                raise RrmConfigurationError(
                    "model_identity must be a non-empty string when "
                    "neural_predictor is supplied"
                )
            # Validate tokenizer_identity type and non-empty
            if not isinstance(tokenizer_identity, str) or not tokenizer_identity.strip():
                raise RrmConfigurationError(
                    "tokenizer_identity must be a non-empty string when "
                    "neural_predictor is supplied"
                )
        else:
            # Deterministic-only mode: identities must be None
            if model_identity is not None:
                raise RrmConfigurationError(
                    "model_identity must be None when no neural_predictor "
                    "is supplied"
                )
            if tokenizer_identity is not None:
                raise RrmConfigurationError(
                    "tokenizer_identity must be None when no neural_predictor "
                    "is supplied"
                )

        self._neural_predictor = neural_predictor
        self._model_identity = model_identity.strip() if isinstance(model_identity, str) else None
        self._tokenizer_identity = (
            tokenizer_identity.strip() if isinstance(tokenizer_identity, str) else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, request: RrmInferenceRequest) -> RrmInferenceResult:
        """Run full RRM analysis on a review.

        Parameters
        ----------
        request : RrmInferenceRequest
            Immutable inference request.  Validated on construction.

        Returns
        -------
        RrmInferenceResult
            Immutable result containing deterministic evidence and
            optional neural task scores.

        Raises
        ------
        TypeError
            If *request* is not an RrmInferenceRequest.
        RrmInferenceError
            If an unexpected neural runtime failure occurs.
        """
        if not isinstance(request, RrmInferenceRequest):
            raise TypeError(
                f"analyze expects RrmInferenceRequest, "
                f"got {type(request).__name__}"
            )

        # --- Step 1: Deterministic prechecks (always runs) ---
        deterministic_result = run_deterministic_prechecks(
            review_id=request.review_id,
            review_text=request.review_text,
            candidates=request.similarity_candidates,
        )

        # --- Step 2: Neural inference (if available) ---
        if self._neural_predictor is None:
            return RrmInferenceResult(
                deterministic_prechecks=deterministic_result,
                task_scores=None,
                neural_availability=NeuralAvailability.UNAVAILABLE,
                neural_unavailable_reason="production_assets_unavailable",
                model_identity=None,
                tokenizer_identity=None,
                runtime_schema_version=RRM_RUNTIME_SCHEMA_VERSION,
            )

        # Neural predictor is available — run inference
        try:
            raw_logits = self._neural_predictor(request.review_text)
        except Exception as exc:
            raise RrmInferenceError(
                "Neural predictor raised an unexpected exception during "
                "inference"
            ) from exc

        # Validate and transform logits
        task_scores = _logits_to_scores(raw_logits)

        return RrmInferenceResult(
            deterministic_prechecks=deterministic_result,
            task_scores=task_scores,
            neural_availability=NeuralAvailability.AVAILABLE,
            neural_unavailable_reason=None,
            model_identity=self._model_identity,
            tokenizer_identity=self._tokenizer_identity,
            runtime_schema_version=RRM_RUNTIME_SCHEMA_VERSION,
        )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _deterministic_prechecks_to_dict(
    result: DeterministicPrecheckResult,
) -> dict:
    """Convert a DeterministicPrecheckResult to a JSON-compatible dict."""
    text_evidence = result.text_evidence
    text_evidence_dict = {
        "character_count": text_evidence.character_count,
        "non_whitespace_character_count": text_evidence.non_whitespace_character_count,
        "word_count": text_evidence.word_count,
        "digit_count": text_evidence.digit_count,
        "uppercase_character_count": text_evidence.uppercase_character_count,
        "exclamation_count": text_evidence.exclamation_count,
        "question_count": text_evidence.question_count,
        "repeated_character_run_count": text_evidence.repeated_character_run_count,
        "max_character_run_length": text_evidence.max_character_run_length,
        "email_count": text_evidence.email_count,
        "phone_count": text_evidence.phone_count,
        "url_count": text_evidence.url_count,
    }

    pii_matches_list = [
        {
            "kind": m.kind,
            "matched_text": m.matched_text,
            "start": m.start,
            "end": m.end,
        }
        for m in result.pii_matches
    ]

    best_match_dict = None
    if result.best_similarity_match is not None:
        bm = result.best_similarity_match
        best_match_dict = {
            "review_id": bm.review_id,
            "score": bm.score,
            "exact_duplicate": bm.exact_duplicate,
        }

    return {
        "review_id": result.review_id,
        "fingerprint": result.fingerprint,
        "text_evidence": text_evidence_dict,
        "pii_matches": pii_matches_list,
        "best_similarity_match": best_match_dict,
    }
