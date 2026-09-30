"""RateMyCollezzz Review Risk Model (RRM) — Layer 3 package.

This package exposes the stable Layer-3 runtime inference contract.
Internal RRM modules remain directly importable for testing and
internal use, but only the names listed in ``__all__`` are part of
the guaranteed public API.

Architectural rule:
    RRM UNDERSTANDS. TRUST DECIDES.

Public API:
    RrmInferenceRequest       — immutable inference request
    RrmInferenceResult        — immutable inference result
    RrmInferenceEngine        — deterministic + neural inference engine
    NeuralAvailability        — neural runtime availability indicator
    RrmConfigurationError     — engine configuration errors
    RrmInferenceError         — neural runtime execution errors
    RRM_RUNTIME_SCHEMA_VERSION — runtime API schema version string
"""

from rrm.inference import (
    RRM_RUNTIME_SCHEMA_VERSION,
    NeuralAvailability,
    RrmConfigurationError,
    RrmInferenceEngine,
    RrmInferenceError,
    RrmInferenceRequest,
    RrmInferenceResult,
)

__all__ = (
    "RRM_RUNTIME_SCHEMA_VERSION",
    "NeuralAvailability",
    "RrmConfigurationError",
    "RrmInferenceEngine",
    "RrmInferenceError",
    "RrmInferenceRequest",
    "RrmInferenceResult",
)
