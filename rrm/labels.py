"""Neutral canonical label contract for the RateMyCollezzz RRM.

This module is the single permanent owner of the canonical label constants.
All RRM components and baselines import from here.

No baseline is the conceptual owner of these constants.

Research provenance:
    - Label taxonomy: PAPER-DERIVED / RRM EXPERIMENTAL DESIGN CHOICE
      (RESEARCH_CONTRACT.md section 4)
    - UNKNOWN_LABEL = -1: RRM EXPERIMENTAL DESIGN CHOICE (enables masking
      without conflating unknown with negative)
"""

from __future__ import annotations

from typing import Tuple

# ---------------------------------------------------------------------------
# Canonical label constants
# ---------------------------------------------------------------------------

#: Canonical label order.  This tuple is immutable and must not be reordered.
#: Every model output tensor, dataset column, metric index, and checkpoint
#: metadata field uses this exact ordering.
PRIMARY_LABELS: Tuple[str, ...] = (
    "spam",
    "deception",
    "toxicity",
    "advertising",
    "off_topic",
    "pii",
)

#: Value representing unknown / insufficient ground truth.
#: Must never be interpreted as a negative example.
UNKNOWN_LABEL: int = -1

#: Number of primary classification labels.
NUM_PRIMARY_LABELS: int = len(PRIMARY_LABELS)

# ---------------------------------------------------------------------------
# Valid label value set
# ---------------------------------------------------------------------------

#: Set of all valid label values for input validation.
_VALID_LABEL_VALUES = frozenset({0, 1, UNKNOWN_LABEL})
