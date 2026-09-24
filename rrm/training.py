"""Neural training utilities for the RRM (RRM 3.9).

Implements neutral, baseline-independent utilities for:

- Masked BCE loss with UNKNOWN label exclusion
- Gradient accumulation with correct supervised-position normalization
- Optimizer-step boundary with AMP support

Research provenance:
    - BCEWithLogitsLoss: PAPER-DERIVED (standard numerically stable
      binary cross-entropy; Kingma & LeCun, 2015; PyTorch implementation)
    - Masked UNKNOWN exclusion: RRM EXPERIMENTAL DESIGN CHOICE
    - Global supervised-position normalization: RRM EXPERIMENTAL DESIGN CHOICE
      (gradient contribution proportional to total supervised positions,
      not per-microbatch average)
    - AMP + GradScaler: STANDARD ENGINEERING PRACTICE (PyTorch AMP)
    - Gradient clipping after normalization: STANDARD ENGINEERING PRACTICE

Architecture rule:
    RRM UNDERSTANDS. TRUST DECIDES.

This module produces training math only.
It does NOT make moderation decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from rrm.labels import NUM_PRIMARY_LABELS, UNKNOWN_LABEL


# ---------------------------------------------------------------------------
# Masked BCE loss result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaskedLossResult:
    """Result of masked BCE computation.

    Attributes
    ----------
    loss_sum : torch.Tensor
        Sum of BCE losses over all supervised positions.
        Scalar tensor, differentiable.
    supervised_count : int
        Number of supervised (non-UNKNOWN) label positions.
    """

    loss_sum: torch.Tensor
    supervised_count: int


# ---------------------------------------------------------------------------
# Masked BCE loss
# ---------------------------------------------------------------------------


def masked_bce_sum_and_count(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Optional[MaskedLossResult]:
    """Compute masked BCE loss sum and supervised position count.

    UNKNOWN_LABEL (-1) positions are excluded from loss computation.
    Targets for UNKNOWN positions are replaced with a safe dummy value (0)
    before BCE to satisfy ``binary_cross_entropy_with_logits`` range
    requirements, then zeroed out via the supervision mask.

    The caller normalizes by ``supervised_count`` at optimizer-boundary
    time, NOT per-microbatch.  This produces correct gradient weighting
    when UNKNOWN density varies across microbatches.

    Parameters
    ----------
    logits : torch.Tensor
        Model output logits.  Shape [B, 6], dtype floating.
    labels : torch.Tensor
        Label tensor.  Shape [B, 6], dtype torch.long.
        Valid values: -1 (UNKNOWN), 0 (negative), 1 (positive).

    Returns
    -------
    MaskedLossResult or None
        ``MaskedLossResult`` if at least one position is supervised.
        ``None`` if all positions are UNKNOWN (caller should skip this
        microbatch without optimizer/scheduler step).
    """
    # --- Input validation ---
    if logits.dim() != 2:
        raise ValueError(
            f"logits must be rank-2 [B, 6], got rank-{logits.dim()}"
        )
    if labels.dim() != 2:
        raise ValueError(
            f"labels must be rank-2 [B, 6], got rank-{labels.dim()}"
        )
    if logits.shape != labels.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} must match "
            f"labels shape {tuple(labels.shape)}"
        )
    if logits.shape[1] != NUM_PRIMARY_LABELS:
        raise ValueError(
            f"labels second dimension must be {NUM_PRIMARY_LABELS} "
            f"(canonical six-task contract), got {logits.shape[1]}"
        )
    if logits.size(0) == 0:
        raise ValueError("batch dimension must be > 0")
    if not logits.is_floating_point():
        raise ValueError(
            f"logits must be floating dtype, got {logits.dtype}"
        )
    if labels.dtype != torch.long:
        raise ValueError(
            f"labels must be torch.long, got {labels.dtype}"
        )
    if logits.device != labels.device:
        raise ValueError(
            f"logits and labels must be on the same device, "
            f"got {logits.device} vs {labels.device}"
        )

    # Check valid label values (UNKNOWN_LABEL is permanently -1)
    unique_labels = labels.unique()
    invalid = unique_labels[
        (unique_labels != UNKNOWN_LABEL)
        & (unique_labels != 0)
        & (unique_labels != 1)
    ]
    if invalid.numel() > 0:
        raise ValueError(
            f"labels contain invalid values {invalid.tolist()}. "
            f"Valid values are {{{UNKNOWN_LABEL}, 0, 1}}."
        )

    supervision_mask = (labels != UNKNOWN_LABEL)  # [B, 6], bool

    # Build safe targets for BCE: UNKNOWN positions get 0.0
    safe_targets = torch.where(
        supervision_mask,
        labels,
        torch.zeros_like(labels),
    ).to(dtype=logits.dtype)

    # Unreduced BCE
    loss_matrix = F.binary_cross_entropy_with_logits(
        logits,
        safe_targets,
        reduction="none",
    )  # [B, num_labels]

    mask_float = supervision_mask.to(dtype=loss_matrix.dtype)
    supervised_count = int(mask_float.sum().item())

    if supervised_count == 0:
        return None

    loss_sum = (loss_matrix * mask_float).sum()
    return MaskedLossResult(loss_sum=loss_sum, supervised_count=supervised_count)


# ---------------------------------------------------------------------------
# Gradient accumulation state
# ---------------------------------------------------------------------------


class GradientAccumulator:
    """Tracks supervised loss sum and count across microbatches.

    Used with gradient accumulation to ensure correct normalization
    when UNKNOWN density varies across microbatches.

    Usage::

        accumulator = GradientAccumulator()

        for microbatch in microbatches:
            result = masked_bce_sum_and_count(logits, labels)
            if result is None:
                continue
            result.loss_sum.backward()
            accumulator.add(result.supervised_count)

        if accumulator.count > 0:
            optimizer_step(accumulator.count)
            accumulator.reset()
    """

    def __init__(self) -> None:
        self._count: int = 0

    def add(self, supervised_count: int) -> None:
        """Record supervised positions from a contributing microbatch."""
        if supervised_count <= 0:
            raise ValueError(
                f"supervised_count must be > 0, got {supervised_count}"
            )
        self._count += supervised_count

    @property
    def count(self) -> int:
        """Total supervised positions accumulated so far."""
        return self._count

    @property
    def contributing(self) -> bool:
        """Whether any microbatches contributed to this group."""
        return self._count > 0

    def reset(self) -> None:
        """Reset accumulator for next accumulation group."""
        self._count = 0


# ---------------------------------------------------------------------------
# Optimizer-step boundary
# ---------------------------------------------------------------------------


def optimizer_step_with_accumulation(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    accumulator: GradientAccumulator,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    max_grad_norm: Optional[float] = None,
    use_amp: bool = False,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> bool:
    """Perform optimizer step with correct gradient normalization.

    Sequence:

    1. If AMP is enabled, call ``scaler.unscale_(optimizer)``.
    2. Divide accumulated gradients by ``accumulator.count``.
    3. Clip gradients if ``max_grad_norm`` is provided.
    4. Optimizer (and scaler) step.
    5. Scheduler step only if optimizer actually stepped.
    6. Zero gradients.

    If ``accumulator.count`` is 0, no optimizer step is performed and
    gradients are zeroed to avoid stale state.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer whose gradients to normalize and step.
    model : torch.nn.Module
        Model whose parameter gradients are normalized.
    accumulator : GradientAccumulator
        Accumulated supervised position count.
    scheduler : LambdaLR or None
        Learning rate scheduler.  Stepped only if optimizer updates.
    max_grad_norm : float or None
        Maximum gradient norm for clipping.  None disables clipping.
    use_amp : bool
        Whether AMP is in use.
    scaler : GradScaler or None
        GradScaler instance if AMP is enabled.

    Returns
    -------
    bool
        True if an optimizer step was performed, False otherwise.
    """
    if not accumulator.contributing:
        # No supervised positions — zero gradients and skip
        optimizer.zero_grad(set_to_none=True)
        return False

    # Step 1: Unscale gradients if AMP is enabled.
    if use_amp and scaler is not None:
        scaler.unscale_(optimizer)

    # Step 2: Divide by supervised count BEFORE clipping.
    norm = float(accumulator.count)
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.div_(norm)

    # Step 3: Clip the NORMALIZED gradients (exactly once).
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # Step 4: Optimizer step.
    if use_amp and scaler is not None:
        scaler.step(optimizer)
        scaler.update()
        stepped = True
    else:
        optimizer.step()
        stepped = True

    # Step 5: Scheduler step only if optimizer actually stepped.
    if stepped and scheduler is not None:
        scheduler.step()

    # Step 6: Zero gradients.
    optimizer.zero_grad(set_to_none=True)
    accumulator.reset()
    return True
