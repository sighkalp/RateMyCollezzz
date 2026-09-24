"""Multi-task prediction heads and top-level model for the RRM (RRM 3.9).

Implements the supervised classification layer on top of the fused
[B, 448] representation, and a top-level model that composes the full
pipeline: encoder → character branch → fusion → heads.

Research provenance:
    - Independent binary logits: RRM EXPERIMENTAL DESIGN CHOICE
    - Linear(448, 6) classifier: RRM EXPERIMENTAL DESIGN CHOICE
    - Initializer Normal(0, 0.02) / zero bias: STANDARD ENGINEERING PRACTICE
    - No classifier dropout: RRM EXPERIMENTAL DESIGN CHOICE
      (fused representation already regularized)
    - No sigmoid/softmax in forward: STANDARD ENGINEERING PRACTICE
      (BCEWithLogitsLoss handles numerical stability)

Architecture rule:
    RRM UNDERSTANDS. TRUST DECIDES.

This module produces logits / evidence only.
It does NOT make moderation decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from rrm.encoder import RmcEncoder, RmcEncoderConfig
from rrm.character_branch import RmcCharacterBranch, RmcCharacterConfig
from rrm.fusion import RmcFusion, RmcFusionOutput
from rrm.labels import NUM_PRIMARY_LABELS


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RmcMultiTaskOutput:
    """Immutable output from the multi-task model forward pass.

    Attributes
    ----------
    logits : torch.Tensor
        Raw classification logits.  Shape [B, 6], dtype floating
        (follows autocast; typically float32 on CPU, float16/bfloat16
        under AMP on CUDA).
    """

    logits: torch.Tensor


@dataclass(frozen=True)
class RmcMultiTaskFeatures:
    """Intermediate representations from the full pipeline.

    Used by tests and ablation analysis.  Not part of the ordinary
    deployment forward pass.
    """

    semantic_repr: torch.Tensor   # [B, 384]
    character_repr: torch.Tensor  # [B, 64]
    fused_repr: torch.Tensor      # [B, 448]
    logits: torch.Tensor          # [B, 6]


# ---------------------------------------------------------------------------
# Multi-task heads
# ---------------------------------------------------------------------------


class RmcMultiTaskHeads(nn.Module):
    """Six independent binary classification heads.

    Permanent, non-configurable architecture:

        Linear(in_features=448, out_features=6)

    This is a single weight matrix.  Each row corresponds to one
    canonical label in ``PRIMARY_LABELS`` order.  Each output is an
    independent binary logit.

    The architecture is locked.  No constructor arguments alter
    dimensions.
    """

    FUSED_DIM = 448
    NUM_LABELS = NUM_PRIMARY_LABELS  # 6

    def __init__(self) -> None:
        super().__init__()
        self.input_dim = self.FUSED_DIM
        self.num_labels = self.NUM_LABELS

        self.linear = nn.Linear(self.FUSED_DIM, self.NUM_LABELS)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights per project convention."""
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, fused_repr: torch.Tensor) -> torch.Tensor:
        """Compute classification logits.

        Parameters
        ----------
        fused_repr : torch.Tensor
            Fused representation.  Shape [B, 448], dtype floating.

        Returns
        -------
        torch.Tensor
            Logits.  Shape [B, 6], dtype follows input.
        """
        self._validate_inputs(fused_repr)
        logits = self.linear(fused_repr)
        return logits

    def _validate_inputs(self, fused_repr: torch.Tensor) -> None:
        if fused_repr.dim() != 2:
            raise ValueError(
                f"fused_repr must be rank-2 [B, D], got rank-{fused_repr.dim()}"
            )
        if fused_repr.shape[1] != self.FUSED_DIM:
            raise ValueError(
                f"fused_repr last dim must be {self.FUSED_DIM}, "
                f"got {fused_repr.shape[1]}"
            )
        if fused_repr.size(0) == 0:
            raise ValueError("batch dimension must be > 0")
        if not fused_repr.is_floating_point():
            raise ValueError(
                f"fused_repr must be floating dtype, got {fused_repr.dtype}"
            )

    def count_parameters(self) -> int:
        """Return trainable parameter count."""
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Top-level multi-task model
# ---------------------------------------------------------------------------


class RmcMultiTaskModel(nn.Module):
    """Full RRM multi-task model composing all representation components.

    Pipeline:

        input_ids + attention_mask
            → RmcEncoder
            → semantic_repr [B, 384]

        byte_ids + byte_attention_mask
            → RmcCharacterBranch
            → character_repr [B, 64]

        semantic_repr + character_repr
            → RmcFusion
            → fused_repr [B, 448]

        fused_repr
            → RmcMultiTaskHeads
            → logits [B, 6]

    Parameters
    ----------
    encoder_cfg : RmcEncoderConfig
        Configuration for the semantic encoder.
    character_cfg : RmcCharacterConfig
        Configuration for the character branch.
    semantic_trainable : bool
        Whether the semantic encoder parameters require gradients.
        Does NOT affect eval/train mode.  Default True.
    """

    def __init__(
        self,
        encoder_cfg: RmcEncoderConfig,
        character_cfg: RmcCharacterConfig,
        semantic_trainable: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = RmcEncoder(encoder_cfg)
        self.character_branch = RmcCharacterBranch(character_cfg)
        self.fusion = RmcFusion()
        self.heads = RmcMultiTaskHeads()

        self._set_semantic_trainable(semantic_trainable)

    # ------------------------------------------------------------------
    # Semantic trainability control
    # ------------------------------------------------------------------

    def _set_semantic_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze the semantic encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = trainable

    def set_semantic_trainable(self, trainable: bool) -> None:
        """Public API to freeze/unfreeze the semantic encoder.

        Parameters
        ----------
        trainable : bool
            If True, semantic encoder parameters require gradients.
            If False, they are frozen.
        """
        self._set_semantic_trainable(trainable)

    def is_semantic_trainable(self) -> bool:
        """Return whether the semantic encoder is currently trainable."""
        return any(
            p.requires_grad for p in self.encoder.parameters()
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        byte_ids: torch.Tensor,
        byte_attention_mask: torch.Tensor,
    ) -> RmcMultiTaskOutput:
        """Full forward pass: encoder → character → fusion → heads.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs.  Shape [B, S], dtype torch.long.
        attention_mask : torch.Tensor
            Attention mask.  Shape [B, S], dtype torch.bool.
        byte_ids : torch.Tensor
            Byte IDs.  Shape [B, L], dtype torch.long.
        byte_attention_mask : torch.Tensor
            Byte mask.  Shape [B, L], dtype torch.bool.

        Returns
        -------
        RmcMultiTaskOutput
            Logits.  Shape [B, 6].
        """
        enc_out = self.encoder(input_ids, attention_mask)
        char_repr = self.character_branch(byte_ids, byte_attention_mask)
        fused_out = self.fusion(enc_out.semantic_repr, char_repr)
        logits = self.heads(fused_out.fused_repr)
        return RmcMultiTaskOutput(logits=logits)

    def forward_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        byte_ids: torch.Tensor,
        byte_attention_mask: torch.Tensor,
    ) -> RmcMultiTaskFeatures:
        """Full forward pass returning all intermediate representations.

        For testing and ablation analysis only.  Ordinary deployment
        should use :meth:`forward` which returns logits only.

        Returns
        -------
        RmcMultiTaskFeatures
            All intermediate representations plus final logits.
        """
        enc_out = self.encoder(input_ids, attention_mask)
        char_repr = self.character_branch(byte_ids, byte_attention_mask)
        fused_out = self.fusion(enc_out.semantic_repr, char_repr)
        logits = self.heads(fused_out.fused_repr)
        return RmcMultiTaskFeatures(
            semantic_repr=enc_out.semantic_repr,
            character_repr=char_repr,
            fused_repr=fused_out.fused_repr,
            logits=logits,
        )

    # ------------------------------------------------------------------
    # Parameter inspection
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        """Return count of parameters with requires_grad=True."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
