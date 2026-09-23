"""Feature fusion for the RRM (RRM 3.8).

Implements the simple concatenation fusion that combines the semantic
representation from the RMC Transformer encoder with the character-level
representation from the character branch.

Research provenance:
    - Concatenation fusion: RRM EXPERIMENTAL DESIGN CHOICE

Architecture rule:
    RRM UNDERSTANDS. TRUST DECIDES.

This module produces a fused representation only.
It does NOT make moderation decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RmcFusionOutput:
    """Immutable output from the fusion module.

    Attributes
    ----------
    fused_repr : torch.Tensor
        Concatenated semantic + character representation.
        Shape [B, 448].
    """

    fused_repr: torch.Tensor


class RmcFusion(nn.Module):
    """Parameter-free concatenation fusion.

    Combines the 384-dimensional semantic representation from the
    RmcEncoder with the 64-dimensional character representation from
    RmcCharacterBranch into a single 448-dimensional vector.

    This module has ZERO trainable parameters.

    Parameters
    ----------
    semantic_dim : int
        Dimension of the semantic representation.  Default 384.
    character_dim : int
        Dimension of the character representation.  Default 64.
    """

    def __init__(
        self,
        semantic_dim: int = 384,
        character_dim: int = 64,
    ) -> None:
        super().__init__()
        self.semantic_dim = semantic_dim
        self.character_dim = character_dim
        self._fused_dim = semantic_dim + character_dim

        # Validate dimensions at construction
        if semantic_dim <= 0:
            raise ValueError(
                f"semantic_dim must be > 0, got {semantic_dim}"
            )
        if character_dim <= 0:
            raise ValueError(
                f"character_dim must be > 0, got {character_dim}"
            )

    def forward(
        self,
        semantic_repr: torch.Tensor,
        character_repr: torch.Tensor,
    ) -> RmcFusionOutput:
        """Fuse semantic and character representations.

        Parameters
        ----------
        semantic_repr : torch.Tensor
            Semantic representation from RmcEncoder.  Shape [B, 384].
        character_repr : torch.Tensor
            Character representation from RmcCharacterBranch.  Shape [B, 64].

        Returns
        -------
        RmcFusionOutput
            Fused representation.  Shape [B, 448].

        Raises
        ------
        ValueError
            If inputs violate the shape, rank, device, or dtype contract.
        """
        # Rank validation
        if semantic_repr.dim() != 2:
            raise ValueError(
                f"semantic_repr must be rank-2 [B, D], got "
                f"rank-{semantic_repr.dim()}"
            )
        if character_repr.dim() != 2:
            raise ValueError(
                f"character_repr must be rank-2 [B, D], got "
                f"rank-{character_repr.dim()}"
            )

        # Shape validation
        if semantic_repr.shape[0] != character_repr.shape[0]:
            raise ValueError(
                f"batch sizes must match: semantic {semantic_repr.shape[0]} "
                f"vs character {character_repr.shape[0]}"
            )
        if semantic_repr.shape[1] != self.semantic_dim:
            raise ValueError(
                f"semantic_repr last dim must be {self.semantic_dim}, "
                f"got {semantic_repr.shape[1]}"
            )
        if character_repr.shape[1] != self.character_dim:
            raise ValueError(
                f"character_repr last dim must be {self.character_dim}, "
                f"got {character_repr.shape[1]}"
            )

        # Device validation
        if semantic_repr.device != character_repr.device:
            raise ValueError(
                f"semantic_repr and character_repr must be on the same "
                f"device, got {semantic_repr.device} vs {character_repr.device}"
            )

        # Dtype validation: must be floating and exactly equal
        if not semantic_repr.is_floating_point():
            raise ValueError(
                f"semantic_repr must be a floating-point tensor, "
                f"got {semantic_repr.dtype}"
            )
        if not character_repr.is_floating_point():
            raise ValueError(
                f"character_repr must be a floating-point tensor, "
                f"got {character_repr.dtype}"
            )
        if semantic_repr.dtype != character_repr.dtype:
            raise ValueError(
                f"semantic_repr and character_repr must have the same "
                f"dtype, got {semantic_repr.dtype} vs {character_repr.dtype}"
            )

        fused = torch.cat([semantic_repr, character_repr], dim=-1)
        return RmcFusionOutput(fused_repr=fused)
