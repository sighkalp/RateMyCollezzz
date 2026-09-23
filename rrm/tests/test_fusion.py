"""Pytest test suite for the RMC feature fusion module (RRM 3.8).

Tests cover:
- Fusion output shape
- Parameter count (0)
- Input validation (rank, shape, dtype, device)
- Dtype mismatch
- Device mismatch
- Frozen output dataclass
- Different inputs produce different fused outputs
- Gradient flow (zero parameters → no gradients)
- Batch dimension consistency

Architecture rule: RRM UNDERSTANDS. TRUST DECIDES.
"""

from __future__ import annotations

import pytest
import torch

from rrm.fusion import RmcFusion, RmcFusionOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_semantic(batch: int, dim: int = 384) -> torch.Tensor:
    return torch.randn(batch, dim)


def _make_character(batch: int, dim: int = 64) -> torch.Tensor:
    return torch.randn(batch, dim)


# ===================================================================
# Construction tests
# ===================================================================

class TestRmcFusionConstruction:
    def test_default_dims(self):
        fusion = RmcFusion()
        assert fusion.semantic_dim == 384
        assert fusion.character_dim == 64

    def test_fused_dim(self):
        fusion = RmcFusion()
        # 384 + 64 = 448
        assert fusion._fused_dim == 448

    def test_zero_parameters(self):
        fusion = RmcFusion()
        assert sum(p.numel() for p in fusion.parameters()) == 0

    def test_custom_dims(self):
        fusion = RmcFusion(semantic_dim=256, character_dim=32)
        assert fusion.semantic_dim == 256
        assert fusion.character_dim == 32
        assert fusion._fused_dim == 288

    def test_semantic_dim_zero_raises(self):
        with pytest.raises(ValueError, match="semantic_dim"):
            RmcFusion(semantic_dim=0)

    def test_semantic_dim_negative_raises(self):
        with pytest.raises(ValueError, match="semantic_dim"):
            RmcFusion(semantic_dim=-1)

    def test_character_dim_zero_raises(self):
        with pytest.raises(ValueError, match="character_dim"):
            RmcFusion(character_dim=0)

    def test_character_dim_negative_raises(self):
        with pytest.raises(ValueError, match="character_dim"):
            RmcFusion(character_dim=-1)


# ===================================================================
# Forward shape tests
# ===================================================================

class TestRmcFusionForward:
    def test_output_shape_default(self):
        fusion = RmcFusion()
        sem = _make_semantic(2)
        char = _make_character(2)
        output = fusion(sem, char)
        assert output.fused_repr.shape == (2, 448)

    def test_output_shape_single(self):
        fusion = RmcFusion()
        sem = _make_semantic(1)
        char = _make_character(1)
        output = fusion(sem, char)
        assert output.fused_repr.shape == (1, 448)

    def test_output_shape_large_batch(self):
        fusion = RmcFusion()
        batch = 64
        sem = _make_semantic(batch)
        char = _make_character(batch)
        output = fusion(sem, char)
        assert output.fused_repr.shape == (batch, 448)

    def test_output_is_concatenation(self):
        """Fused output equals explicit concatenation of inputs."""
        fusion = RmcFusion()
        sem = _make_semantic(2)
        char = _make_character(2)
        output = fusion(sem, char)
        expected = torch.cat([sem, char], dim=-1)
        assert torch.equal(output.fused_repr, expected)

    def test_output_dtype_matches_input(self):
        fusion = RmcFusion()
        sem = _make_semantic(2).float()
        char = _make_character(2).float()
        output = fusion(sem, char)
        assert output.fused_repr.dtype == sem.dtype
        assert output.fused_repr.dtype == char.dtype


# ===================================================================
# Input validation tests
# ===================================================================

class TestRmcFusionInputValidation:
    def test_semantic_rank1_raises(self):
        fusion = RmcFusion()
        sem = torch.zeros(384)
        char = _make_character(1)
        with pytest.raises(ValueError, match="rank-2"):
            fusion(sem, char)

    def test_semantic_rank3_raises(self):
        fusion = RmcFusion()
        sem = torch.zeros((1, 384, 1))
        char = _make_character(1)
        with pytest.raises(ValueError, match="rank-2"):
            fusion(sem, char)

    def test_character_rank1_raises(self):
        fusion = RmcFusion()
        sem = _make_semantic(1)
        char = torch.zeros(64)
        with pytest.raises(ValueError, match="rank-2"):
            fusion(sem, char)

    def test_character_rank3_raises(self):
        fusion = RmcFusion()
        sem = _make_semantic(1)
        char = torch.zeros((1, 64, 1))
        with pytest.raises(ValueError, match="rank-2"):
            fusion(sem, char)

    def test_batch_mismatch_raises(self):
        fusion = RmcFusion()
        sem = _make_semantic(2)
        char = _make_character(3)
        with pytest.raises(ValueError, match="batch sizes must match"):
            fusion(sem, char)

    def test_semantic_dim_mismatch_raises(self):
        fusion = RmcFusion()
        sem = torch.zeros((1, 128))
        char = _make_character(1)
        with pytest.raises(ValueError, match="semantic_repr last dim must be 384"):
            fusion(sem, char)

    def test_character_dim_mismatch_raises(self):
        fusion = RmcFusion()
        sem = _make_semantic(1)
        char = torch.zeros((1, 32))
        with pytest.raises(ValueError, match="character_repr last dim must be 64"):
            fusion(sem, char)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_device_mismatch_raises(self):
        fusion = RmcFusion().cuda()
        sem = _make_semantic(1).cuda()
        char = _make_character(1)  # CPU
        with pytest.raises(ValueError, match="same device"):
            fusion(sem, char)

    def test_integer_dtype_rejected(self):
        fusion = RmcFusion()
        sem = torch.zeros((1, 384), dtype=torch.long)
        char = torch.zeros((1, 64), dtype=torch.long)
        with pytest.raises(ValueError, match="floating-point"):
            fusion(sem, char)

    def test_dtype_mismatch_raises(self):
        fusion = RmcFusion()
        sem = _make_semantic(1).to(dtype=torch.float32)
        char = _make_character(1).to(dtype=torch.float64)
        with pytest.raises(ValueError, match="same dtype"):
            fusion(sem, char)


# ===================================================================
# Frozen output tests
# ===================================================================

class TestRmcFusionOutput:
    def test_frozen(self):
        out = RmcFusionOutput(fused_repr=torch.zeros(1, 448))
        with pytest.raises(AttributeError):
            out.fused_repr = torch.zeros(1, 448)  # type: ignore[misc]

    def test_output_fields(self):
        t = torch.randn(2, 448)
        out = RmcFusionOutput(fused_repr=t)
        assert out.fused_repr.shape == (2, 448)


# ===================================================================
# Behavioral tests
# ===================================================================

class TestRmcFusionBehavior:
    def test_different_inputs_different_output(self):
        fusion = RmcFusion()
        sem1 = _make_semantic(1)
        char1 = _make_character(1)
        sem2 = _make_semantic(1)
        char2 = _make_character(1)
        out1 = fusion(sem1, char1)
        out2 = fusion(sem2, char2)
        assert not torch.allclose(out1.fused_repr, out2.fused_repr, atol=1e-3)

    def test_same_inputs_same_output(self):
        fusion = RmcFusion()
        sem = _make_semantic(1)
        char = _make_character(1)
        out1 = fusion(sem, char)
        out2 = fusion(sem, char)
        assert torch.equal(out1.fused_repr, out2.fused_repr)

    def test_gradients_flow_through(self):
        """Fusion output propagates gradients when inputs require grad."""
        fusion = RmcFusion()
        sem = _make_semantic(2).requires_grad_()
        char = _make_character(2).requires_grad_()
        output = fusion(sem, char)
        # torch.cat of requires_grad inputs produces requires_grad output
        assert output.fused_repr.requires_grad
        loss = output.fused_repr.sum()
        loss.backward()
        assert sem.grad is not None
        assert char.grad is not None
