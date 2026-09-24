"""Tests for rrm/multitask.py — multi-task heads and top-level model."""

from __future__ import annotations

import dataclasses
import inspect

import pytest
import torch

from rrm.character_branch import RmcCharacterConfig, RmcCharacterBranch
from rrm.encoder import RmcEncoder, RmcEncoderConfig
from rrm.fusion import RmcFusion
from rrm.labels import NUM_PRIMARY_LABELS
from rrm.multitask import (
    RmcMultiTaskFeatures,
    RmcMultiTaskHeads,
    RmcMultiTaskModel,
    RmcMultiTaskOutput,
)


# ---------------------------------------------------------------------------
# RmcMultiTaskHeads tests
# ---------------------------------------------------------------------------


class TestRmcMultiTaskHeads:
    """Tests for the classification heads module."""

    def test_default_parameters(self):
        heads = RmcMultiTaskHeads()
        assert heads.input_dim == 448
        assert heads.num_labels == NUM_PRIMARY_LABELS

    def test_locked_dimensions(self):
        """Architecture is permanently locked to Linear(448, 6)."""
        heads = RmcMultiTaskHeads()
        assert heads.input_dim == 448
        assert heads.num_labels == 6
        # Verify the actual linear layer dimensions
        assert heads.linear.in_features == 448
        assert heads.linear.out_features == 6

    def test_no_configurable_dimensions(self):
        """Constructor takes no arguments — dimensions are locked."""
        sig = inspect.signature(RmcMultiTaskHeads.__init__)
        # Only 'self' parameter — no configurable dimensions
        assert len(sig.parameters) == 1

    def test_exact_parameter_count(self):
        """Must be exactly 2,694 trainable parameters."""
        heads = RmcMultiTaskHeads()
        assert heads.count_parameters() == 2_694

    def test_parameter_arithmetic(self):
        """448 * 6 + 6 = 2,694."""
        expected = 448 * NUM_PRIMARY_LABELS + NUM_PRIMARY_LABELS
        assert expected == 2_694

    def test_forward_shape(self):
        heads = RmcMultiTaskHeads()
        fused = torch.randn(4, 448)
        logits = heads(fused)
        assert logits.shape == (4, 6)

    def test_forward_dtype_float32(self):
        heads = RmcMultiTaskHeads()
        fused = torch.randn(2, 448, dtype=torch.float32)
        logits = heads(fused)
        assert logits.dtype == torch.float32

    def test_no_sigmoid_softmax(self):
        """Outputs must be raw logits, not probabilities."""
        heads = RmcMultiTaskHeads()
        fused = torch.randn(2, 448)
        logits = heads(fused)
        # Logits can be outside [0, 1]
        assert (logits > 1).any() or (logits < 0).any()

    def test_batch_size_one(self):
        heads = RmcMultiTaskHeads()
        fused = torch.randn(1, 448)
        logits = heads(fused)
        assert logits.shape == (1, 6)

    def test_large_batch(self):
        heads = RmcMultiTaskHeads()
        fused = torch.randn(256, 448)
        logits = heads(fused)
        assert logits.shape == (256, 6)

    def test_wrong_rank(self):
        heads = RmcMultiTaskHeads()
        with pytest.raises(ValueError, match="rank-2"):
            heads(torch.randn(448))

    def test_wrong_last_dim(self):
        heads = RmcMultiTaskHeads()
        with pytest.raises(ValueError, match="last dim must be 448"):
            heads(torch.randn(2, 64))

    def test_zero_batch(self):
        heads = RmcMultiTaskHeads()
        with pytest.raises(ValueError, match="batch dimension must be > 0"):
            heads(torch.empty(0, 448))

    def test_non_floating_rejected(self):
        heads = RmcMultiTaskHeads()
        with pytest.raises(ValueError, match="floating dtype"):
            heads(torch.zeros(2, 448, dtype=torch.long))

    def test_independent_rows(self):
        """Changing one input row must not affect another output row."""
        heads = RmcMultiTaskHeads()
        fused_a = torch.randn(1, 448)
        fused_b = torch.randn(1, 448)
        fused_stack = torch.cat([fused_a, fused_b], dim=0)
        logits = heads(fused_stack)
        logits_a = heads(fused_a)
        logits_b = heads(fused_b)
        assert torch.allclose(logits[0], logits_a[0])
        assert torch.allclose(logits[1], logits_b[0])


# ---------------------------------------------------------------------------
# RmcMultiTaskOutput tests
# ---------------------------------------------------------------------------


class TestRmcMultiTaskOutput:
    def test_frozen(self):
        out = RmcMultiTaskOutput(logits=torch.zeros(1, 6))
        with pytest.raises(dataclasses.FrozenInstanceError):
            out.logits = torch.ones(1, 6)  # type: ignore[misc]

    def test_logits_shape(self):
        out = RmcMultiTaskOutput(logits=torch.zeros(3, 6))
        assert out.logits.shape == (3, 6)


# ---------------------------------------------------------------------------
# RmcMultiTaskFeatures tests
# ---------------------------------------------------------------------------


class TestRmcMultiTaskFeatures:
    def test_frozen(self):
        feat = RmcMultiTaskFeatures(
            semantic_repr=torch.zeros(1, 384),
            character_repr=torch.zeros(1, 64),
            fused_repr=torch.zeros(1, 448),
            logits=torch.zeros(1, 6),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            feat.logits = torch.ones(1, 6)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RmcMultiTaskModel tests
# ---------------------------------------------------------------------------


class TestRmcMultiTaskModel:
    """Tests for the full composed model."""

    @pytest.fixture()
    def default_model(self):
        return RmcMultiTaskModel(
            encoder_cfg=RmcEncoderConfig(),
            character_cfg=RmcCharacterConfig(),
        )

    def test_total_parameters(self, default_model):
        """Total must be exactly 22,886,150 (measured by iteration)."""
        assert default_model.count_parameters() == 22_886_150

    def test_parameter_breakdown(self, default_model):
        """Verify total = encoder + char + fusion + heads (measured)."""
        enc = RmcEncoder(RmcEncoderConfig())
        char = RmcCharacterBranch(RmcCharacterConfig())
        heads = RmcMultiTaskHeads()
        total = (
            enc.count_parameters()
            + char.count_parameters()
            + sum(p.numel() for p in RmcFusion().parameters())
            + heads.count_parameters()
        )
        assert total == 22_886_150

    def test_trainable_parameters_default(self, default_model):
        """All components trainable by default."""
        trainable = default_model.count_trainable_parameters()
        total = default_model.count_parameters()
        assert trainable == total

    def test_semantic_trainable_default(self, default_model):
        assert default_model.is_semantic_trainable() is True

    def test_set_semantic_frozen(self, default_model):
        default_model.set_semantic_trainable(False)
        assert default_model.is_semantic_trainable() is False
        # Character branch still trainable
        assert any(
            p.requires_grad for p in default_model.character_branch.parameters()
        )

    def test_set_semantic_trainable(self, default_model):
        default_model.set_semantic_trainable(False)
        default_model.set_semantic_trainable(True)
        assert default_model.is_semantic_trainable() is True

    def test_forward_output_type(self, default_model):
        default_model.eval()
        batch = _make_batch(2, seq_len=16, byte_len=32)
        with torch.no_grad():
            out = default_model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["byte_ids"],
                batch["byte_attention_mask"],
            )
        assert isinstance(out, RmcMultiTaskOutput)

    def test_forward_logits_shape(self, default_model):
        default_model.eval()
        batch = _make_batch(4, seq_len=16, byte_len=32)
        with torch.no_grad():
            out = default_model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["byte_ids"],
                batch["byte_attention_mask"],
            )
        assert out.logits.shape == (4, 6)

    def test_forward_features_shape(self, default_model):
        default_model.eval()
        batch = _make_batch(3, seq_len=16, byte_len=32)
        with torch.no_grad():
            feat = default_model.forward_features(
                batch["input_ids"],
                batch["attention_mask"],
                batch["byte_ids"],
                batch["byte_attention_mask"],
            )
        assert isinstance(feat, RmcMultiTaskFeatures)
        assert feat.semantic_repr.shape == (3, 384)
        assert feat.character_repr.shape == (3, 64)
        assert feat.fused_repr.shape == (3, 448)
        assert feat.logits.shape == (3, 6)

    def test_forward_dtype_cpu(self, default_model):
        """CPU forward: output dtype follows module weight dtype (float32)."""
        default_model.eval()
        batch = _make_batch(2, seq_len=16, byte_len=32)
        with torch.no_grad():
            out = default_model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["byte_ids"],
                batch["byte_attention_mask"],
            )
        assert out.logits.dtype == torch.float32

    def test_forward_no_sigmoid_softmax(self, default_model):
        """Outputs must be raw logits."""
        default_model.eval()
        batch = _make_batch(2, seq_len=16, byte_len=32)
        with torch.no_grad():
            out = default_model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["byte_ids"],
                batch["byte_attention_mask"],
            )
        # Raw logits can exceed [0, 1] range
        assert (out.logits > 1.0).any() or (out.logits < 0.0).any()

    def test_eval_mode_no_dropout(self, default_model):
        """In eval mode, dropout should be disabled."""
        default_model.eval()
        batch = _make_batch(2, seq_len=16, byte_len=32)
        with torch.no_grad():
            out1 = default_model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["byte_ids"],
                batch["byte_attention_mask"],
            )
            out2 = default_model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["byte_ids"],
                batch["byte_attention_mask"],
            )
        assert torch.allclose(out1.logits, out2.logits)

    def test_gradient_flow_to_heads(self, default_model):
        """Gradients must flow to the heads."""
        default_model.train()
        batch = _make_batch(2, seq_len=16, byte_len=32)
        out = default_model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["byte_ids"],
            batch["byte_attention_mask"],
        )
        loss = out.logits.sum()
        loss.backward()
        assert default_model.heads.linear.weight.grad is not None

    def test_gradient_flow_to_character_when_trainable(self, default_model):
        """Gradients flow to character branch when trainable."""
        default_model.train()
        default_model.set_semantic_trainable(False)
        batch = _make_batch(2, seq_len=16, byte_len=32)
        out = default_model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["byte_ids"],
            batch["byte_attention_mask"],
        )
        loss = out.logits.sum()
        loss.backward()
        # Character branch gradients should exist
        assert any(
            p.grad is not None
            for p in default_model.character_branch.parameters()
        )

    def test_no_gradient_to_frozen_semantic(self, default_model):
        """Frozen semantic encoder should have no gradients."""
        default_model.train()
        default_model.set_semantic_trainable(False)
        batch = _make_batch(2, seq_len=16, byte_len=32)
        out = default_model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["byte_ids"],
            batch["byte_attention_mask"],
        )
        loss = out.logits.sum()
        loss.backward()
        for name, param in default_model.encoder.named_parameters():
            assert param.grad is None, f"Encoder param {name} has grad"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(
    batch_size: int,
    seq_len: int = 16,
    byte_len: int = 32,
    vocab_size: int = 22000,
) -> dict:
    """Create a minimal valid input batch for the model."""
    torch.manual_seed(0)
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    byte_ids = torch.randint(1, 256, (batch_size, byte_len), dtype=torch.long)
    byte_attention_mask = torch.ones(batch_size, byte_len, dtype=torch.bool)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "byte_ids": byte_ids,
        "byte_attention_mask": byte_attention_mask,
    }
