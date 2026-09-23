"""Pytest test suite for the RMC character branch (RRM 3.8).

Tests cover:
- Config validation and frozen status
- Byte preprocessing (ASCII, Hinglish, Devanagari, emoji, Unicode, U+0000,
  lone surrogate, empty string, truncation, padding, batch rejection)
- Model forward shapes and parameter count (139,904)
- Input validation (rank, shape, dtype, device, value ranges, length bounds)
- Padding invariance (masked positions with nonzero IDs do not affect output)
- Window-mask behavior (short texts, empty rows, no NaN/inf)
- Empty-text exact-zero contract
- Initialization properties
- Gradient flow
- Eval determinism
- Unicode edge cases

Architecture rule: RRM UNDERSTANDS. TRUST DECIDES.
"""

from __future__ import annotations

import pytest
import torch

from rrm.character_branch import (
    RmcCharacterBranch,
    RmcCharacterConfig,
    ByteBatch,
    encode_byte_batch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_cfg() -> RmcCharacterConfig:
    return RmcCharacterConfig()


def _tiny_cfg() -> RmcCharacterConfig:
    return RmcCharacterConfig(
        byte_vocab_size=257,
        pad_id=0,
        embedding_dim=16,
        max_byte_length=32,
        conv_channels=16,
        kernel_sizes=(2, 3),
        character_dim=8,
    )


# ===================================================================
# Config tests
# ===================================================================

class TestRmcCharacterConfig:
    def test_defaults(self):
        cfg = RmcCharacterConfig()
        assert cfg.byte_vocab_size == 257
        assert cfg.pad_id == 0
        assert cfg.embedding_dim == 64
        assert cfg.max_byte_length == 512
        assert cfg.conv_channels == 128
        assert cfg.kernel_sizes == (3, 4, 5)
        assert cfg.character_dim == 64
        assert cfg.embedding_dropout == 0.1
        assert cfg.post_cnn_dropout == 0.1
        assert cfg.initializer_range == 0.02
        assert cfg.layer_norm_eps == 1e-12

    def test_frozen(self):
        cfg = RmcCharacterConfig()
        with pytest.raises(AttributeError):
            cfg.byte_vocab_size = 256  # type: ignore[misc]

    def test_byte_vocab_not_257_raises(self):
        with pytest.raises(ValueError, match="byte_vocab_size"):
            RmcCharacterConfig(byte_vocab_size=256)

    def test_pad_id_not_zero_raises(self):
        with pytest.raises(ValueError, match="pad_id"):
            RmcCharacterConfig(pad_id=1)

    def test_embedding_dim_zero_raises(self):
        with pytest.raises(ValueError, match="embedding_dim"):
            RmcCharacterConfig(embedding_dim=0)

    def test_embedding_dim_negative_raises(self):
        with pytest.raises(ValueError, match="embedding_dim"):
            RmcCharacterConfig(embedding_dim=-1)

    def test_max_byte_length_zero_raises(self):
        with pytest.raises(ValueError, match="max_byte_length"):
            RmcCharacterConfig(max_byte_length=0)

    def test_max_byte_length_negative_raises(self):
        with pytest.raises(ValueError, match="max_byte_length"):
            RmcCharacterConfig(max_byte_length=-1)

    def test_conv_channels_zero_raises(self):
        with pytest.raises(ValueError, match="conv_channels"):
            RmcCharacterConfig(conv_channels=0)

    def test_conv_channels_negative_raises(self):
        with pytest.raises(ValueError, match="conv_channels"):
            RmcCharacterConfig(conv_channels=-1)

    def test_character_dim_zero_raises(self):
        with pytest.raises(ValueError, match="character_dim"):
            RmcCharacterConfig(character_dim=0)

    def test_character_dim_negative_raises(self):
        with pytest.raises(ValueError, match="character_dim"):
            RmcCharacterConfig(character_dim=-1)

    def test_empty_kernel_sizes_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            RmcCharacterConfig(kernel_sizes=())

    def test_duplicate_kernel_sizes_raises(self):
        with pytest.raises(ValueError, match="unique"):
            RmcCharacterConfig(kernel_sizes=(3, 3, 5))

    def test_kernel_zero_raises(self):
        with pytest.raises(ValueError, match="kernel sizes must be > 0"):
            RmcCharacterConfig(kernel_sizes=(3, 0, 5))

    def test_kernel_negative_raises(self):
        with pytest.raises(ValueError, match="kernel sizes must be > 0"):
            RmcCharacterConfig(kernel_sizes=(3, -1, 5))

    def test_max_byte_length_less_than_kernel_raises(self):
        with pytest.raises(ValueError, match="max_byte_length.*max kernel"):
            RmcCharacterConfig(max_byte_length=2)

    def test_embedding_dropout_negative_raises(self):
        with pytest.raises(ValueError, match="embedding_dropout"):
            RmcCharacterConfig(embedding_dropout=-0.1)

    def test_embedding_dropout_one_raises(self):
        with pytest.raises(ValueError, match="embedding_dropout"):
            RmcCharacterConfig(embedding_dropout=1.0)

    def test_post_cnn_dropout_negative_raises(self):
        with pytest.raises(ValueError, match="post_cnn_dropout"):
            RmcCharacterConfig(post_cnn_dropout=-0.1)

    def test_post_cnn_dropout_one_raises(self):
        with pytest.raises(ValueError, match="post_cnn_dropout"):
            RmcCharacterConfig(post_cnn_dropout=1.0)

    def test_initializer_range_zero_raises(self):
        with pytest.raises(ValueError, match="initializer_range"):
            RmcCharacterConfig(initializer_range=0)

    def test_initializer_range_negative_raises(self):
        with pytest.raises(ValueError, match="initializer_range"):
            RmcCharacterConfig(initializer_range=-0.02)

    def test_layer_norm_eps_zero_raises(self):
        with pytest.raises(ValueError, match="layer_norm_eps"):
            RmcCharacterConfig(layer_norm_eps=0)

    def test_layer_norm_eps_negative_raises(self):
        with pytest.raises(ValueError, match="layer_norm_eps"):
            RmcCharacterConfig(layer_norm_eps=-1e-12)


# ===================================================================
# Preprocessor tests
# ===================================================================

class TestEncodeByteBatch:
    def test_ascii(self):
        cfg = _tiny_cfg()
        result = encode_byte_batch(["hello"], cfg)
        assert result.byte_ids.shape == (1, 32)
        assert result.byte_attention_mask.shape == (1, 32)
        assert result.byte_ids.dtype == torch.long
        assert result.byte_attention_mask.dtype == torch.bool
        # "hello" -> bytes [104, 101, 108, 108, 111] -> IDs [105, 102, 109, 109, 112]
        expected_ids = torch.tensor([105, 102, 109, 109, 112], dtype=torch.long)
        assert torch.equal(result.byte_ids[0, :5], expected_ids)
        assert result.byte_attention_mask[0, :5].all()
        assert not result.byte_attention_mask[0, 5:].any()

    def test_hinglish_roman_hindi(self):
        cfg = _tiny_cfg()
        result = encode_byte_batch(["accha college hai"], cfg)
        assert result.byte_ids.dtype == torch.long
        assert result.byte_attention_mask.dtype == torch.bool
        # All real bytes should have IDs in 1..256
        real_ids = result.byte_ids[result.byte_attention_mask]
        assert real_ids.min() >= 1
        assert real_ids.max() <= 256

    def test_devanagari(self):
        cfg = _tiny_cfg()
        # Devanagari: each char is 2-3 UTF-8 bytes
        result = encode_byte_batch(["अच्छा"], cfg)
        real_ids = result.byte_ids[result.byte_attention_mask]
        assert real_ids.min() >= 1
        assert real_ids.max() <= 256
        # Should have more bytes than visible characters
        assert real_ids.numel() > 3  # 3 Devanagari chars, each 2-3 bytes

    def test_emoji(self):
        cfg = _tiny_cfg()
        result = encode_byte_batch(["good college! 🎓"], cfg)
        real_ids = result.byte_ids[result.byte_attention_mask]
        assert real_ids.min() >= 1
        assert real_ids.max() <= 256
        # Emoji is 4 bytes in UTF-8
        assert real_ids.numel() > len("good college! ")

    def test_mixed_unicode(self):
        cfg = _tiny_cfg()
        result = encode_byte_batch(["Hello अच्छा! 🎓"], cfg)
        real_ids = result.byte_ids[result.byte_attention_mask]
        assert real_ids.min() >= 1
        assert real_ids.max() <= 256

    def test_null_character(self):
        cfg = _tiny_cfg()
        result = encode_byte_batch(["a\x00b"], cfg)
        # \x00 encodes as byte 0 -> model ID 1
        real_ids = result.byte_ids[result.byte_attention_mask]
        assert 1 in real_ids  # byte 0x00 mapped to ID 1
        assert 0 not in real_ids  # padding ID 0 should not appear for real bytes

    def test_lone_surrogate(self):
        cfg = _tiny_cfg()
        # U+D800 is a lone surrogate (unpaired)
        text = "hello" + "\ud800" + "world"
        result = encode_byte_batch([text], cfg)
        # Must not raise; replacement character should appear
        assert result.byte_ids.shape == (1, 32)

    def test_empty_string(self):
        cfg = _tiny_cfg()
        result = encode_byte_batch([""], cfg)
        assert result.byte_ids.shape == (1, 32)
        assert torch.all(result.byte_ids == 0)
        assert not result.byte_attention_mask.any()

    def test_mixed_batch(self):
        cfg = _tiny_cfg()
        result = encode_byte_batch(["", "hello", "अच्छा"], cfg)
        assert result.byte_ids.shape == (3, 32)
        assert result.byte_attention_mask.shape == (3, 32)
        # First row: all zeros
        assert torch.all(result.byte_ids[0] == 0)
        assert not result.byte_attention_mask[0].any()
        # Second row: has real bytes
        assert result.byte_attention_mask[1].any()
        # Third row: has real bytes
        assert result.byte_attention_mask[2].any()

    def test_fixed_shape(self):
        cfg = _default_cfg()
        for text in ["hi", "a" * 1000, "", "🎓"]:
            result = encode_byte_batch([text], cfg)
            assert result.byte_ids.shape == (1, 512)
            assert result.byte_attention_mask.shape == (1, 512)

    def test_right_truncation(self):
        cfg = _tiny_cfg()  # max_byte_length = 32
        text = "a" * 100  # 100 bytes
        result = encode_byte_batch([text], cfg)
        assert result.byte_attention_mask[0].sum() == 32
        assert not result.byte_attention_mask[0, 32:].any()

    def test_padding_positions_zero(self):
        cfg = _tiny_cfg()
        result = encode_byte_batch(["hi"], cfg)
        assert torch.all(result.byte_ids[0, 2:] == 0)
        assert not result.byte_attention_mask[0, 2:].any()

    def test_byte_b_plus_one(self):
        cfg = _tiny_cfg()
        # ASCII 'A' = 65 -> model ID 66
        result = encode_byte_batch(["A"], cfg)
        assert result.byte_ids[0, 0].item() == 66
        # ASCII '\x00' = 0 -> model ID 1
        result2 = encode_byte_batch(["\x00"], cfg)
        assert result2.byte_ids[0, 0].item() == 1

    def test_u0000_maps_to_id_one(self):
        cfg = _tiny_cfg()
        result = encode_byte_batch(["\x00"], cfg)
        assert result.byte_ids[0, 0].item() == 1  # NOT padding 0

    def test_empty_batch_rejected(self):
        cfg = _tiny_cfg()
        with pytest.raises(ValueError, match="non-empty"):
            encode_byte_batch([], cfg)

    def test_non_string_rejected(self):
        cfg = _tiny_cfg()
        with pytest.raises(TypeError, match="str"):
            encode_byte_batch([123], cfg)

    def test_no_oov(self):
        """Every encoded byte maps to a valid model ID in [1, 256].
        No byte value produces an out-of-vocabulary ID."""
        cfg = RmcCharacterConfig(max_byte_length=512)
        # latin-1 decoding creates characters for every byte value 0..255
        text = bytes(range(256)).decode("latin-1")
        result = encode_byte_batch([text], cfg)
        real_ids = result.byte_ids[result.byte_attention_mask]
        # All real IDs in [1, 256]
        assert real_ids.min() >= 1
        assert real_ids.max() <= 256
        # ID 0 (padding) never appears in attended positions
        assert 0 not in real_ids


# ===================================================================
# Forward shape and parameter tests
# ===================================================================

class TestCharacterBranchForward:
    def test_default_parameter_count(self):
        model = RmcCharacterBranch(_default_cfg())
        assert model.count_parameters() == 139_904

    def test_tiny_parameter_count(self):
        model = RmcCharacterBranch(_tiny_cfg())
        count = model.count_parameters()
        assert count > 0
        model2 = RmcCharacterBranch(_tiny_cfg())
        assert model2.count_parameters() == count

    def test_eval_forward_shape_default(self):
        model = RmcCharacterBranch(_default_cfg()).eval()
        cfg = _default_cfg()
        batch = encode_byte_batch(["hello world"], cfg)
        byte_ids = batch.byte_ids
        mask = batch.byte_attention_mask
        with torch.no_grad():
            out = model(byte_ids, mask)
        assert out.shape == (1, 64)

    def test_eval_forward_shape_tiny(self):
        model = RmcCharacterBranch(_tiny_cfg()).eval()
        cfg = _tiny_cfg()
        batch = encode_byte_batch(["hello"], cfg)
        byte_ids = batch.byte_ids
        mask = batch.byte_attention_mask
        with torch.no_grad():
            out = model(byte_ids, mask)
        assert out.shape == (1, 8)

    def test_eval_forward_shape_batch2(self):
        model = RmcCharacterBranch(_tiny_cfg()).eval()
        cfg = _tiny_cfg()
        batch = encode_byte_batch(["hello", "world"], cfg)
        byte_ids = batch.byte_ids
        mask = batch.byte_attention_mask
        with torch.no_grad():
            out = model(byte_ids, mask)
        assert out.shape == (2, 8)

    def test_all_finite(self):
        model = RmcCharacterBranch(_tiny_cfg()).eval()
        cfg = _tiny_cfg()
        batch = encode_byte_batch(["test"], cfg)
        byte_ids = batch.byte_ids
        mask = batch.byte_attention_mask
        with torch.no_grad():
            out = model(byte_ids, mask)
        assert torch.isfinite(out).all()


# ===================================================================
# Input validation tests
# ===================================================================

class TestCharacterBranchInputValidation:
    def test_rank1_byte_ids_raises(self):
        model = RmcCharacterBranch(_tiny_cfg())
        ids = torch.zeros(16, dtype=torch.long)
        mask = torch.zeros((1, 16), dtype=torch.bool)
        with pytest.raises(ValueError, match="rank-2"):
            model(ids, mask)

    def test_rank3_byte_ids_raises(self):
        model = RmcCharacterBranch(_tiny_cfg())
        ids = torch.zeros((1, 16, 1), dtype=torch.long)
        mask = torch.zeros((1, 16), dtype=torch.bool)
        with pytest.raises(ValueError, match="rank-2"):
            model(ids, mask)

    def test_rank3_mask_raises(self):
        model = RmcCharacterBranch(_tiny_cfg())
        ids = torch.zeros((1, 16), dtype=torch.long)
        mask = torch.zeros((1, 16, 1), dtype=torch.bool)
        with pytest.raises(ValueError, match="rank-2"):
            model(ids, mask)

    def test_shape_mismatch_raises(self):
        model = RmcCharacterBranch(_tiny_cfg())
        ids = torch.zeros((1, 16), dtype=torch.long)
        mask = torch.zeros((1, 8), dtype=torch.bool)
        with pytest.raises(ValueError, match="must match"):
            model(ids, mask)

    def test_empty_batch_raises(self):
        model = RmcCharacterBranch(_tiny_cfg())
        ids = torch.zeros((0, 16), dtype=torch.long)
        mask = torch.zeros((0, 16), dtype=torch.bool)
        with pytest.raises(ValueError, match="batch"):
            model(ids, mask)

    def test_seq_too_short_raises(self):
        model = RmcCharacterBranch(_tiny_cfg())  # max kernel = 2
        ids = torch.zeros((1, 1), dtype=torch.long)
        mask = torch.ones((1, 1), dtype=torch.bool)
        with pytest.raises(ValueError, match="sequence length.*max kernel"):
            model(ids, mask)

    def test_seq_too_long_raises(self):
        model = RmcCharacterBranch(_tiny_cfg())  # max_byte_length = 32
        ids = torch.zeros((1, 33), dtype=torch.long)
        mask = torch.ones((1, 33), dtype=torch.bool)
        with pytest.raises(ValueError, match="exceeds max_byte_length"):
            model(ids, mask)

    def test_float_ids_raises(self):
        model = RmcCharacterBranch(_tiny_cfg())
        ids = torch.zeros((1, 16), dtype=torch.float32)
        mask = torch.ones((1, 16), dtype=torch.bool)
        with pytest.raises(ValueError, match="torch.long"):
            model(ids, mask)

    def test_float_mask_raises(self):
        model = RmcCharacterBranch(_tiny_cfg())
        ids = torch.zeros((1, 16), dtype=torch.long)
        mask = torch.ones((1, 16), dtype=torch.float32)
        with pytest.raises(ValueError, match="torch.bool"):
            model(ids, mask)

    def test_id_too_low_raises(self):
        model = RmcCharacterBranch(_tiny_cfg())
        ids = torch.zeros((1, 16), dtype=torch.long)
        ids[0, 0] = -1
        mask = torch.ones((1, 16), dtype=torch.bool)
        with pytest.raises(ValueError, match="0, 256"):
            model(ids, mask)

    def test_id_too_high_raises(self):
        model = RmcCharacterBranch(_tiny_cfg())
        ids = torch.zeros((1, 16), dtype=torch.long)
        ids[0, 0] = 257
        mask = torch.ones((1, 16), dtype=torch.bool)
        with pytest.raises(ValueError, match="0, 256"):
            model(ids, mask)

    def test_attended_id_zero_raises(self):
        model = RmcCharacterBranch(_tiny_cfg())
        ids = torch.ones((1, 16), dtype=torch.long)
        ids[0, 0] = 0  # attended position with pad ID
        mask = torch.ones((1, 16), dtype=torch.bool)
        with pytest.raises(ValueError, match="1, 256"):
            model(ids, mask)

    def test_masked_position_nonzero_allowed(self):
        """Masked positions may contain legal nonzero byte IDs."""
        model = RmcCharacterBranch(_tiny_cfg()).eval()
        cfg = _tiny_cfg()
        seq_len = cfg.max_byte_length  # 32
        # attended: first 5 positions
        attended = torch.tensor([10, 20, 30, 40, 50], dtype=torch.long)
        mask = torch.zeros((1, seq_len), dtype=torch.bool)
        mask[0, :5] = True

        # Input A: zeros in masked positions
        ids_a = torch.zeros((1, seq_len), dtype=torch.long)
        ids_a[0, :5] = attended

        # Input B: nonzero values in masked positions (positions 5..31)
        ids_b = torch.zeros((1, seq_len), dtype=torch.long)
        ids_b[0, :5] = attended
        for j in range(5, seq_len):
            ids_b[0, j] = (j % 256) + 1

        with torch.no_grad():
            out_a = model(ids_a, mask)
            out_b = model(ids_b, mask)
        assert torch.allclose(out_a, out_b, atol=1e-6)

    def test_empty_rows_allowed(self):
        model = RmcCharacterBranch(_tiny_cfg()).eval()
        cfg = _tiny_cfg()
        batch = encode_byte_batch(["", "hi"], cfg)
        byte_ids = batch.byte_ids
        mask = batch.byte_attention_mask
        with torch.no_grad():
            out = model(byte_ids, mask)
        assert out.shape == (2, 8)
        assert torch.equal(out[0], torch.zeros(8))
        assert torch.isfinite(out[1]).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_device_mismatch_raises(self):
        model = RmcCharacterBranch(_tiny_cfg()).cuda()
        ids = torch.zeros((1, 16), dtype=torch.long, device="cuda")
        mask = torch.ones((1, 16), dtype=torch.bool)  # CPU
        with pytest.raises(ValueError, match="same device"):
            model(ids, mask)


# ===================================================================
# Padding invariance tests
# ===================================================================

class TestPaddingInvariance:
    def test_masked_nonzero_ids_dont_affect_output(self):
        """Same attended IDs, different nonzero IDs in masked positions,
        same mask → identical character_repr."""
        model = RmcCharacterBranch(_tiny_cfg()).eval()
        cfg = _tiny_cfg()
        batch_size = 2
        seq_len = cfg.max_byte_length

        # Build identical attended content
        torch.manual_seed(42)
        attended = torch.randint(1, 256, (batch_size, 8), dtype=torch.long)
        mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        mask[:, :8] = True

        # Input A: zeros in masked positions
        ids_a = torch.zeros((batch_size, seq_len), dtype=torch.long)
        ids_a[:, :8] = attended

        # Input B: nonzero values in masked positions
        ids_b = torch.zeros((batch_size, seq_len), dtype=torch.long)
        ids_b[:, :8] = attended
        ids_b[:, 8:] = torch.randint(1, 256, (batch_size, seq_len - 8),
                                     dtype=torch.long)

        with torch.no_grad():
            out_a = model(ids_a, mask)
            out_b = model(ids_b, mask)
        assert torch.allclose(out_a, out_b, atol=1e-6)

    def test_different_padding_values_same_output(self):
        """Even extreme nonzero padding values should not affect output."""
        model = RmcCharacterBranch(_tiny_cfg()).eval()
        cfg = _tiny_cfg()
        seq_len = cfg.max_byte_length

        ids = torch.zeros((1, seq_len), dtype=torch.long)
        ids[0, :4] = torch.tensor([10, 20, 30, 40])
        mask = torch.zeros((1, seq_len), dtype=torch.bool)
        mask[0, :4] = True

        # Fill masked positions with max valid ID (256)
        ids_padded = ids.clone()
        ids_padded[0, 4:] = 256

        with torch.no_grad():
            out_orig = model(ids, mask)
            out_padded = model(ids_padded, mask)
        assert torch.allclose(out_orig, out_padded, atol=1e-6)


# ===================================================================
# Window-mask tests
# ===================================================================

class TestWindowMask:
    def test_short_real_content_but_above_min_length(self):
        """Sequence length >= max kernel, but very short real content.
        The no-valid-window fallback should produce zero output."""
        cfg = _tiny_cfg()  # kernels (2, 3), max_byte_length = 32
        model = RmcCharacterBranch(cfg).eval()
        seq_len = cfg.max_byte_length  # 32
        # Only 1 real byte, rest is padding
        byte_ids = torch.zeros((1, seq_len), dtype=torch.long)
        byte_ids[0, 0] = 65
        mask = torch.zeros((1, seq_len), dtype=torch.bool)
        mask[0, 0] = True
        with torch.no_grad():
            out = model(byte_ids, mask)
        assert out.shape == (1, 8)
        assert torch.isfinite(out).all()
        # With only 1 real byte and max kernel=3, no valid windows → zero
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_exact_kernel_length(self):
        """Text length equals largest kernel: exactly one valid window."""
        cfg = _tiny_cfg()  # kernels (2, 3), max_byte_length = 32
        model = RmcCharacterBranch(cfg).eval()
        # seq_len=3, largest kernel=3 → 1 valid window for k=3
        byte_ids = torch.zeros((1, 3), dtype=torch.long)
        byte_ids[0, 0] = 65
        byte_ids[0, 1] = 66
        byte_ids[0, 2] = 67
        mask = torch.ones((1, 3), dtype=torch.bool)
        with torch.no_grad():
            out = model(byte_ids, mask)
        assert out.shape == (1, 8)
        assert torch.isfinite(out).all()

    def test_mixed_lengths_in_batch(self):
        """Batch with short and long texts."""
        cfg = _tiny_cfg()
        model = RmcCharacterBranch(cfg).eval()
        seq_len = cfg.max_byte_length
        # Row 0: 1 real byte (short)
        # Row 1: full length (long)
        byte_ids = torch.zeros((2, seq_len), dtype=torch.long)
        byte_ids[0, 0] = 65
        byte_ids[1, :16] = torch.randint(1, 256, (16,), dtype=torch.long)
        mask = torch.zeros((2, seq_len), dtype=torch.bool)
        mask[0, 0] = True
        mask[1, :16] = True
        with torch.no_grad():
            out = model(byte_ids, mask)
        assert out.shape == (2, 8)
        assert torch.isfinite(out).all()
        # Row 0 should be near-zero (no valid windows for most kernels)
        assert torch.allclose(out[0], torch.zeros_like(out[0]), atol=1e-6)

    def test_no_nan_no_inf(self):
        model = RmcCharacterBranch(_tiny_cfg()).eval()
        cfg = _tiny_cfg()
        # Mix of empty, short, and normal texts
        texts = ["", "a", "hello world", "अच्छा"]
        batch = encode_byte_batch(texts, cfg)
        byte_ids = batch.byte_ids
        mask = batch.byte_attention_mask
        with torch.no_grad():
            out = model(byte_ids, mask)
        assert torch.isfinite(out).all()


# ===================================================================
# Empty-text tests
# ===================================================================

class TestEmptyText:
    def test_empty_string_exact_zero(self):
        cfg = _default_cfg()
        model = RmcCharacterBranch(cfg).eval()
        batch = encode_byte_batch([""], cfg)
        byte_ids = batch.byte_ids
        mask = batch.byte_attention_mask
        assert torch.all(byte_ids == 0)
        assert not mask.any()
        with torch.no_grad():
            out = model(byte_ids, mask)
        assert out.shape == (1, 64)
        assert torch.isfinite(out).all()
        assert torch.equal(out, torch.zeros_like(out))

    def test_mixed_batch_empty_and_text(self):
        cfg = _tiny_cfg()
        model = RmcCharacterBranch(cfg).eval()
        batch = encode_byte_batch(["", "good"], cfg)
        byte_ids = batch.byte_ids
        mask = batch.byte_attention_mask
        with torch.no_grad():
            out = model(byte_ids, mask)
        assert out.shape == (2, 8)
        assert torch.equal(out[0], torch.zeros(8))
        assert torch.isfinite(out[1]).all()

    def test_train_mode_empty_text(self):
        cfg = _tiny_cfg()
        model = RmcCharacterBranch(cfg).train()
        batch = encode_byte_batch([""], cfg)
        byte_ids = batch.byte_ids
        mask = batch.byte_attention_mask
        out = model(byte_ids, mask)
        assert out.shape == (1, 8)
        assert torch.isfinite(out).all()
        assert torch.equal(out, torch.zeros_like(out))


# ===================================================================
# Initialization tests
# ===================================================================

class TestInitialization:
    def test_conv_biases_zero(self):
        model = RmcCharacterBranch(_tiny_cfg())
        for i, conv in enumerate(model.convs):
            assert torch.allclose(conv.bias, torch.zeros_like(conv.bias)), (
                f"Conv {i} bias not zero"
            )

    def test_linear_bias_zero(self):
        model = RmcCharacterBranch(_tiny_cfg())
        assert torch.allclose(
            model.projection.bias, torch.zeros_like(model.projection.bias)
        )

    def test_layernorm_weights_one(self):
        model = RmcCharacterBranch(_tiny_cfg())
        assert torch.allclose(
            model.final_ln.weight, torch.ones_like(model.final_ln.weight)
        )

    def test_layernorm_biases_zero(self):
        model = RmcCharacterBranch(_tiny_cfg())
        assert torch.allclose(
            model.final_ln.bias, torch.zeros_like(model.final_ln.bias)
        )

    def test_pad_embedding_zero(self):
        model = RmcCharacterBranch(_tiny_cfg())
        assert torch.equal(
            model.byte_embeddings.weight[0],
            torch.zeros_like(model.byte_embeddings.weight[0]),
        )

    def test_pad_id_is_padding_idx(self):
        model = RmcCharacterBranch(_tiny_cfg())
        assert model.byte_embeddings.padding_idx == 0

    def test_non_pad_embeddings_not_all_zero(self):
        model = RmcCharacterBranch(_tiny_cfg())
        non_pad = model.byte_embeddings.weight[1:]
        assert not torch.allclose(
            non_pad, torch.zeros_like(non_pad), atol=1e-12
        )

    def test_conv_weights_not_all_zero(self):
        model = RmcCharacterBranch(_tiny_cfg())
        for i, conv in enumerate(model.convs):
            assert not torch.allclose(
                conv.weight, torch.zeros_like(conv.weight)
            ), f"Conv {i} weight is all zero"


# ===================================================================
# Gradient test
# ===================================================================

class TestGradientFlow:
    def test_gradients_exist_and_finite(self):
        model = RmcCharacterBranch(_tiny_cfg())
        model.train()
        cfg = _tiny_cfg()
        batch = encode_byte_batch(["hello", "world"], cfg)
        byte_ids = batch.byte_ids
        mask = batch.byte_attention_mask
        out = model(byte_ids, mask)
        loss = out.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )
        assert has_grad, "No parameter has a gradient"
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), (
                    f"Non-finite gradient in {name}"
                )


# ===================================================================
# Eval determinism
# ===================================================================

class TestEvalDeterminism:
    def test_eval_deterministic(self):
        model = RmcCharacterBranch(_tiny_cfg()).eval()
        cfg = _tiny_cfg()
        batch = encode_byte_batch(["test", "hello"], cfg)
        byte_ids = batch.byte_ids
        mask = batch.byte_attention_mask
        with torch.no_grad():
            out1 = model(byte_ids, mask)
            out2 = model(byte_ids, mask)
        assert torch.equal(out1, out2)


# ===================================================================
# Unicode sensitivity tests
# ===================================================================

class TestUnicodeSensitivity:
    def test_different_inputs_produce_different_outputs(self):
        """Different raw texts should generally produce different
        character representations (not guaranteed mathematically, but
        expected for random weights)."""
        model = RmcCharacterBranch(_tiny_cfg()).eval()
        cfg = _tiny_cfg()
        texts = ["good", "goooood"]
        batch = encode_byte_batch(texts, cfg)
        byte_ids = batch.byte_ids
        mask = batch.byte_attention_mask
        with torch.no_grad():
            out = model(byte_ids, mask)
        # Different raw bytes → generally different CNN activations
        assert not torch.allclose(out[0], out[1], atol=1e-3)

    def test_devanagari_forward(self):
        model = RmcCharacterBranch(_tiny_cfg()).eval()
        cfg = _tiny_cfg()
        batch = encode_byte_batch(["अच्छा"], cfg)
        byte_ids = batch.byte_ids
        mask = batch.byte_attention_mask
        with torch.no_grad():
            out = model(byte_ids, mask)
        assert out.shape == (1, 8)
        assert torch.isfinite(out).all()

    def test_emoji_forward(self):
        model = RmcCharacterBranch(_tiny_cfg()).eval()
        cfg = _tiny_cfg()
        batch = encode_byte_batch(["🎓🎓"], cfg)
        byte_ids = batch.byte_ids
        mask = batch.byte_attention_mask
        with torch.no_grad():
            out = model(byte_ids, mask)
        assert out.shape == (1, 8)
        assert torch.isfinite(out).all()
