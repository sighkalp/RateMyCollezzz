"""Custom RMC Transformer semantic encoder (RRM 3.7).

Implements the permanent custom semantic encoder for the RateMyCollezzz
Review Risk Model.  This is the encoder architecture only — no training,
no teacher, no MLM head, no task heads, no CharCNN.

Research provenance:
    - Transformer encoder: PAPER-DERIVED (Vaswani et al., 2017)
    - Pre-LN layout: RRM EXPERIMENTAL DESIGN CHOICE (training stability)
    - GELU activation: PAPER-DERIVED (Devlin et al., 2018; Hendrycks & Gimpel, 2016)
    - Learned absolute position embeddings: PAPER-DERIVED (Devlin et al., 2018)
    - Dropout 0.1, initializer_range 0.02: RRM EXPERIMENTAL DESIGN CHOICE
    - No token-type embeddings: RRM EXPERIMENTAL DESIGN CHOICE
    - No permanent pooler: RRM EXPERIMENTAL DESIGN CHOICE

Architecture rule:
    RRM UNDERSTANDS. TRUST DECIDES.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RmcEncoderConfig:
    """Immutable configuration for the RMC custom semantic encoder.

    All fields are frozen after construction.  Defaults target the
    locked 22,743,552-parameter encoder design.
    """

    vocab_size: int = 22_000
    hidden_size: int = 384
    num_layers: int = 8
    num_attention_heads: int = 6
    ffn_size: int = 1536
    max_seq_length: int = 256
    pad_id: int = 0
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {self.hidden_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {self.num_layers}")
        if self.num_attention_heads <= 0:
            raise ValueError(
                f"num_attention_heads must be > 0, got {self.num_attention_heads}"
            )
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.ffn_size <= self.hidden_size:
            raise ValueError(
                f"ffn_size ({self.ffn_size}) must be > hidden_size "
                f"({self.hidden_size})"
            )
        if self.max_seq_length <= 0:
            raise ValueError(
                f"max_seq_length must be > 0, got {self.max_seq_length}"
            )
        if not (0 <= self.pad_id < self.vocab_size):
            raise ValueError(
                f"pad_id ({self.pad_id}) must be in [0, {self.vocab_size})"
            )
        if not (0 <= self.dropout < 1):
            raise ValueError(
                f"dropout must be in [0, 1), got {self.dropout}"
            )
        if not (0 <= self.attention_dropout < 1):
            raise ValueError(
                f"attention_dropout must be in [0, 1), got {self.attention_dropout}"
            )
        if self.layer_norm_eps <= 0:
            raise ValueError(
                f"layer_norm_eps must be > 0, got {self.layer_norm_eps}"
            )
        if self.initializer_range <= 0:
            raise ValueError(
                f"initializer_range must be > 0, got {self.initializer_range}"
            )


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RmcEncoderOutput:
    """Immutable output from the RMC semantic encoder.

    Attributes
    ----------
    last_hidden_state : torch.Tensor
        Full sequence of hidden states.  Shape [batch, seq, hidden_size].
    semantic_repr : torch.Tensor
        Pooled semantic representation from the final LayerNorm output
        at position 0 (the <cls> token).  Shape [batch, hidden_size].
    """

    last_hidden_state: torch.Tensor
    semantic_repr: torch.Tensor


# ---------------------------------------------------------------------------
# Internal Transformer block
# ---------------------------------------------------------------------------

class _RmcTransformerBlock(nn.Module):
    """Single Pre-LN Transformer encoder block.

    Topology:
        x = x + Attention(LN1(x))
        x = x + FFN(LN2(x))
    """

    def __init__(self, cfg: RmcEncoderConfig) -> None:
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.ffn_size = cfg.ffn_size

        self.ln1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.ln2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            dropout=cfg.attention_dropout,
            batch_first=True,
        )

        self.ffn1 = nn.Linear(cfg.hidden_size, cfg.ffn_size)
        self.ffn2 = nn.Linear(cfg.ffn_size, cfg.hidden_size)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Attention sublayer
        normed = self.ln1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)

        # FFN sublayer
        normed = self.ln2(x)
        ffn_out = self.ffn2(self.dropout(F.gelu(self.ffn1(normed))))
        x = x + self.dropout(ffn_out)

        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class RmcEncoder(nn.Module):
    """Permanent custom RMC semantic encoder.

    Architecture (locked, RRM 3.6/3.7):

        input_ids + position_ids
            -> Embedding LayerNorm
            -> Embedding Dropout
            -> 8 x Pre-LN Transformer blocks
            -> Final LayerNorm
            -> last_hidden_state [B, S, 384]
            -> semantic_repr = last_hidden_state[:, 0, :]  [B, 384]

    No permanent pooler.  No token-type embeddings.
    No teacher, no task heads, no MLM head.
    """

    def __init__(self, cfg: RmcEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Embeddings
        self.token_embeddings = nn.Embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            padding_idx=cfg.pad_id,
        )
        self.position_embeddings = nn.Embedding(cfg.max_seq_length, cfg.hidden_size)

        # Embedding pipeline normalization
        self.embed_ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.embed_dropout = nn.Dropout(cfg.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [_RmcTransformerBlock(cfg) for _ in range(cfg.num_layers)]
        )

        # Final LayerNorm
        self.final_ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

        # Apply initialization
        self._init_weights()

    def _init_weights(self) -> None:
        ir = self.cfg.initializer_range
        # Embeddings and linear weights: Normal(0, ir)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=ir)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=ir)
        # Zero the padding embedding row
        with torch.no_grad():
            self.token_embeddings.weight[self.cfg.pad_id].fill_(0.0)
        # LayerNorm defaults: weights=1, biases=0
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # MultiheadAttention stores Q/K/V in combined parameters, not
        # separate nn.Linear modules, so they need explicit initialization.
        for module in self.modules():
            if isinstance(module, nn.MultiheadAttention):
                if module.in_proj_weight is not None:
                    nn.init.normal_(
                        module.in_proj_weight,
                        mean=0.0,
                        std=ir,
                    )
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)

    def _validate_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be rank-2 (batch, seq), got rank-{input_ids.dim()}"
            )
        if attention_mask.dim() != 2:
            raise ValueError(
                f"attention_mask must be rank-2 (batch, seq), got "
                f"rank-{attention_mask.dim()}"
            )
        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                f"input_ids shape {tuple(input_ids.shape)} must match "
                f"attention_mask shape {tuple(attention_mask.shape)}"
            )
        batch, seq = input_ids.shape
        if batch == 0:
            raise ValueError("input_ids batch dimension must be > 0")
        if seq == 0:
            raise ValueError("input_ids sequence dimension must be > 0")
        if seq > self.cfg.max_seq_length:
            raise ValueError(
                f"Sequence length {seq} exceeds max_seq_length "
                f"{self.cfg.max_seq_length}"
            )
        # attention_mask must be binary
        mask_unique = attention_mask.unique()
        if mask_unique.numel() > 2 or (
            mask_unique.numel() == 2
            and not (
                (mask_unique[0] == 0 and mask_unique[1] == 1)
                or (mask_unique[0] == 1 and mask_unique[1] == 0)
            )
        ) or (mask_unique.numel() == 1 and mask_unique[0] not in (0, 1)):
            raise ValueError(
                "attention_mask must contain only 0 and 1 values"
            )
        # Every row must have at least one real token
        row_sums = attention_mask.sum(dim=1)
        if (row_sums == 0).any():
            raise ValueError(
                "attention_mask contains rows with all zeros — "
                "every sequence must have at least one attended token"
            )
        # Position 0 must be attended (semantic_repr uses it)
        if (attention_mask[:, 0] != 1).any():
            raise ValueError(
                "attention_mask[:, 0] must be 1 for every batch item — "
                "position 0 is used for semantic_repr"
            )

        # input_ids must be integer type
        if input_ids.dtype not in (torch.int64, torch.int32):
            raise ValueError(
                f"input_ids must be int64 or int32, got {input_ids.dtype}"
            )

        # attention_mask must be bool or integer type
        if attention_mask.dtype not in (torch.bool, torch.int64, torch.int32):
            raise ValueError(
                f"attention_mask must be bool or integer, got {attention_mask.dtype}"
            )

        # Both tensors must be on the same device
        if input_ids.device != attention_mask.device:
            raise ValueError(
                f"input_ids and attention_mask must be on the same device, "
                f"got {input_ids.device} vs {attention_mask.device}"
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> RmcEncoderOutput:
        self._validate_inputs(input_ids, attention_mask)

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Position IDs: 0, 1, ..., seq_len-1
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(
            batch_size, -1
        )

        # Key padding mask for MultiheadAttention: True = ignore
        key_padding_mask = (attention_mask == 0)

        # Embedding pipeline
        token_emb = self.token_embeddings(input_ids)
        pos_emb = self.position_embeddings(position_ids)
        x = self.embed_ln(token_emb + pos_emb)
        x = self.embed_dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        # Final LayerNorm
        x = self.final_ln(x)

        semantic_repr = x[:, 0, :]

        return RmcEncoderOutput(
            last_hidden_state=x,
            semantic_repr=semantic_repr,
        )

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())
