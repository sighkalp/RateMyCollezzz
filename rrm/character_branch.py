"""Character-level neural branch for the RRM (RRM 3.8).

Implements a byte-level parallel Conv1d architecture that captures
orthographic and noisy-text patterns complementary to the semantic
Transformer encoder.

Research provenance:
    - Character-level CNN: PAPER-DERIVED (Zhang, Zhao & LeCun, 2015)
    - Byte-level input: RRM EXPERIMENTAL DESIGN CHOICE
    - Parallel Conv1d with global max pooling: PAPER-DERIVED (Kim, 2014)
    - GELU activation: PAPER-DERIVED (Hendrycks & Gimpel, 2016)
    - LayerNorm: PAPER-DERIVED (Ba, Kiros & Hinton, 2016)
    - Dropout defaults, initializer_range 0.02: RRM EXPERIMENTAL DESIGN CHOICE

Architecture rule:
    RRM UNDERSTANDS. TRUST DECIDES.

This module produces character-level representation evidence only.
It does NOT make moderation decisions.
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
class RmcCharacterConfig:
    """Immutable configuration for the RMC character branch."""

    byte_vocab_size: int = 257
    pad_id: int = 0
    embedding_dim: int = 64
    max_byte_length: int = 512
    conv_channels: int = 128
    kernel_sizes: tuple[int, ...] = (3, 4, 5)
    character_dim: int = 64
    embedding_dropout: float = 0.1
    post_cnn_dropout: float = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.byte_vocab_size != 257:
            raise ValueError(
                f"byte_vocab_size must be 257 for the locked architecture, "
                f"got {self.byte_vocab_size}"
            )
        if self.pad_id != 0:
            raise ValueError(
                f"pad_id must be 0 for the locked architecture, "
                f"got {self.pad_id}"
            )
        if self.embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be > 0, got {self.embedding_dim}"
            )
        if self.max_byte_length <= 0:
            raise ValueError(
                f"max_byte_length must be > 0, got {self.max_byte_length}"
            )
        if self.conv_channels <= 0:
            raise ValueError(
                f"conv_channels must be > 0, got {self.conv_channels}"
            )
        if self.character_dim <= 0:
            raise ValueError(
                f"character_dim must be > 0, got {self.character_dim}"
            )
        if not self.kernel_sizes:
            raise ValueError("kernel_sizes must be non-empty")
        seen_kernels: set[int] = set()
        for k in self.kernel_sizes:
            if k <= 0:
                raise ValueError(
                    f"kernel sizes must be > 0, got {k}"
                )
            if k in seen_kernels:
                raise ValueError(
                    f"kernel sizes must be unique, duplicate: {k}"
                )
            seen_kernels.add(k)
        if self.max_byte_length < max(self.kernel_sizes):
            raise ValueError(
                f"max_byte_length ({self.max_byte_length}) must be >= "
                f"max kernel size ({max(self.kernel_sizes)})"
            )
        if not (0 <= self.embedding_dropout < 1):
            raise ValueError(
                f"embedding_dropout must be in [0, 1), "
                f"got {self.embedding_dropout}"
            )
        if not (0 <= self.post_cnn_dropout < 1):
            raise ValueError(
                f"post_cnn_dropout must be in [0, 1), "
                f"got {self.post_cnn_dropout}"
            )
        if self.initializer_range <= 0:
            raise ValueError(
                f"initializer_range must be > 0, got {self.initializer_range}"
            )
        if self.layer_norm_eps <= 0:
            raise ValueError(
                f"layer_norm_eps must be > 0, got {self.layer_norm_eps}"
            )


# ---------------------------------------------------------------------------
# Preprocessing output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ByteBatch:
    """Immutable batch of encoded bytes.

    Attributes
    ----------
    byte_ids : torch.Tensor
        Integer byte IDs.  Shape [B, max_byte_length], dtype torch.long.
        Real bytes map to IDs 1..256, padding is ID 0.
    byte_attention_mask : torch.Tensor
        Boolean mask.  Shape [B, max_byte_length], dtype torch.bool.
        True for real bytes, False for padding.
    """

    byte_ids: torch.Tensor
    byte_attention_mask: torch.Tensor


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

def encode_byte_batch(
    texts: list[str],
    config: RmcCharacterConfig,
) -> ByteBatch:
    """Encode a batch of raw strings into byte IDs and masks.

    Parameters
    ----------
    texts : list of str
        Raw review texts.
    config : RmcCharacterConfig
        Character branch configuration.

    Returns
    -------
    ByteBatch
        Fixed-length byte IDs and boolean attention mask.

    Raises
    ------
    ValueError
        If ``texts`` is empty or contains non-string elements.
    TypeError
        If ``config`` is not an ``RmcCharacterConfig``.

    Notes
    -----
    Encoding contract:
    - UTF-8 with ``errors="replace"`` (malformed Unicode → ASCII ``?``)
    - Raw byte ``b`` maps to model ID ``b + 1``
    - Padding ID is 0
    - Right-side truncation to ``config.max_byte_length``
    - Fixed output shape [B, max_byte_length]
    """
    if not isinstance(config, RmcCharacterConfig):
        raise TypeError(
            f"config must be RmcCharacterConfig, got {type(config).__name__}"
        )
    if not texts:
        raise ValueError("texts must be a non-empty list")

    batch_size = len(texts)
    max_len = config.max_byte_length

    byte_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(
                f"texts[{i}] must be str, got {type(text).__name__}"
            )
        raw_bytes = text.encode("utf-8", errors="replace")
        real_len = min(len(raw_bytes), max_len)
        for j in range(real_len):
            byte_ids[i, j] = raw_bytes[j] + 1  # b -> b+1, PAD=0
            mask[i, j] = True

    return ByteBatch(byte_ids=byte_ids, byte_attention_mask=mask)


# ---------------------------------------------------------------------------
# Character branch model
# ---------------------------------------------------------------------------

class RmcCharacterBranch(nn.Module):
    """Byte-level parallel Conv1d character branch.

    Permanent topology:

        byte_ids [B, L]
            -> Embedding(257, 64, padding_idx=0)
            -> embedding Dropout(0.1)
            -> explicit mask: x = x * mask.unsqueeze(-1)
            -> transpose to [B, 64, L]
            -> parallel Conv1d(64, 128, k=3/4/5)
            -> GELU
            -> mask-aware global max pooling per kernel
            -> concatenate [B, 384]
            -> Dropout(0.1)
            -> Linear(384, 64)
            -> LayerNorm(64)
            -> explicit empty-row zeroing
            -> character_repr [B, 64]

    Parameters
    ----------
    config : RmcCharacterConfig
        Immutable configuration.
    """

    def __init__(self, config: RmcCharacterConfig) -> None:
        super().__init__()
        self.cfg = config

        # Embeddings
        self.byte_embeddings = nn.Embedding(
            config.byte_vocab_size,
            config.embedding_dim,
            padding_idx=config.pad_id,
        )
        self.embed_dropout = nn.Dropout(config.embedding_dropout)

        # Parallel Conv1d branches
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=config.embedding_dim,
                out_channels=config.conv_channels,
                kernel_size=k,
                padding=0,
                bias=True,
            )
            for k in config.kernel_sizes
        ])

        # Post-CNN pipeline
        self.post_dropout = nn.Dropout(config.post_cnn_dropout)
        self.projection = nn.Linear(
            config.conv_channels * len(config.kernel_sizes),
            config.character_dim,
        )
        self.final_ln = nn.LayerNorm(
            config.character_dim, eps=config.layer_norm_eps
        )

        self._init_weights()

    def _init_weights(self) -> None:
        ir = self.cfg.initializer_range
        # Embeddings
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=ir)
        # Zero padding embedding row
        with torch.no_grad():
            self.byte_embeddings.weight[self.cfg.pad_id].fill_(0.0)
        # Conv1d weights and biases
        for conv in self.convs:
            nn.init.normal_(conv.weight, mean=0.0, std=ir)
            nn.init.zeros_(conv.bias)
        # Linear weights and biases
        nn.init.normal_(self.projection.weight, mean=0.0, std=ir)
        nn.init.zeros_(self.projection.bias)
        # LayerNorm
        nn.init.ones_(self.final_ln.weight)
        nn.init.zeros_(self.final_ln.bias)

    def _validate_inputs(
        self,
        byte_ids: torch.Tensor,
        byte_attention_mask: torch.Tensor,
    ) -> None:
        if byte_ids.dim() != 2:
            raise ValueError(
                f"byte_ids must be rank-2 [B, L], got rank-{byte_ids.dim()}"
            )
        if byte_attention_mask.dim() != 2:
            raise ValueError(
                f"byte_attention_mask must be rank-2 [B, L], got "
                f"rank-{byte_attention_mask.dim()}"
            )
        if byte_ids.shape != byte_attention_mask.shape:
            raise ValueError(
                f"byte_ids shape {tuple(byte_ids.shape)} must match "
                f"byte_attention_mask shape {tuple(byte_attention_mask.shape)}"
            )
        batch, seq_len = byte_ids.shape
        if batch == 0:
            raise ValueError("batch dimension must be > 0")
        max_kernel = max(self.cfg.kernel_sizes)
        if seq_len < max_kernel:
            raise ValueError(
                f"sequence length {seq_len} is less than max kernel size "
                f"{max_kernel}"
            )
        if seq_len > self.cfg.max_byte_length:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_byte_length "
                f"{self.cfg.max_byte_length}"
            )
        if byte_ids.dtype != torch.long:
            raise ValueError(
                f"byte_ids must be torch.long, got {byte_ids.dtype}"
            )
        if byte_attention_mask.dtype != torch.bool:
            raise ValueError(
                f"byte_attention_mask must be torch.bool, got "
                f"{byte_attention_mask.dtype}"
            )
        if byte_ids.device != byte_attention_mask.device:
            raise ValueError(
                f"byte_ids and byte_attention_mask must be on the same "
                f"device, got {byte_ids.device} vs {byte_attention_mask.device}"
            )
        # All byte IDs must be in [0, 256]
        id_min = byte_ids.min().item()
        id_max = byte_ids.max().item()
        if id_min < 0 or id_max > 256:
            raise ValueError(
                f"byte_ids must be in [0, 256], got range [{id_min}, {id_max}]"
            )
        # Attended positions must have IDs in [1, 256]
        attended_ids = byte_ids[byte_attention_mask]
        if attended_ids.numel() > 0:
            att_min = attended_ids.min().item()
            att_max = attended_ids.max().item()
            if att_min < 1 or att_max > 256:
                raise ValueError(
                    f"attended byte_ids must be in [1, 256], "
                    f"got range [{att_min}, {att_max}]"
                )

    def forward(
        self,
        byte_ids: torch.Tensor,
        byte_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute character-level representation.

        Parameters
        ----------
        byte_ids : torch.Tensor
            Byte ID tensor.  Shape [B, L], dtype torch.long.
            Real byte IDs are 1..256, padding is 0.
        byte_attention_mask : torch.Tensor
            Boolean mask.  Shape [B, L], dtype torch.bool.
            True for real bytes, False for padding.

        Returns
        -------
        torch.Tensor
            Character representation.  Shape [B, character_dim].
        """
        self._validate_inputs(byte_ids, byte_attention_mask)

        batch_size, seq_len = byte_ids.shape
        device = byte_ids.device

        # Embedding pipeline
        x = self.byte_embeddings(byte_ids)  # [B, L, embedding_dim]
        x = self.embed_dropout(x)

        # Explicit pre-conv masking: zero out padded positions
        x = x * byte_attention_mask.unsqueeze(-1)

        # Transpose for Conv1d: [B, embedding_dim, L]
        x = x.transpose(1, 2)

        # Parallel Conv1d branches
        pooled_outputs: list[torch.Tensor] = []
        for conv in self.convs:
            kernel_size = conv.kernel_size[0]
            # Convolution: [B, conv_channels, L_conv]
            features = conv(x)
            features = F.gelu(features)

            # Compute valid-window mask using unfold
            # unfold: [B, 1, L_conv, kernel_size]
            windows = byte_attention_mask.unfold(
                dimension=1, size=kernel_size, step=1
            )
            valid_window = windows.all(dim=-1)  # [B, L_conv]

            # Mask invalid positions to -inf before max pooling
            features = features.masked_fill(
                ~valid_window.unsqueeze(1),
                torch.finfo(features.dtype).min,
            )

            # Global max pooling over temporal dimension
            pooled = features.max(dim=-1).values  # [B, conv_channels]

            # Fallback: if no valid window, output zero vector
            has_valid = valid_window.any(dim=-1, keepdim=True)  # [B, 1]
            pooled = torch.where(
                has_valid,
                pooled,
                torch.zeros_like(pooled),
            )

            pooled_outputs.append(pooled)

        # Concatenate all kernel outputs: [B, conv_channels * num_kernels]
        x = torch.cat(pooled_outputs, dim=-1)

        # Post-CNN pipeline
        x = self.post_dropout(x)
        x = self.projection(x)
        x = self.final_ln(x)

        # Explicit empty-row zeroing: rows with no real bytes → exact zero
        has_real_byte = byte_attention_mask.any(dim=1, keepdim=True)
        x = torch.where(has_real_byte, x, torch.zeros_like(x))

        return x

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())
