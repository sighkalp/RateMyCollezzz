"""RMC SentencePiece tokenizer (RRM 3.5).

Implements the SentencePiece-based tokenizer infrastructure for the
future custom RRM encoder.  Does NOT train the production 22k tokenizer;
that remains gated by the RRM 3.6 encoder-strategy decision.

Research provenance:
    - SentencePiece Unigram: PAPER-DERIVED (Kudo & Richardson, 2018)
    - Config defaults: RRM EXPERIMENTAL DESIGN CHOICE
    - Special-token contract: RRM EXPERIMENTAL DESIGN CHOICE

Architecture rule:
    RRM UNDERSTANDS. TRUST DECIDES.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Union

import sentencepiece as spm
from sentencepiece import SentencePieceProcessor


# ---------------------------------------------------------------------------
# Hashing utilities
# ---------------------------------------------------------------------------

def compute_file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of *path*'s exact bytes.

    Uses a fixed-size read buffer so large files are handled safely.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RmcTokenizerConfig:
    """Immutable configuration for the RMC SentencePiece tokenizer.

    All fields are frozen after construction.  The production-oriented
    defaults target the agreed 22k vocabulary.  Unit tests override
    these with small values (e.g. 320–512) validated against tiny
    temporary corpora.

    Research provenance:
        - Default values: RRM EXPERIMENTAL DESIGN CHOICE
        - Not proven optimal; validated later in RRM 3.10.
    """

    model_type: str = "unigram"
    vocab_size: int = 22_000
    character_coverage: float = 1.0
    byte_fallback: bool = True
    normalization_rule_name: str = "identity"
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = -1
    eos_id: int = -1
    user_defined_symbols: Tuple[str, ...] = ("<cls>", "<sep>", "<mask>")
    hard_vocab_limit: bool = True
    shuffle_input_sentence: bool = False
    num_threads: int = 1
    max_seq_length: int = 256

    def __post_init__(self) -> None:
        if self.model_type not in ("unigram", "bpe"):
            raise ValueError(
                f"model_type must be 'unigram' or 'bpe', got {self.model_type!r}"
            )
        if self.vocab_size < 100:
            raise ValueError(
                f"vocab_size must be >= 100, got {self.vocab_size}"
            )
        if not (0.0 < self.character_coverage <= 1.0):
            raise ValueError(
                f"character_coverage must be in (0, 1], "
                f"got {self.character_coverage}"
            )
        if self.pad_id < 0 or self.unk_id < 0:
            raise ValueError("pad_id and unk_id must be non-negative")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _build_manifest(
    config: RmcTokenizerConfig,
    corpus_path: Optional[Path],
    model_path: Path,
    vocab_path: Optional[Path],
    actual_vocab_size: int,
    special_ids: dict,
) -> dict:
    """Build the tokenizer manifest dictionary.

    Records training provenance, configuration, and artifact hashes.
    The manifest is intentionally NOT bit-identical across runs because
    ``created_at`` is included as metadata.  Deterministic identity
    (if needed later) should be derived from stable fields, excluding
    the timestamp.
    """
    try:
        sp_version = spm.__version__
    except AttributeError:
        sp_version = "unknown"

    manifest: dict = {
        "manifest_version": "1.0",
        "tokenizer_name": "rmc-tokenizer",
        "model_type": config.model_type,
        "vocab_size_target": config.vocab_size,
        "vocab_size_actual": actual_vocab_size,
        "normalization_rule_name": config.normalization_rule_name,
        "character_coverage": config.character_coverage,
        "byte_fallback": config.byte_fallback,
        "special_tokens": {
            "pad_piece": "<pad>",
            "pad_id": special_ids.get("pad_id"),
            "unk_piece": "<unk>",
            "unk_id": special_ids.get("unk_id"),
            "cls_piece": "<cls>",
            "cls_id": special_ids.get("cls_id"),
            "sep_piece": "<sep>",
            "sep_id": special_ids.get("sep_id"),
            "mask_piece": "<mask>",
            "mask_id": special_ids.get("mask_id"),
        },
        "sentencepiece_version": sp_version,
        "corpus_sha256": (
            compute_file_sha256(corpus_path) if corpus_path else None
        ),
        "corpus_record_count": _count_corpus_records(corpus_path),
        "corpus_source": str(corpus_path) if corpus_path else None,
        "corpus_policy": None,
        "model_sha256": compute_file_sha256(model_path),
        "vocab_sha256": (
            compute_file_sha256(vocab_path) if vocab_path else None
        ),
        "training_options": _config_to_dict(config),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return manifest


def _count_corpus_records(corpus_path: Optional[Path]) -> Optional[int]:
    """Count newline-terminated records in the corpus file."""
    if corpus_path is None:
        return None
    count = 0
    with corpus_path.open("rb") as fh:
        for _ in fh:
            count += 1
    return count


def _config_to_dict(config: RmcTokenizerConfig) -> dict:
    """Serialize config to a JSON-safe dictionary."""
    result = {}
    for f in fields(config):
        val = getattr(config, f.name)
        if isinstance(val, tuple):
            val = list(val)
        result[f.name] = val
    return result


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class RmcTokenizer:
    """SentencePiece-based tokenizer for the RMC corpus.

    This class wraps ``sentencepiece.SentencePieceProcessor`` and
    provides a small API suitable for the future custom RRM encoder.

    Important:
        This implementation does NOT train the production 22k tokenizer.
        Production training is gated by the RRM 3.6 encoder-strategy
        decision.  During RRM 3.5, training uses small temporary corpora
        for validation only.

    Architecture rule:
        RRM UNDERSTANDS. TRUST DECIDES.

    Parameters
    ----------
    processor : SentencePieceProcessor
        Loaded SentencePiece model.
    config : RmcTokenizerConfig
        Configuration used (or that would be used) for training.
    manifest : dict or None
        Loaded manifest, if available.
    """

    # Special-token piece strings
    PAD_PIECE = "<pad>"
    UNK_PIECE = "<unk>"
    CLS_PIECE = "<cls>"
    SEP_PIECE = "<sep>"
    MASK_PIECE = "<mask>"

    def __init__(
        self,
        processor: SentencePieceProcessor,
        config: RmcTokenizerConfig,
        manifest: Optional[dict] = None,
    ) -> None:
        self._processor = processor
        self._config = config
        self._manifest = manifest

        # Resolve special-token IDs from the loaded model
        self._pad_id = processor.piece_to_id(self.PAD_PIECE)
        self._unk_id = processor.piece_to_id(self.UNK_PIECE)
        self._cls_id = processor.piece_to_id(self.CLS_PIECE)
        self._sep_id = processor.piece_to_id(self.SEP_PIECE)
        self._mask_id = processor.piece_to_id(self.MASK_PIECE)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def unk_id(self) -> int:
        return self._unk_id

    @property
    def cls_id(self) -> int:
        return self._cls_id

    @property
    def sep_id(self) -> int:
        return self._sep_id

    @property
    def mask_id(self) -> int:
        return self._mask_id

    @property
    def vocab_size(self) -> int:
        return self._processor.get_piece_size()

    @property
    def manifest(self) -> Optional[dict]:
        return self._manifest

    @property
    def config(self) -> RmcTokenizerConfig:
        return self._config

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        corpus_path: Union[str, Path],
        output_dir: Union[str, Path],
        config: Optional[RmcTokenizerConfig] = None,
        corpus_source: str = "unspecified",
        corpus_policy: Optional[str] = None,
    ) -> "RmcTokenizer":
        """Train a SentencePiece tokenizer on *corpus_path*.

        Parameters
        ----------
        corpus_path : path-like
            Path to a text file with one review per line.
        output_dir : path-like
            Directory where ``.model``, ``.vocab``, and
            ``tokenizer_manifest.json`` will be written.  Must resolve
            **outside** the repository root.
        config : RmcTokenizerConfig, optional
            Tokenizer configuration.  Uses production defaults if not
            provided.  Unit tests should pass a small-vocabulary config.
        corpus_source : str
            Identifier for the corpus source (e.g. ``"train-split-v1"``).
        corpus_policy : str, optional
            Policy note (e.g. ``"train-split-only"``).

        Returns
        -------
        RmcTokenizer
            Loaded tokenizer ready for encode/decode.

        Raises
        ------
        FileExistsError
            Any tokenizer artifact already exists in *output_dir*.
            Tokenizer artifacts cannot be replaced in this phase;
            version management will be designed later.
        ValueError
            Output directory resolves inside the repository.
        RuntimeError
            Special-token IDs do not match the expected contract, or
            hard_vocab_limit is True but the actual vocabulary size
            does not match the requested size.
        """
        corpus_path = Path(corpus_path)
        output_dir = Path(output_dir)
        config = config or RmcTokenizerConfig()

        # Safety: output must be outside the repository
        repo_root = _find_repo_root()
        resolved_output = output_dir.resolve()
        resolved_repo = repo_root.resolve()
        try:
            resolved_output.relative_to(resolved_repo)
        except ValueError:
            pass  # outside repo — good
        else:
            raise ValueError(
                f"output_dir {output_dir} resolves inside the repository "
                f"root {repo_root}.  Tokenizer artifacts must be stored "
                f"outside the repository."
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / "rmc-tokenizer.model"
        vocab_path = output_dir / "rmc-tokenizer.vocab"
        manifest_path = output_dir / "tokenizer_manifest.json"

        # Refuse to overwrite any existing artifact — version management
        # can introduce replacement behavior in a later phase.
        for p in (model_path, vocab_path, manifest_path):
            if p.exists():
                raise FileExistsError(
                    f"Tokenizer artifact already exists: {p}.  "
                    f"Tokenizer artifacts cannot be replaced in this phase."
                )

        # Compute corpus hash before training
        corpus_sha256 = compute_file_sha256(corpus_path)
        corpus_record_count = _count_corpus_records(corpus_path)

        # Build SentencePiece Trainer keyword arguments
        sp_kwargs = dict(
            input=str(corpus_path),
            model_prefix=str(model_path.with_suffix("")),
            model_type=config.model_type,
            vocab_size=config.vocab_size,
            character_coverage=config.character_coverage,
            byte_fallback=config.byte_fallback,
            normalization_rule_name=config.normalization_rule_name,
            pad_id=config.pad_id,
            unk_id=config.unk_id,
            bos_id=config.bos_id,
            eos_id=config.eos_id,
            user_defined_symbols=list(config.user_defined_symbols),
            hard_vocab_limit=config.hard_vocab_limit,
            shuffle_input_sentence=config.shuffle_input_sentence,
            num_threads=config.num_threads,
        )

        spm.SentencePieceTrainer.Train(**sp_kwargs)

        # Verify the model was created
        if not model_path.exists():
            raise RuntimeError(
                f"SentencePiece training completed but model file not found: {model_path}"
            )

        # Load and verify
        tokenizer = cls.load(
            model_path,
            config=config,
            manifest=None,
            verify_special_ids=True,
        )

        # Hard-vocab postcondition: when hard_vocab_limit=True, the
        # actual vocabulary size must equal the requested size.
        # SentencePiece enforces this internally, but we verify it
        # explicitly so the contract is not silently delegated.
        if config.hard_vocab_limit:
            actual = tokenizer.vocab_size
            target = config.vocab_size
            if actual != target:
                raise RuntimeError(
                    f"hard_vocab_limit=True but actual vocabulary size "
                    f"({actual}) does not match requested size ({target})."
                )

        # Build manifest
        special_ids = {
            "pad_id": tokenizer._pad_id,
            "unk_id": tokenizer._unk_id,
            "cls_id": tokenizer._cls_id,
            "sep_id": tokenizer._sep_id,
            "mask_id": tokenizer._mask_id,
        }
        manifest = _build_manifest(
            config=config,
            corpus_path=corpus_path,
            model_path=model_path,
            vocab_path=vocab_path if vocab_path.exists() else None,
            actual_vocab_size=tokenizer.vocab_size,
            special_ids=special_ids,
        )
        manifest["corpus_source"] = corpus_source
        manifest["corpus_policy"] = corpus_policy

        # Write manifest atomically
        tmp_manifest = manifest_path.with_suffix(".tmp")
        tmp_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False)
        )
        tmp_manifest.replace(manifest_path)

        # Return with manifest attached
        return cls(
            processor=tokenizer._processor,
            config=config,
            manifest=manifest,
        )

    @classmethod
    def load(
        cls,
        model_path: Union[str, Path],
        config: Optional[RmcTokenizerConfig] = None,
        manifest: Optional[dict] = None,
        verify_special_ids: bool = True,
    ) -> "RmcTokenizer":
        """Load a SentencePiece tokenizer from *model_path*.

        Parameters
        ----------
        model_path : path-like
            Path to the ``.model`` file produced by training.
        config : RmcTokenizerConfig, optional
            If provided, used for max_seq_length and other settings.
            Defaults to production-oriented values.
        manifest : dict, optional
            Pre-parsed manifest dictionary.  If None and a
            ``tokenizer_manifest.json`` exists beside the model, it is
            loaded automatically.
        verify_special_ids : bool
            If True, raise RuntimeError when special-token IDs do not
            match the expected contract (<pad>=0, <unk>=1, <cls>=2,
            <sep>=3, <mask>=4).

        Raises
        ------
        ValueError
            Manifest exists but is malformed JSON, has wrong root type,
            or is missing required fields.  Also raised when model or
            vocab SHA-256 hashes do not match the manifest.
        RuntimeError
            Special-token IDs violate the contract.
        """
        model_path = Path(model_path)
        config = config or RmcTokenizerConfig()

        processor = SentencePieceProcessor()
        processor.Load(str(model_path))

        tokenizer = cls(
            processor=processor,
            config=config,
            manifest=manifest,
        )

        if verify_special_ids:
            tokenizer._verify_special_ids()

        # Auto-load and validate manifest if not provided
        if manifest is None:
            mf_path = model_path.parent / "tokenizer_manifest.json"
            if mf_path.exists():
                tokenizer._manifest = cls._load_and_validate_manifest(
                    mf_path, model_path
                )

        return tokenizer

    @staticmethod
    def _load_and_validate_manifest(
        mf_path: Path, model_path: Path
    ) -> dict:
        """Load, validate schema, and verify hashes of a manifest file.

        Raises ValueError on any integrity failure.
        """
        try:
            raw = mf_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(
                f"Tokenizer manifest exists but cannot be read: {mf_path}"
            ) from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Tokenizer manifest is malformed JSON: {mf_path}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                f"Tokenizer manifest root must be a JSON object, "
                f"got {type(data).__name__}: {mf_path}"
            )

        # Validate required top-level fields
        required_fields = [
            "manifest_version",
            "tokenizer_name",
            "model_type",
            "vocab_size_target",
            "vocab_size_actual",
            "normalization_rule_name",
            "character_coverage",
            "byte_fallback",
            "special_tokens",
            "sentencepiece_version",
            "corpus_sha256",
            "corpus_record_count",
            "corpus_source",
            "corpus_policy",
            "model_sha256",
            "vocab_sha256",
            "training_options",
            "created_at",
        ]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(
                f"Tokenizer manifest is missing required fields: "
                f"{', '.join(missing)}"
            )

        # Validate special_tokens sub-fields
        st = data.get("special_tokens", {})
        required_st_fields = [
            "pad_piece", "pad_id",
            "unk_piece", "unk_id",
            "cls_piece", "cls_id",
            "sep_piece", "sep_id",
            "mask_piece", "mask_id",
        ]
        missing_st = [f for f in required_st_fields if f not in st]
        if missing_st:
            raise ValueError(
                f"Tokenizer manifest special_tokens is missing required "
                f"fields: {', '.join(missing_st)}"
            )

        # Verify model SHA-256
        RmcTokenizer._verify_file_hash(model_path, data["model_sha256"], "model")

        # Verify vocab SHA-256 if vocab file exists
        vocab_path = model_path.parent / "rmc-tokenizer.vocab"
        if vocab_path.exists() and data.get("vocab_sha256"):
            RmcTokenizer._verify_file_hash(vocab_path, data["vocab_sha256"], "vocab")

        return data

    @staticmethod
    def _verify_file_hash(
        file_path: Path, expected_hash: Optional[str], label: str
    ) -> None:
        """Raise ValueError if file hash does not match expected."""
        if expected_hash is None:
            return
        actual = compute_file_sha256(file_path)
        if actual != expected_hash:
            raise ValueError(
                f"Tokenizer {label} SHA-256 mismatch for {file_path}: "
                f"manifest={expected_hash}, actual={actual}"
            )

    def _verify_special_ids(self) -> None:
        """Raise RuntimeError if special-token IDs violate the contract."""
        expected = {
            self.PAD_PIECE: 0,
            self.UNK_PIECE: 1,
            self.CLS_PIECE: 2,
            self.SEP_PIECE: 3,
            self.MASK_PIECE: 4,
        }
        actual = {
            self.PAD_PIECE: self._pad_id,
            self.UNK_PIECE: self._unk_id,
            self.CLS_PIECE: self._cls_id,
            self.SEP_PIECE: self._sep_id,
            self.MASK_PIECE: self._mask_id,
        }
        mismatches = {
            piece: (expected[piece], actual[piece])
            for piece in expected
            if expected[piece] != actual[piece]
        }
        if mismatches:
            raise RuntimeError(
                "Special-token ID contract violated.  Expected "
                "<pad>=0, <unk>=1, <cls>=2, <sep>=3, <mask>=4.  "
                f"Mismatches: {mismatches}"
            )

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> List[int]:
        """Encode *text* into a list of token IDs.

        Parameters
        ----------
        text : str
            Raw review text.  Must not be None.
        add_special_tokens : bool
            If True, prepend <cls> and append <sep>.
        max_length : int, optional
            Maximum sequence length including special tokens.
        truncation : bool
            If True, truncate to max_length.

        Returns
        -------
        list of int
        """
        if text is None:
            raise TypeError("text must be a str, not None")
        if not isinstance(text, str):
            raise TypeError(
                f"text must be a str, got {type(text).__name__}"
            )

        max_len = max_length or self._config.max_seq_length

        pieces = self._processor.EncodeAsPieces(text)
        ids: List[int] = [self._processor.piece_to_id(p) for p in pieces]

        if add_special_tokens:
            ids = [self._cls_id] + ids + [self._sep_id]

        if truncation and max_len is not None:
            ids = ids[:max_len]

        return ids

    def decode(
        self,
        ids: List[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs back to text.

        Parameters
        ----------
        ids : list of int
            Token IDs.
        skip_special_tokens : bool
            If True, omit special-token pieces from output.

        Returns
        -------
        str
        """
        if skip_special_tokens:
            special = {
                self._pad_id,
                self._cls_id,
                self._sep_id,
                self._mask_id,
            }
            ids = [i for i in ids if i not in special]

        # Filter out-of-range IDs
        vocab_size = self._processor.get_piece_size()
        ids = [i for i in ids if 0 <= i < vocab_size]

        return self._processor.DecodeIds(ids)

    # ------------------------------------------------------------------
    # Batch encode
    # ------------------------------------------------------------------

    def batch_encode(
        self,
        texts: List[str],
        *,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> dict:
        """Encode a batch of texts.

        Parameters
        ----------
        texts : list of str
        max_length : int, optional
        padding : bool
            If True, pad all sequences to the longest in the batch
            (or max_length if set).
        truncation : bool
            If True, truncate sequences.

        Returns
        -------
        dict with keys:
            input_ids : list of list of int
            attention_mask : list of list of int
        """
        max_len = max_length or self._config.max_seq_length

        encoded = [
            self.encode(
                t,
                add_special_tokens=True,
                max_length=max_len if truncation else None,
                truncation=truncation,
            )
            for t in texts
        ]

        if padding:
            pad_len = min(max_len, max(len(e) for e in encoded))
            pad_len = max(pad_len, max(len(e) for e in encoded))
            pad_id = self._pad_id

            input_ids = []
            attention_mask = []
            for seq in encoded:
                if truncation:
                    seq = seq[:max_len]
                pad_count = max_len - len(seq)
                input_ids.append(seq + [pad_id] * pad_count)
                attention_mask.append([1] * len(seq) + [0] * pad_count)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        else:
            return {
                "input_ids": encoded,
                "attention_mask": [
                    [1] * len(seq) for seq in encoded
                ],
            }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def compute_diagnostics(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        language_slices: Optional[dict] = None,
    ) -> dict:
        """Compute lightweight tokenizer quality diagnostics.

        Parameters
        ----------
        texts : list of str
            Review texts to analyze.
        max_length : int, optional
            Sequence length cap for truncation metrics.
        language_slices : dict, optional
            Mapping of slice_name -> list of texts for stratified
            diagnostics.  Slice names must match DATASET_SPEC language
            categories (ENGLISH, HINGLISH, ROMAN_HINDI, OTHER, MIXED_OTHER).

        Returns
        -------
        dict
        """
        max_len = max_length or self._config.max_seq_length
        unk_id = self._unk_id
        pad_id = self._pad_id
        vocab_size = self._processor.get_piece_size()

        def _slice_metrics(slice_texts: List[str]) -> dict:
            token_counts = []
            unk_count = 0
            byte_fb_count = 0
            total_tokens = 0
            over_max = 0
            piece_counter: dict = {}

            for text in slice_texts:
                if not text:
                    ids = [self._cls_id, self._sep_id]
                else:
                    ids = self.encode(
                        text, add_special_tokens=True,
                        max_length=max_len, truncation=True,
                    )
                token_counts.append(len(ids))
                total_tokens += len(ids)

                if len(ids) > max_len:
                    over_max += 1

                for tid in ids:
                    piece_counter[tid] = piece_counter.get(tid, 0) + 1
                    if tid == unk_id:
                        unk_count += 1
                    # Byte-fallback pieces in SentencePiece are named
                    # <0xNN> where NN is a hex byte value
                    if tid >= vocab_size - 256 and tid < vocab_size:
                        byte_fb_count += 1

            n = len(slice_texts)
            import statistics
            avg_tok = total_tokens / n if n else 0.0
            med_tok = statistics.median(token_counts) if token_counts else 0.0
            sorted_tok = sorted(token_counts)
            p95_idx = min(int(n * 0.95), n - 1) if n else 0
            p95_tok = sorted_tok[p95_idx] if sorted_tok else 0.0

            return {
                "token_count": total_tokens,
                "avg_tokens_per_review": round(avg_tok, 2),
                "median_tokens_per_review": round(med_tok, 2),
                "p95_tokens_per_review": round(p95_tok, 2),
                "unk_rate": round(unk_count / total_tokens, 6)
                if total_tokens else 0.0,
                "byte_fallback_rate": round(byte_fb_count / total_tokens, 6)
                if total_tokens else 0.0,
                "fraction_over_max_length": round(over_max / n, 6)
                if n else 0.0,
                "vocab_utilization": len(piece_counter) / vocab_size
                if vocab_size else 0.0,
            }

        result: dict = {"overall": _slice_metrics(texts)}

        if language_slices:
            for name, slice_texts in language_slices.items():
                result[name] = _slice_metrics(slice_texts)

        return result


# ---------------------------------------------------------------------------
# Repository root helper
# ---------------------------------------------------------------------------

def _find_repo_root(start: Optional[Path] = None) -> Path:
    """Find the repository root by walking up to find .git or a known marker."""
    current = (start or Path(__file__).resolve()).parent
    while True:
        if (current / ".git").exists() or (current / "ARCHITECTURE.md").exists():
            return current
        parent = current.parent
        if parent == current:
            return Path.cwd()
        current = parent
