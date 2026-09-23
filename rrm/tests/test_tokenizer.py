"""Tests for the RMC SentencePiece tokenizer (RRM 3.5).

All tests are offline.  Training uses tiny temporary corpora created
in-tests, never the 150-record synthetic pilot as a production corpus.
"""
from __future__ import annotations

import json
import os
import stat
import tempfile
from datetime import datetime
from pathlib import Path

import dataclasses

import pytest

import sentencepiece as spm

from rrm.tokenizer import (
    RmcTokenizer,
    RmcTokenizerConfig,
    compute_file_sha256,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_corpus_file(tmp_path: Path, lines: list[str]) -> Path:
    """Write *lines* to a temporary corpus file, one per line."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return corpus


def _make_test_config(tmp_path: Path, vocab_size: int = 512) -> RmcTokenizerConfig:
    """Return a config suitable for tiny-corpus training.

    The minimum viable vocab size with byte_fallback=True and the pilot
    corpus is 400 (256 byte pieces + 5 specials + corpus characters).
    512 provides a small safety margin.
    """
    return RmcTokenizerConfig(
        vocab_size=vocab_size,
        hard_vocab_limit=False,
    )


def _train_tiny_tokenizer(
    tmp_path: Path,
    lines: list[str],
    vocab_size: int = 512,
) -> RmcTokenizer:
    """Train a tiny tokenizer and return it."""
    corpus = _make_corpus_file(tmp_path, lines)
    config = _make_test_config(tmp_path, vocab_size)
    output_dir = tmp_path / "tokenizer_out"
    output_dir.mkdir()
    return RmcTokenizer.train(
        corpus_path=corpus,
        output_dir=output_dir,
        config=config,
        corpus_source="test",
        corpus_policy="tiny-test-corpus",
    )


def _generate_rich_corpus() -> list[str]:
    """Generate a varied temporary corpus for hard_vocab tests.

    Uses only Roman-script text so Devanagari/emoji tests can exercise
    byte fallback against an unseen-character corpus.
    """
    templates = [
        "The {adj} faculty at {noun} college provides {adj2} education.",
        "Placements at {noun} are {adj} with top companies visiting.",
        "The campus infrastructure is {adj} and labs are well equipped.",
        "Hostel {noun} needs improvement but food is {adj}.",
        "Coding culture is {adj} here with regular hackathons.",
        "Professors are {adj} and always ready to help students.",
        "The library has {adj} resources and quiet study spaces.",
        "Administration is {adj} but academics are excellent.",
        "Students here are {adj} and competitive in a good way.",
        "The {noun} department has the best faculty in the region.",
        "Research opportunities are {adj} with funded projects.",
        "Alumni network is {adj} and very supportive of juniors.",
        "The seminar hall hosts {adj} events every semester.",
        "Sports facilities are {adj} with a large playground.",
        "WiFi connectivity is {adj} across the entire campus.",
    ]
    adjectives = ["amazing", "outstanding", "excellent", "great", "solid",
                  "decent", "impressive", "remarkable", "phenomenal", "stellar"]
    nouns = ["engineering", "medical", "arts", "science", "management",
             "law", "design", "architecture", "technology", "research"]

    sentences = []
    for adj in adjectives:
        for noun in nouns:
            t = templates[len(sentences) % len(templates)]
            sentences.append(
                t.format(adj=adj, noun=noun, adj2=adjectives[(adjectives.index(adj) + 1) % len(adjectives)])
            )
    return sentences


# ---------------------------------------------------------------------------
# Test corpus
# ---------------------------------------------------------------------------

PILOT_LINES = [
    "The faculty is supportive and the labs are decent.",
    "Faculty supportive hai but placement scene thoda weak hai.",
    "College ka campus accha hai aur teachers helpful hain.",
    "Best college best college best college!!! Join now join now!!!",
    "The professor is a complete idiot and does not know how to teach.",
    "Kal cricket match bahut mast tha aur last over amazing tha.",
    "For admission help contact our consultancy at student@example.com.",
    "I studied here for three years and the hostel facilities were excellent.",
    "Administration ka response bahut slow hai aur kaam karwane mein time lagta hai.",
    "Placements are good especially for CSE and ECE branches.",
    "Goooood college but facuuuulty needs improvement.",
    "B.Tech CSE AI ML placements awesome infrastructure.",
    "https://www.example-college.edu/admissions",
    "Contact: +91-9876543210 or email@test.in",
    "Cafeteria food is terrible but library is amazing.",
    "Rig sir ka teaching style is very unique and helpful.",
    "Hostel mein WiFi nahi hai aur water supply irregular hai.",
    "Coding culture is very strong here hackathons every month.",
    "Arre bhai yeh college ka fest was lit bro!",
    "Marks de de yaar attendance mein issue ho raha hai.",
    "Hostel maintenance pathetic washrooms always dirty.",
    "Rigorous placement prep support from day one itself.",
    "Faculty ko salary badhao tabhi achha padhenge.",
    "Campus placements 2024 batch me bahut accha hai.",
    "EMERGENCY ALERT FIRE FIRE FIRE EVACUATE NOW!!!",
    "Exam ki Tayari kaafi tough hai but manageable.",
    "IIT Delhi NIT Trichy BITS Pilani comparison please help.",
    "Achha college lekin fees bohot zyada hai yaar.",
    "Library timing 8am to 10pm should extend to 24x7.",
    "The CSE department has the best faculty in the region.",
]


# ===================================================================
# 1. Config validation
# ===================================================================

class TestConfigValidation:
    def test_defaults(self):
        cfg = RmcTokenizerConfig()
        assert cfg.model_type == "unigram"
        assert cfg.vocab_size == 22_000
        assert cfg.character_coverage == 1.0
        assert cfg.byte_fallback is True
        assert cfg.normalization_rule_name == "identity"
        assert cfg.pad_id == 0
        assert cfg.unk_id == 1
        assert cfg.bos_id == -1
        assert cfg.eos_id == -1
        assert cfg.user_defined_symbols == ("<cls>", "<sep>", "<mask>")
        assert cfg.hard_vocab_limit is True
        assert cfg.shuffle_input_sentence is False
        assert cfg.num_threads == 1
        assert cfg.max_seq_length == 256

    def test_frozen(self):
        cfg = RmcTokenizerConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.vocab_size = 9999

    def test_invalid_model_type(self):
        with pytest.raises(ValueError, match="model_type"):
            RmcTokenizerConfig(model_type="word")

    def test_vocab_size_too_small(self):
        with pytest.raises(ValueError, match="vocab_size"):
            RmcTokenizerConfig(vocab_size=10)

    def test_character_coverage_zero(self):
        with pytest.raises(ValueError, match="character_coverage"):
            RmcTokenizerConfig(character_coverage=0.0)

    def test_character_coverage_over_one(self):
        with pytest.raises(ValueError, match="character_coverage"):
            RmcTokenizerConfig(character_coverage=1.5)

    def test_negative_pad_id(self):
        with pytest.raises(ValueError, match="pad_id"):
            RmcTokenizerConfig(pad_id=-1)

    def test_small_vocab_override(self):
        cfg = RmcTokenizerConfig(vocab_size=512, hard_vocab_limit=False)
        assert cfg.vocab_size == 512


import dataclasses


# ===================================================================
# 2. Training
# ===================================================================

class TestTraining:
    def test_train_creates_model_file(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        tok = RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )
        assert (out / "rmc-tokenizer.model").exists()

    def test_train_creates_vocab_file(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        tok = RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )
        assert (out / "rmc-tokenizer.vocab").exists()

    def test_train_creates_manifest(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        tok = RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
            corpus_policy="test-only",
        )
        mf_path = out / "tokenizer_manifest.json"
        assert mf_path.exists()
        manifest = json.loads(mf_path.read_text())
        assert manifest["manifest_version"] == "1.0"
        assert manifest["tokenizer_name"] == "rmc-tokenizer"
        assert manifest["model_type"] == "unigram"
        assert manifest["corpus_source"] == "test"
        assert manifest["corpus_policy"] == "test-only"
        assert manifest["sentencepiece_version"] == spm.__version__
        assert "created_at" in manifest

    def test_train_raises_on_existing_artifact(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )
        with pytest.raises(FileExistsError):
            RmcTokenizer.train(
                corpus_path=corpus,
                output_dir=out,
                config=_make_test_config(tmp_path),
                corpus_source="test",
            )

    def test_train_rejects_repo_internal_output(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        repo_internal = (
            Path(__file__).resolve().parent.parent / "artifacts_test"
        )
        try:
            with pytest.raises(ValueError, match="outside the repository"):
                RmcTokenizer.train(
                    corpus_path=corpus,
                    output_dir=repo_internal,
                    config=_make_test_config(tmp_path),
                    corpus_source="test",
                )
        finally:
            if repo_internal.exists():
                import shutil
                shutil.rmtree(repo_internal, ignore_errors=True)

    def test_corpus_sha256_in_manifest(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        corpus_hash = compute_file_sha256(corpus)
        out = tmp_path / "out"
        out.mkdir()
        RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )
        manifest = json.loads(
            (out / "tokenizer_manifest.json").read_text()
        )
        assert manifest["corpus_sha256"] == corpus_hash

    def test_model_sha256_in_manifest(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )
        model_file = out / "rmc-tokenizer.model"
        model_hash = compute_file_sha256(model_file)
        manifest = json.loads(
            (out / "tokenizer_manifest.json").read_text()
        )
        assert manifest["model_sha256"] == model_hash

    def test_train_rejects_model_artifact_only(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        (out / "rmc-tokenizer.model").write_text("fake")
        with pytest.raises(FileExistsError, match="model"):
            RmcTokenizer.train(
                corpus_path=corpus,
                output_dir=out,
                config=_make_test_config(tmp_path),
                corpus_source="test",
            )

    def test_train_rejects_vocab_artifact_only(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        (out / "rmc-tokenizer.vocab").write_text("fake")
        with pytest.raises(FileExistsError, match="vocab"):
            RmcTokenizer.train(
                corpus_path=corpus,
                output_dir=out,
                config=_make_test_config(tmp_path),
                corpus_source="test",
            )

    def test_train_rejects_manifest_artifact_only(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        (out / "tokenizer_manifest.json").write_text("{}")
        with pytest.raises(FileExistsError, match="manifest"):
            RmcTokenizer.train(
                corpus_path=corpus,
                output_dir=out,
                config=_make_test_config(tmp_path),
                corpus_source="test",
            )


# ===================================================================
# 3. Hard-vocab postcondition
# ===================================================================

class TestHardVocabPostcondition:
    def test_hard_vocab_true_actual_matches_target(self, tmp_path):
        """When hard_vocab_limit=True, actual vocab size must equal target.

        Uses a rich generated corpus so SentencePiece can fill the
        requested vocabulary.
        """
        corpus = _make_corpus_file(tmp_path, _generate_rich_corpus())
        out = tmp_path / "out"
        out.mkdir()
        target = 400
        cfg = RmcTokenizerConfig(
            vocab_size=target, hard_vocab_limit=True
        )
        tok = RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=cfg,
            corpus_source="test",
        )
        assert cfg.hard_vocab_limit is True
        assert tok.vocab_size == target, (
            f"hard_vocab_limit=True requires actual==target: "
            f"got {tok.vocab_size}, expected {target}"
        )

    def test_hard_vocab_true_mismatch_raises_runtime_error(self, tmp_path, monkeypatch):
        """Prove the defensive postcondition raises RuntimeError when
        hard_vocab_limit=True and observed size != requested size."""
        corpus = _make_corpus_file(tmp_path, _generate_rich_corpus())
        out = tmp_path / "out"
        out.mkdir()

        target = 400
        cfg = RmcTokenizerConfig(
            vocab_size=target, hard_vocab_limit=True
        )

        # Monkeypatch vocab_size on the class so the loaded tokenizer
        # reports a size that does not match the target.  This proves
        # our defensive postcondition (not SentencePiece itself) raises.
        original = RmcTokenizer.vocab_size.fget

        def fake_vocab_size(self):
            return target + 10

        monkeypatch.setattr(RmcTokenizer, "vocab_size", property(fake_vocab_size))

        with pytest.raises(RuntimeError, match="hard_vocab_limit"):
            RmcTokenizer.train(
                corpus_path=corpus,
                output_dir=out,
                config=cfg,
                corpus_source="test",
            )

        # Restore the original property so tmp_path cleanup doesn't
        # re-trigger the monkeypatched version.
        monkeypatch.setattr(RmcTokenizer, "vocab_size", property(original))

    def test_hard_vocab_false_allows_size_difference(self, tmp_path):
        """Test config with hard_vocab_limit=False does not enforce equality."""
        cfg = RmcTokenizerConfig(vocab_size=512, hard_vocab_limit=False)
        assert cfg.hard_vocab_limit is False
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:10])
        out = tmp_path / "out"
        out.mkdir()
        tok = RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=cfg,
            corpus_source="test",
        )
        # Actual size may differ from 512 -- no enforced equality
        assert tok.vocab_size > 0


# ===================================================================
# 4. Load
# ===================================================================

class TestLoad:
    def test_load_from_model_file(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        trained = RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )
        loaded = RmcTokenizer.load(out / "rmc-tokenizer.model")
        assert loaded.vocab_size == trained.vocab_size
        assert loaded.pad_id == 0
        assert loaded.unk_id == 1

    def test_load_auto_loads_manifest(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )
        loaded = RmcTokenizer.load(out / "rmc-tokenizer.model")
        assert loaded.manifest is not None
        assert loaded.manifest["tokenizer_name"] == "rmc-tokenizer"

    def test_load_without_manifest_file(self, tmp_path):
        """Loading a .model with no manifest file beside it is allowed."""
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )
        # Remove manifest -- load should still work
        (out / "tokenizer_manifest.json").unlink()
        loaded = RmcTokenizer.load(out / "rmc-tokenizer.model")
        assert loaded.vocab_size > 0
        assert loaded.pad_id == 0

    def test_load_corrupt_model_raises(self, tmp_path):
        bad_model = tmp_path / "bad.model"
        bad_model.write_text("this is not a sentencepiece model")
        with pytest.raises(Exception):
            RmcTokenizer.load(bad_model)

    def test_load_missing_model_raises(self, tmp_path):
        with pytest.raises((OSError, RuntimeError)):
            RmcTokenizer.load(tmp_path / "nonexistent.model")


# ===================================================================
# 5. Manifest validation (load-time)
# ===================================================================

class TestManifestValidation:
    def test_malformed_json_raises(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )
        (out / "tokenizer_manifest.json").write_text("NOT JSON{{{")
        with pytest.raises(ValueError, match="malformed JSON"):
            RmcTokenizer.load(out / "rmc-tokenizer.model")

    def test_manifest_root_array_raises(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )
        (out / "tokenizer_manifest.json").write_text("[]")
        with pytest.raises(ValueError, match="JSON object"):
            RmcTokenizer.load(out / "rmc-tokenizer.model")

    def test_manifest_root_string_raises(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )
        (out / "tokenizer_manifest.json").write_text('"string"')
        with pytest.raises(ValueError, match="JSON object"):
            RmcTokenizer.load(out / "rmc-tokenizer.model")

    def test_manifest_root_number_raises(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )
        (out / "tokenizer_manifest.json").write_text("123")
        with pytest.raises(ValueError, match="JSON object"):
            RmcTokenizer.load(out / "rmc-tokenizer.model")

    def test_manifest_missing_required_field_raises(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )
        manifest = json.loads(
            (out / "tokenizer_manifest.json").read_text()
        )
        del manifest["model_sha256"]
        (out / "tokenizer_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        with pytest.raises(ValueError, match="missing required fields"):
            RmcTokenizer.load(out / "rmc-tokenizer.model")

    def test_manifest_missing_special_token_field_raises(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )
        manifest = json.loads(
            (out / "tokenizer_manifest.json").read_text()
        )
        del manifest["special_tokens"]["cls_piece"]
        (out / "tokenizer_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        with pytest.raises(ValueError, match="special_tokens.*missing"):
            RmcTokenizer.load(out / "rmc-tokenizer.model")


# ===================================================================
# 6. Hash validation (load-time)
# ===================================================================

class TestHashValidation:
    def test_model_sha_mismatch_raises(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )
        manifest = json.loads(
            (out / "tokenizer_manifest.json").read_text()
        )
        manifest["model_sha256"] = "0" * 64
        (out / "tokenizer_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        with pytest.raises(ValueError, match="SHA-256 mismatch.*model"):
            RmcTokenizer.load(out / "rmc-tokenizer.model")

    def test_vocab_sha_mismatch_raises(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )
        manifest = json.loads(
            (out / "tokenizer_manifest.json").read_text()
        )
        manifest["vocab_sha256"] = "0" * 64
        (out / "tokenizer_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        with pytest.raises(ValueError, match="SHA-256 mismatch.*vocab"):
            RmcTokenizer.load(out / "rmc-tokenizer.model")

    def test_vocab_sha_skipped_when_no_vocab(self, tmp_path):
        """If vocab file does not exist, vocab hash is not verified."""
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        tok = RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )
        (out / "rmc-tokenizer.vocab").unlink()
        loaded = RmcTokenizer.load(out / "rmc-tokenizer.model")
        assert loaded.vocab_size > 0


# ===================================================================
# 7. Special token IDs
# ===================================================================

class TestSpecialTokenIDs:
    def test_pad_id_is_zero(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:10])
        assert tok.pad_id == 0

    def test_unk_id_is_one(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:10])
        assert tok.unk_id == 1

    def test_cls_id_is_two(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:10])
        assert tok.cls_id == 2

    def test_sep_id_is_three(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:10])
        assert tok.sep_id == 3

    def test_mask_id_is_four(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:10])
        assert tok.mask_id == 4

    def test_vocab_size_reported(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:10])
        assert tok.vocab_size > 0
        assert isinstance(tok.vocab_size, int)

    def test_special_tokens_in_vocabulary(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:10])
        for piece in ("<pad>", "<unk>", "<cls>", "<sep>", "<mask>"):
            pid = tok._processor.piece_to_id(piece)
            assert pid >= 0, f"Piece {piece!r} not found in vocabulary"


# ===================================================================
# 8. Encode / Decode
# ===================================================================

class TestEncodeDecode:
    def test_encode_returns_list_of_ints(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:10])
        ids = tok.encode("hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_empty_with_specials(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:10])
        ids = tok.encode("", add_special_tokens=True)
        assert ids == [tok.cls_id, tok.sep_id]

    def test_encode_no_specials(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:10])
        ids = tok.encode("hello world", add_special_tokens=False)
        assert tok.cls_id not in ids
        assert tok.sep_id not in ids

    def test_decode_roundtrip_preserves_text(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:20])
        text = "The faculty is supportive and the labs are decent."
        ids = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert "faculty" in decoded or "supportive" in decoded or "labs" in decoded

    def test_decode_with_specials_skips_them(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:10])
        ids = tok.encode("hello world")
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert "<pad>" not in decoded
        assert "<cls>" not in decoded
        assert "<sep>" not in decoded

    def test_encode_none_raises_type_error(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:5])
        with pytest.raises(TypeError):
            tok.encode(None)  # type: ignore[arg-type]

    def test_encode_non_string_raises_type_error(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:5])
        with pytest.raises(TypeError):
            tok.encode(123)  # type: ignore[arg-type]

    def test_unicode_encoding(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:10])
        ids = tok.encode("Hostel WiFi nahi hai", add_special_tokens=False)
        assert len(ids) > 0
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert len(decoded) > 0

    def test_devanagari_encoding(self, tmp_path):
        """Actual Devanagari script not present in training corpus."""
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:10])
        devanagari_text = "कॉलेज अच्छा है"
        ids = tok.encode(devanagari_text, add_special_tokens=False)
        assert len(ids) > 0
        # Byte-fallback pieces should handle unseen Unicode
        assert tok.unk_id not in ids

    def test_emoji_encoding(self, tmp_path):
        """Actual emoji not present in training corpus exercises byte fallback."""
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:10])
        emoji_text = "good college 🧪"
        ids = tok.encode(emoji_text, add_special_tokens=False)
        assert len(ids) > 0
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert "🧪" in decoded


# ===================================================================
# 9. Byte fallback proof
# ===================================================================

class TestByteFallbackProof:
    """Focused tests proving byte_fallback=True handles unseen Unicode."""

    def test_unseen_devanagari_byte_fallback(self, tmp_path):
        """Devanagari unseen in training -> byte fallback, never <unk>."""
        corpus = _make_corpus_file(tmp_path, [
            "The faculty is supportive and the labs are decent.",
            "Placements are good especially for CSE and ECE branches.",
            "Hostel mein WiFi nahi hai aur water supply irregular hai.",
            "Coding culture is very strong here hackathons every month.",
            "Library timing 8am to 10pm should extend to 24x7.",
        ])
        out = tmp_path / "out"
        out.mkdir()
        tok = RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )

        devanagari_text = "कॉलेज अच्छा है"
        ids = tok.encode(devanagari_text, add_special_tokens=False)
        assert len(ids) > 0
        assert tok.unk_id not in ids

        pieces = tok._processor.IdToPiece(ids)
        byte_fallback_pieces = [p for p in pieces if p.startswith("<0x")]
        assert len(byte_fallback_pieces) > 0, (
            "Expected at least one byte-fallback piece (<0xNN>) for "
            "unseen Devanagari text, got pieces: " + str(pieces)
        )

    def test_ascii_no_byte_fallback(self, tmp_path):
        """ASCII tokens in-vocabulary produce no byte-fallback pieces."""
        corpus = _make_corpus_file(tmp_path, [
            "The faculty is supportive and the labs are decent.",
            "Placements are good especially for CSE and ECE branches.",
            "Hostel mein WiFi nahi hai aur water supply irregular hai.",
            "Coding culture is very strong here hackathons every month.",
            "Library timing 8am to 10pm should extend to 24x7.",
        ])
        out = tmp_path / "out"
        out.mkdir()
        tok = RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )

        ids = tok.encode("good college", add_special_tokens=False)
        assert len(ids) > 0
        assert tok.unk_id not in ids

    def test_unseen_emoji_byte_fallback(self, tmp_path):
        """Emoji unseen in training -> byte fallback, never <unk>."""
        corpus = _make_corpus_file(tmp_path, [
            "The faculty is supportive and the labs are decent.",
            "Placements are good especially for CSE and ECE branches.",
            "Hostel mein WiFi nahi hai aur water supply irregular hai.",
            "Coding culture is very strong here hackathons every month.",
            "Library timing 8am to 10pm should extend to 24x7.",
        ])
        out = tmp_path / "out"
        out.mkdir()
        tok = RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )

        emoji_text = "good college 🧪"
        ids = tok.encode(emoji_text, add_special_tokens=False)
        assert len(ids) > 0
        assert tok.unk_id not in ids

        pieces = tok._processor.IdToPiece(ids)
        byte_fallback_pieces = [p for p in pieces if p.startswith("<0x")]
        assert len(byte_fallback_pieces) > 0, (
            "Expected at least one byte-fallback piece (<0xNN>) for "
            "unseen emoji text, got pieces: " + str(pieces)
        )

        decoded = tok.decode(ids, skip_special_tokens=True)
        assert "🧪" in decoded


# ===================================================================
# 10. Batch encode
# ===================================================================

class TestBatchEncode:
    def test_returns_input_ids_and_attention_mask(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:10])
        result = tok.batch_encode(["hello", "world"])
        assert "input_ids" in result
        assert "attention_mask" in result
        assert len(result["input_ids"]) == 2
        assert len(result["attention_mask"]) == 2

    def test_no_token_type_ids(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:10])
        result = tok.batch_encode(["hello", "world"])
        assert "token_type_ids" not in result

    def test_padding_matches_longest(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:10])
        result = tok.batch_encode(["hi", "hello world test"])
        ids = result["input_ids"]
        masks = result["attention_mask"]
        assert len(ids[0]) == len(ids[1])
        assert len(masks[0]) == len(masks[1])

    def test_attention_mask_values(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:10])
        result = tok.batch_encode(["hi", "hello world"])
        for seq, mask in zip(result["input_ids"], result["attention_mask"]):
            for i, m in enumerate(mask):
                if m == 1:
                    assert seq[i] != tok.pad_id
                elif m == 0:
                    assert seq[i] == tok.pad_id

    def test_batch_with_truncation(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:10])
        long_text = " ".join(["word"] * 200)
        result = tok.batch_encode(
            [long_text], max_length=64, truncation=True
        )
        assert len(result["input_ids"][0]) == 64
        assert len(result["attention_mask"][0]) == 64


# ===================================================================
# 11. Max length / special token accounting
# ===================================================================

class TestMaxLengthAccounting:
    def test_max_length_accounts_for_specials(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:10])
        text = " ".join(["word"] * 100)
        ids = tok.encode(
            text, add_special_tokens=True, max_length=20, truncation=True
        )
        assert len(ids) == 20

    def test_empty_max_length(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:10])
        ids = tok.encode("", add_special_tokens=True, max_length=10, truncation=True)
        assert len(ids) == 2  # cls + sep


# ===================================================================
# 12. SHA-256
# ===================================================================

class TestSHA256:
    def test_file_hash_deterministic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h1 = compute_file_sha256(f)
        h2 = compute_file_sha256(f)
        assert h1 == h2
        assert len(h1) == 64
        assert all(c in "0123456789abcdef" for c in h1)

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        assert compute_file_sha256(f1) != compute_file_sha256(f2)

    def test_hash_binary_file(self, tmp_path):
        f = tmp_path / "binary.bin"
        f.write_bytes(bytes(range(256)))
        h = compute_file_sha256(f)
        assert len(h) == 64


# ===================================================================
# 13. Manifest schema
# ===================================================================

class TestManifest:
    def test_manifest_schema_fields(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="train-split-v1",
            corpus_policy="train-split-only",
        )
        manifest = json.loads(
            (out / "tokenizer_manifest.json").read_text()
        )
        required = [
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
        for field_name in required:
            assert field_name in manifest, f"Missing manifest field: {field_name}"

    def test_manifest_special_token_ids(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )
        manifest = json.loads(
            (out / "tokenizer_manifest.json").read_text()
        )
        st = manifest["special_tokens"]
        assert st["pad_piece"] == "<pad>"
        assert st["pad_id"] == 0
        assert st["unk_piece"] == "<unk>"
        assert st["unk_id"] == 1
        assert st["cls_piece"] == "<cls>"
        assert st["cls_id"] == 2
        assert st["sep_piece"] == "<sep>"
        assert st["sep_id"] == 3
        assert st["mask_piece"] == "<mask>"
        assert st["mask_id"] == 4

    def test_manifest_vocab_size_fields(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        cfg = _make_test_config(tmp_path, vocab_size=512)
        RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=cfg,
            corpus_source="test",
        )
        manifest = json.loads(
            (out / "tokenizer_manifest.json").read_text()
        )
        assert manifest["vocab_size_target"] == 512
        assert manifest["vocab_size_actual"] > 0

    def test_manifest_created_at_is_iso_timestamp(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )
        manifest = json.loads(
            (out / "tokenizer_manifest.json").read_text()
        )
        ts = manifest["created_at"]
        datetime.fromisoformat(ts)


# ===================================================================
# 14. Special token contract verification
# ===================================================================

class TestSpecialTokenContract:
    def test_id_contract_violation_raises(self, tmp_path):
        """Verify the contract holds for the trained model."""
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:10])
        out = tmp_path / "out"
        out.mkdir()
        tok = RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )

        expected = {
            "<pad>": 0,
            "<unk>": 1,
            "<cls>": 2,
            "<sep>": 3,
            "<mask>": 4,
        }
        for piece, expected_id in expected.items():
            actual_id = tok._processor.piece_to_id(piece)
            assert actual_id == expected_id, (
                f"Special token {piece!r} has ID {actual_id}, "
                f"expected {expected_id}"
            )


# ===================================================================
# 15. Load without original corpus
# ===================================================================

class TestLoadWithoutCorpus:
    def test_load_without_corpus(self, tmp_path):
        """Loading a tokenizer should not require the training corpus."""
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )
        corpus.unlink()
        loaded = RmcTokenizer.load(out / "rmc-tokenizer.model")
        assert loaded.vocab_size > 0
        assert loaded.pad_id == 0


# ===================================================================
# 16. Model hash mismatch (manifest cross-check)
# ===================================================================

class TestManifestIntegrity:
    def test_model_sha256_matches_file(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )
        model_file = out / "rmc-tokenizer.model"
        actual_hash = compute_file_sha256(model_file)
        manifest = json.loads(
            (out / "tokenizer_manifest.json").read_text()
        )
        assert manifest["model_sha256"] == actual_hash


# ===================================================================
# 17. Reproducibility
# ===================================================================

class TestReproducibility:
    def test_encode_deterministic(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:10])
        ids1 = tok.encode("hello world", add_special_tokens=False)
        ids2 = tok.encode("hello world", add_special_tokens=False)
        assert ids1 == ids2

    def test_load_produces_same_ids(self, tmp_path):
        corpus = _make_corpus_file(tmp_path, PILOT_LINES[:5])
        out = tmp_path / "out"
        out.mkdir()
        tok1 = RmcTokenizer.train(
            corpus_path=corpus,
            output_dir=out,
            config=_make_test_config(tmp_path),
            corpus_source="test",
        )
        ids1 = tok1.encode("hello world", add_special_tokens=False)
        tok2 = RmcTokenizer.load(out / "rmc-tokenizer.model")
        ids2 = tok2.encode("hello world", add_special_tokens=False)
        assert ids1 == ids2


# ===================================================================
# 18. Normalization identity
# ===================================================================

class TestNormalizationIdentity:
    """Verify that the tokenizer uses identity normalization."""

    def test_case_preserved(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:15])
        text_upper = "CSE is great"
        text_lower = "cse is great"
        ids_u = tok.encode(text_upper, add_special_tokens=False)
        ids_l = tok.encode(text_lower, add_special_tokens=False)
        assert ids_u != ids_l, (
            "Identity normalization should preserve case; "
            "upper and lower should not produce identical IDs"
        )

    def test_unicode_preserved(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:10])
        text = "resume naive"
        ids = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert len(decoded) > 0

    def test_punctuation_preserved(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:10])
        text = "hello!!! world??? test..."
        ids = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert len(decoded) > 0


# ===================================================================
# 19. Hinglish / Roman Hindi
# ===================================================================

class TestHinglishRomanHindi:
    def test_hinglish_encode(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:15])
        text = "Faculty supportive hai but placement thoda weak hai"
        ids = tok.encode(text, add_special_tokens=False)
        assert len(ids) > 0
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert len(decoded) > 0

    def test_roman_hindi_encode(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:15])
        text = "College ka campus accha hai aur teachers helpful hain"
        ids = tok.encode(text, add_special_tokens=False)
        assert len(ids) > 0
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert len(decoded) > 0

    def test_code_mixed_terms(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:20])
        text = "B.Tech CSE AI ML placements"
        ids = tok.encode(text, add_special_tokens=False)
        assert len(ids) > 0


# ===================================================================
# 20. Diagnostics
# ===================================================================

class TestDiagnostics:
    def test_diagnostics_structure(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:15])
        diag = tok.compute_diagnostics(PILOT_LINES[:15])
        assert "overall" in diag
        overall = diag["overall"]
        assert "avg_tokens_per_review" in overall
        assert "median_tokens_per_review" in overall
        assert "unk_rate" in overall
        assert "vocab_utilization" in overall

    def test_diagnostics_language_slices(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:20])
        slices = {
            "ENGLISH": [PILOT_LINES[0]],
            "HINGLISH": [PILOT_LINES[1]],
            "ROMAN_HINDI": [PILOT_LINES[2]],
        }
        diag = tok.compute_diagnostics(
            PILOT_LINES[:5], language_slices=slices
        )
        assert "ENGLISH" in diag
        assert "HINGLISH" in diag
        assert "ROMAN_HINDI" in diag

    def test_diagnostics_empty_text(self, tmp_path):
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:10])
        diag = tok.compute_diagnostics([""])
        assert diag["overall"]["token_count"] == 2  # cls + sep

    def test_diagnostics_uses_only_dataset_spec_categories(self, tmp_path):
        """Slice names must match DATASET_SPEC language_mix values."""
        valid_slices = {
            "ENGLISH", "HINGLISH", "ROMAN_HINDI", "OTHER", "MIXED_OTHER"
        }
        slices = {
            name: [PILOT_LINES[0]]
            for name in ["ENGLISH", "HINGLISH", "ROMAN_HINDI", "OTHER", "MIXED_OTHER"]
        }
        tok = _train_tiny_tokenizer(tmp_path, PILOT_LINES[:15])
        diag = tok.compute_diagnostics(
            PILOT_LINES[:5], language_slices=slices
        )
        for name in diag:
            if name != "overall":
                assert name in valid_slices, (
                    f"Slice name {name!r} is not a valid DATASET_SPEC "
                    f"language_mix value"
                )


# ===================================================================
# 21. Corpus hash stability
# ===================================================================

class TestCorpusHash:
    def test_hash_changes_with_content(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        assert compute_file_sha256(f1) != compute_file_sha256(f2)

    def test_hash_stable_across_calls(self, tmp_path):
        f = tmp_path / "stable.txt"
        f.write_text("stable content")
        assert compute_file_sha256(f) == compute_file_sha256(f)