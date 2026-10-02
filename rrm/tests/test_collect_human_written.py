"""Tests for rrm.collect_human_written CLI module.

Public API tested:
    run_collection_cli
    main (argparse entry point)
"""

from __future__ import annotations

import json
import os

import pytest


# ---------------------------------------------------------------------------
# A — consent handling (Blockers G, H)
# ---------------------------------------------------------------------------


class TestConsentHandling:
    def test_decline_prints_cancel_message(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", lambda _: "NO")
        from rrm.collect_human_written import run_collection_cli
        run_collection_cli(root_dir=str(tmp_path / "decline"))
        captured = capsys.readouterr()
        assert "declined" in captured.out.lower() or "goodbye" in captured.out.lower()

    def test_decline_creates_no_files(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", lambda _: "NO")
        from rrm.collect_human_written import run_collection_cli
        root = str(tmp_path / "decline2")
        run_collection_cli(root_dir=root)
        captured = capsys.readouterr()
        # FIX 7 — assert all four files absent on decline
        assert not os.path.exists(os.path.join(root, "records.jsonl"))
        assert not os.path.exists(os.path.join(root, "consents.jsonl"))
        assert not os.path.exists(os.path.join(root, "pii_evidence.jsonl"))
        assert not os.path.exists(os.path.join(root, "session_manifest.json"))
        assert "declined" in captured.out.lower() or "goodbye" in captured.out.lower()

    def test_non_yes_decline_exercises_cli(self, tmp_path, monkeypatch, capsys):
        prompts_received = []
        monkeypatch.setattr(
            "builtins.input",
            lambda prompt: (prompts_received.append(prompt) or "maybe"),
        )
        from rrm.collect_human_written import run_collection_cli
        root = str(tmp_path / "maybe_decline")
        run_collection_cli(root_dir=root)
        captured = capsys.readouterr()
        # Only the consent prompt should have been shown
        assert len(prompts_received) == 1
        # No collection files should exist
        assert not os.path.exists(os.path.join(root, "records.jsonl"))
        # Cancellation path
        assert "declined" in captured.out.lower() or "goodbye" in captured.out.lower()

    def test_yes_lowercase_accepted(self, tmp_path, monkeypatch):
        # FIX 5 — two real responses via iterator
        responses = iter(["yes", "test review"])
        monkeypatch.setattr(
            "builtins.input",
            lambda prompt: next(responses),
        )
        from rrm.collect_human_written import run_collection_cli
        root = str(tmp_path / "yes_lower")
        run_collection_cli(root_dir=root)
        # records.jsonl should exist (collection succeeded)
        assert os.path.exists(os.path.join(root, "records.jsonl"))

    def test_whitespace_yes_accepted(self, tmp_path, monkeypatch):
        # FIX 6 — two real responses via iterator
        responses = iter(["  YES  ", "test review"])
        monkeypatch.setattr(
            "builtins.input",
            lambda prompt: next(responses),
        )
        from rrm.collect_human_written import run_collection_cli
        root = str(tmp_path / "yes_ws")
        run_collection_cli(root_dir=root)
        assert os.path.exists(os.path.join(root, "records.jsonl"))

    def test_no_review_prompt_after_decline(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", lambda _: "NO")
        from rrm.collect_human_written import run_collection_cli
        root = str(tmp_path / "no_review")
        run_collection_cli(root_dir=root)
        captured = capsys.readouterr()
        assert "enter your review" not in captured.out.lower()


# ---------------------------------------------------------------------------
# B — PII warning display + event order (Blocker K, FIX 4)
# ---------------------------------------------------------------------------


class TestPIIWarning:
    def test_pii_warning_text_exists(self):
        from rrm.collect_human_written import _PII_WARNING
        assert "private personal information" in _PII_WARNING

    def test_consent_statement_text_exists(self):
        from rrm.collect_human_written import _CONSENT_STATEMENT
        assert "research" in _CONSENT_STATEMENT.lower()
        assert "development" in _CONSENT_STATEMENT.lower()

    def test_pii_warning_event_order_accepted_consent(
        self, tmp_path, monkeypatch
    ):
        # Import the exact constants for unambiguous matching
        from rrm.collect_human_written import (
            _CONSENT_STATEMENT,
            _PII_WARNING,
        )

        # Single chronological events list
        events = []
        original_print = print

        def tracking_print(*args, **kwargs):
            text = " ".join(str(a) for a in args)
            events.append(("print", text))
            original_print(*args, **kwargs)

        responses = iter(["YES", "test review"])
        monkeypatch.setattr(
            "builtins.input",
            lambda prompt: (events.append(("input", prompt)) or next(responses)),
        )
        monkeypatch.setattr("builtins.print", tracking_print)

        from rrm.collect_human_written import run_collection_cli
        run_collection_cli(root_dir=str(tmp_path / "pii_order"))

        # Find GLOBAL indexes in the single chronological events list
        # using exact-strip matching against the known constants
        consent_print_idx = next(
            (i for i, (t, txt) in enumerate(events)
             if t == "print" and txt.strip() == _CONSENT_STATEMENT.strip()),
            None,
        )
        consent_input_idx = next(
            (i for i, (t, _) in enumerate(events)
             if t == "input"),
            None,
        )
        pii_warning_idx = next(
            (i for i, (t, txt) in enumerate(events)
             if t == "print" and txt.strip() == _PII_WARNING.strip()),
            None,
        )
        review_input_idx = next(
            (i for i, (t, _) in enumerate(events)
             if t == "input" and i > (consent_input_idx or -1)),
            None,
        )

        assert consent_print_idx is not None, "Consent statement not printed"
        assert consent_input_idx is not None, "Consent input not requested"
        assert pii_warning_idx is not None, "PII warning not printed"
        assert review_input_idx is not None, "Review input not requested"
        # Assert the required ordering
        assert consent_print_idx < consent_input_idx, (
            f"Consent print ({consent_print_idx}) must be before "
            f"consent input ({consent_input_idx})"
        )
        assert consent_input_idx < pii_warning_idx, (
            f"Consent input ({consent_input_idx}) must be before "
            f"PII warning ({pii_warning_idx})"
        )
        assert pii_warning_idx < review_input_idx, (
            f"PII warning ({pii_warning_idx}) must be before "
            f"review input ({review_input_idx})"
        )


# ---------------------------------------------------------------------------
# C — successful collection flow
# ---------------------------------------------------------------------------


class TestSuccessfulCollection:
    def test_full_flow_creates_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "builtins.input",
            lambda prompt: "YES" if "consent" in prompt.lower() or "submit" in prompt.lower() or "review" not in prompt.lower() else "Great campus!",
        )
        from rrm.collect_human_written import run_collection_cli
        root = str(tmp_path / "flow")
        run_collection_cli(root_dir=root)
        assert os.path.exists(os.path.join(root, "records.jsonl"))
        assert os.path.exists(os.path.join(root, "consents.jsonl"))
        assert os.path.exists(os.path.join(root, "session_manifest.json"))

    def test_success_prints_exact_review_id(self, tmp_path, monkeypatch, capsys):
        call_count = [0]
        def mock_input(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                return "YES"
            return "Great campus!"
        monkeypatch.setattr("builtins.input", mock_input)
        from rrm.collect_human_written import run_collection_cli
        root = str(tmp_path / "review_id")
        run_collection_cli(root_dir=root)
        captured = capsys.readouterr()
        # Read the exact review_id from persisted record
        with open(os.path.join(root, "records.jsonl"), "r", encoding="utf-8") as fh:
            record = json.loads(fh.readline())
        exact_id = record["review_id"]
        assert exact_id in captured.out

    def test_contributor_pseudonym_not_printed(self, tmp_path, monkeypatch, capsys):
        call_count = [0]
        def mock_input(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                return "YES"
            return "Great campus!"
        monkeypatch.setattr("builtins.input", mock_input)
        from rrm.collect_human_written import run_collection_cli
        root = str(tmp_path / "no_text_print")
        run_collection_cli(root_dir=root)
        captured = capsys.readouterr()
        with open(os.path.join(root, "records.jsonl"), "r", encoding="utf-8") as fh:
            record = json.loads(fh.readline())
        # Pseudonym not in stdout
        assert record["contributor_pseudonym"] not in captured.out
        # Review text not in stdout
        assert "Great campus!" not in captured.out

    def test_empty_review_creates_no_files(self, tmp_path, monkeypatch, capsys):
        call_count = [0]
        def mock_input(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                return "YES"
            return ""
        monkeypatch.setattr("builtins.input", mock_input)
        from rrm.collect_human_written import run_collection_cli
        root = str(tmp_path / "empty")
        run_collection_cli(root_dir=root)
        captured = capsys.readouterr()
        assert not os.path.exists(os.path.join(root, "records.jsonl"))
        assert "no data" in captured.out.lower()

    def test_consent_statement_printed(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", lambda _: "NO")
        from rrm.collect_human_written import run_collection_cli
        root = str(tmp_path / "consent_print")
        run_collection_cli(root_dir=root)
        captured = capsys.readouterr()
        assert "research" in captured.out.lower()
        assert "development" in captured.out.lower()


# ---------------------------------------------------------------------------
# D — CLI persisted-record field assertions (Blocker J)
# ---------------------------------------------------------------------------


class TestCLIPersistedRecord:
    def test_persisted_fields(self, tmp_path, monkeypatch):
        call_count = [0]
        def mock_input(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                return "YES"
            return "Good college with nice campus"
        monkeypatch.setattr("builtins.input", mock_input)
        from rrm.collect_human_written import run_collection_cli
        root = str(tmp_path / "fields")
        run_collection_cli(root_dir=root)
        with open(os.path.join(root, "records.jsonl"), "r", encoding="utf-8") as fh:
            record = json.loads(fh.readline())
        assert record["source_type"] == "HUMAN_WRITTEN_RMC"
        assert record["annotation_status"] == "UNANNOTATED"
        assert record["collection_method"] == "local_cli"
        assert record["spam"] is None
        assert record["deception"] is None
        assert record["toxicity"] is None
        assert record["advertising"] is None
        assert record["off_topic"] is None
        assert record["pii"] is None
        assert record["language_mix"] is None
        assert record["college_category"] is None
        assert record["dataset_version"] is None
        assert record["split_membership"] is None
        assert record["split_group_id"] is None
        assert "source_text_raw" not in record


# ---------------------------------------------------------------------------
# E — default root directory (Blocker I / 18)
# ---------------------------------------------------------------------------


class TestDefaultRootDir:
    def test_default_root_dir_value(self):
        from rrm.collect_human_written import _get_default_root_dir
        assert _get_default_root_dir() == os.path.join(
            "local_research_data", "gate_c", "human_written"
        )

    def test_default_root_used_when_none(self, tmp_path, monkeypatch):
        from rrm import collect_human_written as cli_module
        call_count = [0]
        def mock_input(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                return "YES"
            return "test review"
        monkeypatch.setattr("builtins.input", mock_input)
        monkeypatch.setattr(
            cli_module,
            "_get_default_root_dir",
            lambda: str(tmp_path / "default_root"),
        )
        run_collection_cli = cli_module.run_collection_cli
        run_collection_cli()  # No root_dir passed
        assert os.path.exists(
            os.path.join(str(tmp_path / "default_root"), "records.jsonl")
        )


# ---------------------------------------------------------------------------
# F — argparse / main entry point (FIX 3)
# ---------------------------------------------------------------------------


class TestArgparseEntry:
    def test_main_with_output_dir(self, tmp_path, monkeypatch):
        # FIX 3 — canonical --output-dir flag
        from rrm.collect_human_written import main
        responses = iter(["YES", "test review"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        ret = main([
            "--output-dir",
            str(tmp_path / "main_test"),
        ])
        assert ret == 0
        assert os.path.exists(
            os.path.join(str(tmp_path / "main_test"), "records.jsonl")
        )

    def test_main_default_dir(self, tmp_path, monkeypatch):
        from rrm.collect_human_written import main
        from rrm import collect_human_written as cli_module
        responses = iter(["YES", "test review"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        monkeypatch.setattr(
            cli_module,
            "_get_default_root_dir",
            lambda: str(tmp_path / "main_default"),
        )
        ret = main([])
        assert ret == 0
        assert os.path.exists(
            os.path.join(str(tmp_path / "main_default"), "records.jsonl")
        )
