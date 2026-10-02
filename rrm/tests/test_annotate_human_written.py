"""Tests for rrm.annotate_human_written CLI module.

Public API tested:
    main (argparse entry point)

All helpers are local — no cross-test module imports.
"""

from __future__ import annotations

import pytest

from rrm.corpus.annotation_workspace import HumanWrittenAnnotationWorkspace
from rrm.corpus.models import LanguageMix
from rrm.corpus.workflow import CorpusWorkflow


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def _make_workflow():
    from rrm.corpus.annotation import (
        AnnotationSubmissionStore,
        ControlledProtocolTruthStore,
        DispositionRegister,
        ReannotationRegister,
    )
    return CorpusWorkflow(
        AnnotationSubmissionStore(),
        ControlledProtocolTruthStore(),
        DispositionRegister(),
        ReannotationRegister(),
    )


def _collect(text="Good college with nice campus") -> object:
    from rrm.corpus.intake import collect_human_written_review
    return collect_human_written_review(
        workflow=_make_workflow(),
        submitted_text=text,
        consent_granted=True,
        collected_at="2025-01-01T00:00:00Z",
        collection_method="web_form",
        retain_source_text_raw=False,
    )


def _seed_collection(root_dir, texts):
    """Create a collection store and seed it with review texts."""
    from rrm.corpus.annotation import (
        AnnotationSubmissionStore,
        ControlledProtocolTruthStore,
        DispositionRegister,
        ReannotationRegister,
    )
    from rrm.corpus.collection_store import (
        CollectionSession,
        HumanWrittenCollectionStore,
    )
    store = HumanWrittenCollectionStore(root_dir)
    session = store.start_session(
        started_at="2025-01-01T00:00:00Z",
        collection_method="web_form",
    )
    wf = CorpusWorkflow(
        AnnotationSubmissionStore(),
        ControlledProtocolTruthStore(),
        DispositionRegister(),
        ReannotationRegister(),
    )
    ids = []
    for text in texts:
        result = _collect(text=text)
        store.append_intake(session, result)
        ids.append(result.record.review_id)
    return ids


def _choices(language_mix=None, college_category=None):
    """Build a valid annotation choices dict."""
    if language_mix is None:
        language_mix = LanguageMix.ENGLISH
    c = {
        "spam": 0,
        "toxicity": 0,
        "advertising": 0,
        "off_topic": 0,
        "pii": 0,
        "language_mix": language_mix,
    }
    if college_category is not None:
        c["college_category"] = college_category
    return c


# ---------------------------------------------------------------------------
# A — CLI arguments (Requirements 10, 11, 12)
# ---------------------------------------------------------------------------

class TestCLIArguments:
    def test_no_args_prints_help(self, capsys):
        from rrm.annotate_human_written import main
        with pytest.raises(SystemExit):
            main([])
        captured = capsys.readouterr()
        # argparse writes usage to stderr on missing required args
        assert "usage" in captured.err.lower() or "required" in captured.err.lower()

    def test_root_dir_default(self):
        from rrm.annotate_human_written import _DEFAULT_ROOT_DIR
        assert _DEFAULT_ROOT_DIR == "local_research_data/gate_c/human_written"

    def test_slot_required(self, tmp_path, monkeypatch):
        from rrm.annotate_human_written import main
        _seed_collection(str(tmp_path / "data"), ["text"])
        monkeypatch.chdir(str(tmp_path))
        with pytest.raises(SystemExit):
            main(["--root-dir", str(tmp_path / "data"),
                  "--annotator-id", "ann-a"])

    def test_annotator_id_required(self, tmp_path, monkeypatch):
        from rrm.annotate_human_written import main
        _seed_collection(str(tmp_path / "data"), ["text"])
        with pytest.raises(SystemExit):
            main(["--root-dir", str(tmp_path / "data"),
                  "--slot", "A"])

    def test_blank_annotator_id_rejected(self, tmp_path):
        from rrm.annotate_human_written import main
        _seed_collection(str(tmp_path / "data"), ["text"])
        ret = main(["--root-dir", str(tmp_path / "data"),
                    "--slot", "A", "--annotator-id", "   "])
        assert ret == 1


# ---------------------------------------------------------------------------
# B — workspace loading via CLI
# ---------------------------------------------------------------------------

class TestWorkspaceLoading:
    def test_no_records_prints_none(self, tmp_path, capsys):
        from rrm.annotate_human_written import main
        ret = main(["--root-dir", str(tmp_path / "empty"),
                    "--slot", "A", "--annotator-id", "ann-a"])
        assert ret == 0

    def test_with_records_selects_next(self, tmp_path, monkeypatch, capsys):
        from rrm.annotate_human_written import main
        _seed_collection(str(tmp_path / "data"), ["Review text"])
        # Normal input sequence: 5 binary + language + college
        responses = iter(["0", "0", "0", "0", "0", "ENGLISH", "NONE"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        ret = main(["--root-dir", str(tmp_path / "data"),
                    "--slot", "A", "--annotator-id", "ann-a"])
        captured = capsys.readouterr()
        assert ret == 0
        assert "Review ID" in captured.out


# ---------------------------------------------------------------------------
# C — slot selection via CLI
# ---------------------------------------------------------------------------

class TestSlotSelection:
    def test_slot_a_accepted(self, tmp_path, monkeypatch, capsys):
        from rrm.annotate_human_written import main
        _seed_collection(str(tmp_path / "data"), ["Review text"])
        responses = iter(["0", "0", "0", "0", "0", "ENGLISH", "NONE"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        ret = main(["--root-dir", str(tmp_path / "data"),
                    "--slot", "A", "--annotator-id", "ann-a"])
        captured = capsys.readouterr()
        assert ret == 0
        assert "Annotation saved" in captured.out

    def test_slot_b_after_a(self, tmp_path, monkeypatch, capsys):
        from rrm.annotate_human_written import main
        _seed_collection(str(tmp_path / "data"), ["Review text"])
        # A first
        responses_a = iter(["0", "0", "0", "0", "0", "ENGLISH", "NONE"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses_a))
        ret = main(["--root-dir", str(tmp_path / "data"),
                    "--slot", "A", "--annotator-id", "ann-a"])
        assert ret == 0
        # B second
        responses_b = iter(["0", "0", "0", "0", "0", "ENGLISH", "NONE"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses_b))
        ret = main(["--root-dir", str(tmp_path / "data"),
                    "--slot", "B", "--annotator-id", "ann-b"])
        captured = capsys.readouterr()
        assert ret == 0
        assert "Annotation saved" in captured.out


# ---------------------------------------------------------------------------
# D — blindness (review_id and review_text only)
# ---------------------------------------------------------------------------

class TestBlindness:
    def test_pseudonym_not_shown(self, tmp_path, monkeypatch, capsys):
        from rrm.annotate_human_written import main
        _seed_collection(str(tmp_path / "data"), ["Review text"])
        responses = iter(["0", "0", "0", "0", "0", "ENGLISH", "NONE"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        main(["--root-dir", str(tmp_path / "data"),
              "--slot", "A", "--annotator-id", "ann-a"])
        captured = capsys.readouterr()
        assert "contrib-" not in captured.out

    def test_no_pii_warning(self, tmp_path, monkeypatch, capsys):
        from rrm.annotate_human_written import main
        _seed_collection(str(tmp_path / "data"), ["Review text"])
        responses = iter(["0", "0", "0", "0", "0", "ENGLISH", "NONE"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        main(["--root-dir", str(tmp_path / "data"),
              "--slot", "A", "--annotator-id", "ann-a"])
        captured = capsys.readouterr()
        assert "private personal" not in captured.out.lower()

    def test_no_other_annotator_info(self, tmp_path, monkeypatch, capsys):
        from rrm.annotate_human_written import main
        ids = _seed_collection(str(tmp_path / "data"), ["Review text"])
        # Submit A with a distinctive annotator ID, then run B
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        ws.submit_annotation(
            slot="A",
            review_id=ids[0],
            annotator_id="secret-ann-a",
            choices=_choices(),
        )
        responses = iter(["0", "0", "0", "0", "0", "ENGLISH", "NONE"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        main(["--root-dir", str(tmp_path / "data"),
              "--slot", "B", "--annotator-id", "ann-b"])
        captured = capsys.readouterr()
        assert "secret-ann-a" not in captured.out
        assert "annotator_A_id" not in captured.out


# ---------------------------------------------------------------------------
# E — deception policy
# ---------------------------------------------------------------------------

class TestDeceptionPolicy:
    def test_deception_unknown_displayed(self, tmp_path, monkeypatch, capsys):
        from rrm.annotate_human_written import main
        _seed_collection(str(tmp_path / "data"), ["Review text"])
        responses = iter(["0", "0", "0", "0", "0", "ENGLISH", "NONE"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        main(["--root-dir", str(tmp_path / "data"),
              "--slot", "A", "--annotator-id", "ann-a"])
        captured = capsys.readouterr()
        assert "UNKNOWN" in captured.out
        assert "-1" in captured.out

    def test_no_deception_prompt(self, tmp_path, monkeypatch, capsys):
        from rrm.annotate_human_written import main
        _seed_collection(str(tmp_path / "data"), ["Review text"])
        # Provide only 5 binary + language + college (no deception prompt)
        responses = iter(["0", "0", "0", "0", "0", "ENGLISH", "NONE"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        main(["--root-dir", str(tmp_path / "data"),
              "--slot", "A", "--annotator-id", "ann-a"])
        captured = capsys.readouterr()
        # "deception" should not appear in any interactive prompt label
        assert "Deception" not in captured.out


# ---------------------------------------------------------------------------
# F — privacy instruction
# ---------------------------------------------------------------------------

class TestPrivacyInstruction:
    def test_privacy_notice_displayed(self, tmp_path, monkeypatch, capsys):
        from rrm.annotate_human_written import main
        _seed_collection(str(tmp_path / "data"), ["Review text"])
        responses = iter(["0", "0", "0", "0", "0", "ENGLISH", "NONE"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        main(["--root-dir", str(tmp_path / "data"),
              "--slot", "A", "--annotator-id", "ann-a"])
        captured = capsys.readouterr()
        assert "pseudonymous" in captured.out
        assert "real name" in captured.out


# ---------------------------------------------------------------------------
# G — one review per invocation
# ---------------------------------------------------------------------------

class TestOneReviewPerInvocation:
    def test_one_annotation_per_invocation(self, tmp_path, monkeypatch, capsys):
        from rrm.annotate_human_written import main
        _seed_collection(str(tmp_path / "data"), ["a", "b", "c"])
        responses = iter(["0", "0", "0", "0", "0", "ENGLISH", "NONE"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        main(["--root-dir", str(tmp_path / "data"),
              "--slot", "A", "--annotator-id", "ann-a"])
        captured = capsys.readouterr()
        assert "Annotation saved" in captured.out
        # No "Annotate another" prompt
        assert "another" not in captured.out.lower()


# ---------------------------------------------------------------------------
# H — exact review selection via --review-id
# ---------------------------------------------------------------------------

class TestExactReviewSelection:
    def test_exact_review_id_accepted(self, tmp_path, monkeypatch, capsys):
        from rrm.annotate_human_written import main
        ids = _seed_collection(str(tmp_path / "data"), ["target review"])
        responses = iter(["0", "0", "0", "0", "0", "ENGLISH", "NONE"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        ret = main(["--root-dir", str(tmp_path / "data"),
                    "--slot", "A", "--annotator-id", "ann-a",
                    "--review-id", ids[0]])
        captured = capsys.readouterr()
        assert ret == 0
        assert ids[0] in captured.out

    def test_unknown_review_id_returns_error(self, tmp_path):
        from rrm.annotate_human_written import main
        _seed_collection(str(tmp_path / "data"), ["text"])
        ret = main(["--root-dir", str(tmp_path / "data"),
                    "--slot", "A", "--annotator-id", "ann-a",
                    "--review-id", "nonexistent"])
        assert ret == 1

    def test_no_eligible_returns_zero(self, tmp_path, capsys):
        from rrm.annotate_human_written import main
        _seed_collection(str(tmp_path / "data"), ["text"])
        ret = main(["--root-dir", str(tmp_path / "data"),
                    "--slot", "B", "--annotator-id", "ann-b"])
        captured = capsys.readouterr()
        assert ret == 0


# ---------------------------------------------------------------------------
# I — Hinglish review via CLI
# ---------------------------------------------------------------------------

class TestHinglishCLI:
    def test_hinglish_review_annotated(self, tmp_path, monkeypatch, capsys):
        from rrm.annotate_human_written import main
        _seed_collection(
            str(tmp_path / "data"),
            ["यह एक हिंदी review है, placement is good"],
        )
        responses = iter(["0", "0", "0", "0", "0", "HINGLISH", "NONE"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        ret = main(["--root-dir", str(tmp_path / "data"),
                    "--slot", "A", "--annotator-id", "ann-a"])
        captured = capsys.readouterr()
        assert ret == 0
        assert "Annotation saved" in captured.out
