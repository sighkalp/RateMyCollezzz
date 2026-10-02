"""Tests for rrm.corpus.annotation_workspace module.

Public API tested:
    HumanWrittenAnnotationWorkspace
    AnnotationTask

All helpers are local — no cross-test module imports.
"""

from __future__ import annotations

import json
import os

import pytest

from rrm.corpus.annotation_workspace import (
    AnnotationTask,
    HumanWrittenAnnotationWorkspace,
)
from rrm.corpus.collection_store import (
    CollectionSession,
    HumanWrittenCollectionStore,
)
from rrm.corpus.intake import (
    HumanWrittenIntakeResult,
    collect_human_written_review,
)
from rrm.corpus.models import (
    AnnotationStatus,
    CollegeCategory,
    LanguageMix,
    SourceType,
)
from rrm.corpus.workflow import CorpusWorkflow


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def _make_workflow() -> CorpusWorkflow:
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


def _collect(
    text: str = "Good college with nice campus",
) -> HumanWrittenIntakeResult:
    return collect_human_written_review(
        workflow=_make_workflow(),
        submitted_text=text,
        consent_granted=True,
        collected_at="2025-01-01T00:00:00Z",
        collection_method="web_form",
        retain_source_text_raw=False,
    )


def _seed_collection(root_dir: str, texts: list[str]) -> list[str]:
    """Create a collection store and seed it with review texts.

    Returns the list of review_ids in insertion order.
    """
    from rrm.corpus.annotation import (
        AnnotationSubmissionStore,
        ControlledProtocolTruthStore,
        DispositionRegister,
        ReannotationRegister,
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


def _build_choices(spam=0, toxicity=0, advertising=0, off_topic=0, pii=0,
                   language_mix=None, college_category=None) -> dict:
    """Build a valid annotation choices dict."""
    if language_mix is None:
        language_mix = LanguageMix.ENGLISH
    choices = {
        "spam": spam,
        "toxicity": toxicity,
        "advertising": advertising,
        "off_topic": off_topic,
        "pii": pii,
        "language_mix": language_mix,
    }
    if college_category is not None:
        choices["college_category"] = college_category
    return choices


# ---------------------------------------------------------------------------
# A — workspace construction (Requirement 1)
# ---------------------------------------------------------------------------

class TestWorkspaceConstruction:
    def test_construct_from_root_dir(self, tmp_path):
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "ws"))
        assert ws.root_dir == str(tmp_path / "ws")

    def test_non_string_root_raises(self):
        with pytest.raises(TypeError, match="root_dir"):
            HumanWrittenAnnotationWorkspace(123)

    def test_empty_string_root_raises(self):
        with pytest.raises(TypeError, match="root_dir"):
            HumanWrittenAnnotationWorkspace("")

    def test_no_records_file_is_fine(self, tmp_path):
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "empty"))
        assert ws.get_eligible_count("A") == 0
        assert ws.get_eligible_count("B") == 0

    def test_loads_records_from_file(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), [
            "First review text",
            "Second review text",
        ])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        assert ws.get_record(ids[0]) is not None
        assert ws.get_record(ids[1]) is not None
        assert ws.get_record("nonexistent") is None


# ---------------------------------------------------------------------------
# B — source record validation (Requirement 3)
# ---------------------------------------------------------------------------

class TestSourceRecordValidation:
    def test_non_human_written_rejected(self, tmp_path):
        root = str(tmp_path / "bad")
        os.makedirs(root, exist_ok=True)
        record_data = {
            "review_id": "bad-1",
            "source_type": "CONTROLLED_RMC",
            "review_text": "text",
            "consent_status": "CONSENTED",
            "annotation_status": "UNANNOTATED",
            "spam": None,
            "deception": None,
            "toxicity": None,
            "advertising": None,
            "off_topic": None,
            "pii": None,
            "language_mix": None,
            "college_category": None,
            "dataset_version": None,
            "split_membership": None,
            "split_group_id": None,
        }
        with open(os.path.join(root, "records.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(record_data, ensure_ascii=False) + "\n")
        with pytest.raises(ValueError, match="source_type"):
            HumanWrittenAnnotationWorkspace(root)

    def test_missing_consent_rejected(self, tmp_path):
        root = str(tmp_path / "bad2")
        os.makedirs(root, exist_ok=True)
        record_data = {
            "review_id": "bad-2",
            "source_type": "HUMAN_WRITTEN_RMC",
            "review_text": "text",
            "consent_status": "NOT_CONSENTED",
            "annotation_status": "UNANNOTATED",
            "spam": None,
            "deception": None,
            "toxicity": None,
            "advertising": None,
            "off_topic": None,
            "pii": None,
            "language_mix": None,
            "college_category": None,
            "dataset_version": None,
            "split_membership": None,
            "split_group_id": None,
        }
        with open(os.path.join(root, "records.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(record_data, ensure_ascii=False) + "\n")
        with pytest.raises(ValueError, match="consent_status"):
            HumanWrittenAnnotationWorkspace(root)

    def test_empty_review_text_rejected(self, tmp_path):
        root = str(tmp_path / "bad3")
        os.makedirs(root, exist_ok=True)
        record_data = {
            "review_id": "bad-3",
            "source_type": "HUMAN_WRITTEN_RMC",
            "review_text": "   ",
            "consent_status": "CONSENTED",
            "annotation_status": "UNANNOTATED",
            "spam": None,
            "deception": None,
            "toxicity": None,
            "advertising": None,
            "off_topic": None,
            "pii": None,
            "language_mix": None,
            "college_category": None,
            "dataset_version": None,
            "split_membership": None,
            "split_group_id": None,
        }
        with open(os.path.join(root, "records.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(record_data, ensure_ascii=False) + "\n")
        with pytest.raises(ValueError, match="review_text"):
            HumanWrittenAnnotationWorkspace(root)

    def test_source_text_raw_rejected(self, tmp_path):
        root = str(tmp_path / "bad4")
        os.makedirs(root, exist_ok=True)
        record_data = {
            "review_id": "bad-4",
            "source_type": "HUMAN_WRITTEN_RMC",
            "source_text_raw": "secret raw",
            "review_text": "safe text",
            "consent_status": "CONSENTED",
            "annotation_status": "UNANNOTATED",
            "spam": None,
            "deception": None,
            "toxicity": None,
            "advertising": None,
            "off_topic": None,
            "pii": None,
            "language_mix": None,
            "college_category": None,
            "dataset_version": None,
            "split_membership": None,
            "split_group_id": None,
        }
        with open(os.path.join(root, "records.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(record_data, ensure_ascii=False) + "\n")
        with pytest.raises(ValueError, match="source_text_raw"):
            HumanWrittenAnnotationWorkspace(root)

    def test_dataset_version_set_rejected(self, tmp_path):
        root = str(tmp_path / "bad5")
        os.makedirs(root, exist_ok=True)
        record_data = {
            "review_id": "bad-5",
            "source_type": "HUMAN_WRITTEN_RMC",
            "review_text": "text",
            "consent_status": "CONSENTED",
            "annotation_status": "UNANNOTATED",
            "spam": None,
            "deception": None,
            "toxicity": None,
            "advertising": None,
            "off_topic": None,
            "pii": None,
            "language_mix": None,
            "college_category": None,
            "dataset_version": "v1",
            "split_membership": None,
            "split_group_id": None,
        }
        with open(os.path.join(root, "records.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(record_data, ensure_ascii=False) + "\n")
        with pytest.raises(ValueError, match="dataset_version"):
            HumanWrittenAnnotationWorkspace(root)

    def test_split_membership_set_rejected(self, tmp_path):
        root = str(tmp_path / "bad6")
        os.makedirs(root, exist_ok=True)
        record_data = {
            "review_id": "bad-6",
            "source_type": "HUMAN_WRITTEN_RMC",
            "review_text": "text",
            "consent_status": "CONSENTED",
            "annotation_status": "UNANNOTATED",
            "spam": None,
            "deception": None,
            "toxicity": None,
            "advertising": None,
            "off_topic": None,
            "pii": None,
            "language_mix": None,
            "college_category": None,
            "dataset_version": None,
            "split_membership": "train",
            "split_group_id": None,
        }
        with open(os.path.join(root, "records.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(record_data, ensure_ascii=False) + "\n")
        with pytest.raises(ValueError, match="split_membership"):
            HumanWrittenAnnotationWorkspace(root)

    def test_split_group_id_set_rejected(self, tmp_path):
        root = str(tmp_path / "bad7")
        os.makedirs(root, exist_ok=True)
        record_data = {
            "review_id": "bad-7",
            "source_type": "HUMAN_WRITTEN_RMC",
            "review_text": "text",
            "consent_status": "CONSENTED",
            "annotation_status": "UNANNOTATED",
            "spam": None,
            "deception": None,
            "toxicity": None,
            "advertising": None,
            "off_topic": None,
            "pii": None,
            "language_mix": None,
            "college_category": None,
            "dataset_version": None,
            "split_membership": None,
            "split_group_id": "grp-1",
        }
        with open(os.path.join(root, "records.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(record_data, ensure_ascii=False) + "\n")
        with pytest.raises(ValueError, match="split_group_id"):
            HumanWrittenAnnotationWorkspace(root)

    def test_duplicate_review_id_rejected(self, tmp_path):
        root = str(tmp_path / "bad8")
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "records.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "review_id": "dup-1",
                "source_type": "HUMAN_WRITTEN_RMC",
                "review_text": "first",
                "consent_status": "CONSENTED",
                "annotation_status": "UNANNOTATED",
                "spam": None, "deception": None, "toxicity": None,
                "advertising": None, "off_topic": None, "pii": None,
                "language_mix": None, "college_category": None,
                "dataset_version": None, "split_membership": None,
                "split_group_id": None,
            }, ensure_ascii=False) + "\n")
            fh.write(json.dumps({
                "review_id": "dup-1",
                "source_type": "HUMAN_WRITTEN_RMC",
                "review_text": "second",
                "consent_status": "CONSENTED",
                "annotation_status": "UNANNOTATED",
                "spam": None, "deception": None, "toxicity": None,
                "advertising": None, "off_topic": None, "pii": None,
                "language_mix": None, "college_category": None,
                "dataset_version": None, "split_membership": None,
                "split_group_id": None,
            }, ensure_ascii=False) + "\n")
        with pytest.raises(ValueError, match="Duplicate review_id"):
            HumanWrittenAnnotationWorkspace(root)

    def test_malformed_json_rejected(self, tmp_path):
        root = str(tmp_path / "bad9")
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "records.jsonl"), "w", encoding="utf-8") as fh:
            fh.write("not json\n")
        with pytest.raises(ValueError, match="Malformed JSON"):
            HumanWrittenAnnotationWorkspace(root)

    def test_blank_review_id_rejected(self, tmp_path):
        root = str(tmp_path / "bad10")
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "records.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "review_id": "",
                "source_type": "HUMAN_WRITTEN_RMC",
                "review_text": "text",
                "consent_status": "CONSENTED",
                "annotation_status": "UNANNOTATED",
                "spam": None, "deception": None, "toxicity": None,
                "advertising": None, "off_topic": None, "pii": None,
                "language_mix": None, "college_category": None,
                "dataset_version": None, "split_membership": None,
                "split_group_id": None,
            }, ensure_ascii=False) + "\n")
        with pytest.raises(ValueError, match="review_id"):
            HumanWrittenAnnotationWorkspace(root)


# ---------------------------------------------------------------------------
# C — record selection (Requirements 5, 6)
# ---------------------------------------------------------------------------

class TestRecordSelection:
    def test_select_a_returns_first_unannotated(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), [
            "First review",
            "Second review",
            "Third review",
        ])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        task = ws.select_next("A")
        assert task is not None
        assert task.review_id == ids[0]
        assert task.review_text == "First review"
        assert task.slot == "A"

    def test_select_b_requires_a_first(self, tmp_path):
        _seed_collection(str(tmp_path / "data"), ["First review"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        assert ws.select_next("B") is None

    def test_select_advances_after_a_submission(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), [
            "First review",
            "Second review",
        ])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))

        ws.submit_annotation(
            slot="A",
            review_id=ids[0],
            annotator_id="ann-a",
            choices=_build_choices(language_mix=LanguageMix.ENGLISH),
        )

        task = ws.select_next("A")
        assert task is not None
        assert task.review_id == ids[1]

        task_b = ws.select_next("B")
        assert task_b is not None
        assert task_b.review_id == ids[0]

    def test_select_returns_none_when_exhausted(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["First review"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))

        ws.submit_annotation(
            slot="A",
            review_id=ids[0],
            annotator_id="ann-a",
            choices=_build_choices(),
        )
        ws.submit_annotation(
            slot="B",
            review_id=ids[0],
            annotator_id="ann-b",
            choices=_build_choices(),
        )

        assert ws.select_next("A") is None
        assert ws.select_next("B") is None

    def test_deterministic_selection_order(self, tmp_path):
        texts = [f"Review number {i}" for i in range(20)]
        ids = _seed_collection(str(tmp_path / "data"), texts)

        firsts = []
        for _ in range(3):
            ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
            task = ws.select_next("A")
            firsts.append(task.review_id if task else None)
        assert all(f == ids[0] for f in firsts)

    def test_invalid_slot_raises(self, tmp_path):
        _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        with pytest.raises(ValueError, match="slot"):
            ws.select_next("X")

    def test_get_task_a_eligible(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        task = ws.get_task(ids[0], "A")
        assert task.review_id == ids[0]
        assert task.review_text == "text"
        assert task.slot == "A"

    def test_get_task_unknown_raises(self, tmp_path):
        _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        with pytest.raises(ValueError, match="Unknown review_id"):
            ws.get_task("nonexistent", "A")

    def test_get_task_b_before_a_raises(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        with pytest.raises(ValueError, match="Annotator A has not submitted"):
            ws.get_task(ids[0], "B")

    def test_get_task_non_human_written_raises(self, tmp_path):
        """Non-Human records are rejected during workspace loading."""
        root = str(tmp_path / "bad")
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "records.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "review_id": "ctrl-1",
                "source_type": "CONTROLLED_RMC",
                "review_text": "text",
                "consent_status": "CONSENTED",
                "annotation_status": "UNANNOTATED",
                "spam": None, "deception": None, "toxicity": None,
                "advertising": None, "off_topic": None, "pii": None,
                "language_mix": None, "college_category": None,
                "dataset_version": None, "split_membership": None,
                "split_group_id": None,
            }, ensure_ascii=False) + "\n")
        with pytest.raises(ValueError, match="source_type"):
            HumanWrittenAnnotationWorkspace(root)


# ---------------------------------------------------------------------------
# D — submit_annotation (Requirements 4, 7, 8, 9)
# ---------------------------------------------------------------------------

class TestSubmitAnnotation:
    def test_a_submission_changes_status(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        updated = ws.submit_annotation(
            slot="A",
            review_id=ids[0],
            annotator_id="ann-a",
            choices=_build_choices(),
        )
        assert updated.annotation_status == AnnotationStatus.ANNOTATING
        assert updated.annotator_A_id == "ann-a"
        assert updated.deception is None
        # A submission persists deception=-1 per Human-Written policy
        subs_path = os.path.join(str(tmp_path / "data"), "annotation_submissions.jsonl")
        with open(subs_path, "r", encoding="utf-8") as fh:
            first = json.loads(fh.readline())
        assert first["deception"] == -1

    def test_b_submission_agreement_stays_annotating(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        ws.submit_annotation(
            slot="A",
            review_id=ids[0],
            annotator_id="ann-a",
            choices=_build_choices(),
        )
        updated = ws.submit_annotation(
            slot="B",
            review_id=ids[0],
            annotator_id="ann-b",
            choices=_build_choices(),
        )
        assert updated.annotation_status == AnnotationStatus.ANNOTATING
        assert updated.annotator_B_id == "ann-b"

    def test_b_submission_disagreement_goes_to_adjudication(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        ws.submit_annotation(
            slot="A",
            review_id=ids[0],
            annotator_id="ann-a",
            choices=_build_choices(),
        )
        updated = ws.submit_annotation(
            slot="B",
            review_id=ids[0],
            annotator_id="ann-b",
            choices=_build_choices(spam=1),
        )
        assert updated.annotation_status == AnnotationStatus.ADJUDICATION_REQUIRED

    def test_duplicate_a_raises(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        ws.submit_annotation(
            slot="A",
            review_id=ids[0],
            annotator_id="ann-a",
            choices=_build_choices(),
        )
        with pytest.raises(ValueError, match="Annotator A already submitted"):
            ws.submit_annotation(
                slot="A",
                review_id=ids[0],
                annotator_id="ann-a2",
                choices=_build_choices(),
            )

    def test_duplicate_b_raises(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        ws.submit_annotation(
            slot="A",
            review_id=ids[0],
            annotator_id="ann-a",
            choices=_build_choices(),
        )
        ws.submit_annotation(
            slot="B",
            review_id=ids[0],
            annotator_id="ann-b",
            choices=_build_choices(),
        )
        with pytest.raises(ValueError, match="Annotator B already submitted"):
            ws.submit_annotation(
                slot="B",
                review_id=ids[0],
                annotator_id="ann-b2",
                choices=_build_choices(),
            )

    def test_same_annotator_both_slots_raises(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        ws.submit_annotation(
            slot="A",
            review_id=ids[0],
            annotator_id="ann-same",
            choices=_build_choices(),
        )
        with pytest.raises(ValueError, match="must be independent"):
            ws.submit_annotation(
                slot="B",
                review_id=ids[0],
                annotator_id="ann-same",
                choices=_build_choices(),
            )

    def test_b_without_a_raises(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        with pytest.raises(ValueError, match="Annotator A has not submitted"):
            ws.submit_annotation(
                slot="B",
                review_id=ids[0],
                annotator_id="ann-b",
                choices=_build_choices(),
            )

    def test_deception_in_choices_raises(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        with pytest.raises(ValueError, match="deception"):
            ws.submit_annotation(
                slot="A",
                review_id=ids[0],
                annotator_id="ann-a",
                choices={**_build_choices(), "deception": 1},
            )

    def test_missing_key_raises(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        with pytest.raises(ValueError, match="Missing required choices"):
            ws.submit_annotation(
                slot="A",
                review_id=ids[0],
                annotator_id="ann-a",
                choices={"spam": 0, "toxicity": 0, "advertising": 0,
                         "off_topic": 0, "pii": 0},
            )

    def test_unknown_review_id_raises(self, tmp_path):
        _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        with pytest.raises(ValueError, match="Unknown review_id"):
            ws.submit_annotation(
                slot="A",
                review_id="nonexistent",
                annotator_id="ann-a",
                choices=_build_choices(),
            )

    def test_invalid_slot_raises(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        with pytest.raises(ValueError, match="slot"):
            ws.submit_annotation(
                slot="X",
                review_id=ids[0],
                annotator_id="ann-a",
                choices=_build_choices(),
            )

    def test_non_string_annotator_id_raises(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        with pytest.raises(TypeError, match="annotator_id"):
            ws.submit_annotation(
                slot="A",
                review_id=ids[0],
                annotator_id=123,
                choices=_build_choices(),
            )

    def test_empty_annotator_id_raises(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        with pytest.raises(TypeError, match="annotator_id"):
            ws.submit_annotation(
                slot="A",
                review_id=ids[0],
                annotator_id="",
                choices=_build_choices(),
            )

    def test_string_binary_label_rejected(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        with pytest.raises(ValueError, match="spam"):
            ws.submit_annotation(
                slot="A",
                review_id=ids[0],
                annotator_id="ann-a",
                choices={**_build_choices(), "spam": "1"},
            )

    def test_bool_binary_label_rejected(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        with pytest.raises(ValueError, match="spam"):
            ws.submit_annotation(
                slot="A",
                review_id=ids[0],
                annotator_id="ann-a",
                choices={**_build_choices(), "spam": True},
            )

    def test_integer_two_rejected(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        with pytest.raises(ValueError, match="spam"):
            ws.submit_annotation(
                slot="A",
                review_id=ids[0],
                annotator_id="ann-a",
                choices={**_build_choices(), "spam": 2},
            )

    def test_string_language_mix_rejected(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        with pytest.raises(ValueError, match="language_mix"):
            ws.submit_annotation(
                slot="A",
                review_id=ids[0],
                annotator_id="ann-a",
                choices={**_build_choices(), "language_mix": "ENGLISH"},
            )

    def test_string_college_category_rejected(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        with pytest.raises(ValueError, match="college_category"):
            ws.submit_annotation(
                slot="A",
                review_id=ids[0],
                annotator_id="ann-a",
                choices={**_build_choices(), "college_category": "ACADEMICS"},
            )


# ---------------------------------------------------------------------------
# E — college_category optional
# ---------------------------------------------------------------------------

class TestCollegeCategoryOptional:
    def test_none_college_category_accepted(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        updated = ws.submit_annotation(
            slot="A",
            review_id=ids[0],
            annotator_id="ann-a",
            choices=_build_choices(),
        )
        assert updated.college_category is None

    def test_college_category_accepted(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        ws.submit_annotation(
            slot="A",
            review_id=ids[0],
            annotator_id="ann-a",
            choices=_build_choices(college_category=CollegeCategory.PLACEMENTS),
        )
        # A submission does not materialize college_category onto record
        # (frozen workflow behavior); value is in persisted submission
        subs_path = os.path.join(str(tmp_path / "data"), "annotation_submissions.jsonl")
        with open(subs_path, "r", encoding="utf-8") as fh:
            first = json.loads(fh.readline())
        assert first["college_category"] == "PLACEMENTS"


# ---------------------------------------------------------------------------
# F — persistence (Requirements 10, 11)
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_submission_persisted(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        ws.submit_annotation(
            slot="A",
            review_id=ids[0],
            annotator_id="ann-a",
            choices=_build_choices(),
        )
        assert os.path.exists(
            os.path.join(str(tmp_path / "data"), "annotation_submissions.jsonl")
        )

    def test_record_state_persisted(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["text"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        ws.submit_annotation(
            slot="A",
            review_id=ids[0],
            annotator_id="ann-a",
            choices=_build_choices(),
        )
        assert os.path.exists(
            os.path.join(str(tmp_path / "data"), "annotation_record_states.jsonl")
        )

    def test_reconstruction_after_restart(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["text"])
        ws1 = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        ws1.submit_annotation(
            slot="A",
            review_id=ids[0],
            annotator_id="ann-a",
            choices=_build_choices(),
        )

        ws2 = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        assert ws2.get_record(ids[0]).annotation_status == AnnotationStatus.ANNOTATING
        with pytest.raises(ValueError, match="Annotator A already submitted"):
            ws2.submit_annotation(
                slot="A",
                review_id=ids[0],
                annotator_id="ann-a2",
                choices=_build_choices(),
            )

    def test_records_jsonl_not_mutated(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["text"])
        with open(os.path.join(str(tmp_path / "data"), "records.jsonl"), "r", encoding="utf-8") as fh:
            original_count = sum(1 for _ in fh)
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        ws.submit_annotation(
            slot="A",
            review_id=ids[0],
            annotator_id="ann-a",
            choices=_build_choices(),
        )
        with open(os.path.join(str(tmp_path / "data"), "records.jsonl"), "r", encoding="utf-8") as fh:
            new_count = sum(1 for _ in fh)
        assert new_count == original_count


# ---------------------------------------------------------------------------
# G — eligible_count
# ---------------------------------------------------------------------------

class TestEligibleCount:
    def test_eligible_count_a(self, tmp_path):
        _seed_collection(str(tmp_path / "data"), ["a", "b", "c"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        assert ws.get_eligible_count("A") == 3
        assert ws.get_eligible_count("B") == 0

    def test_eligible_count_updates(self, tmp_path):
        ids = _seed_collection(str(tmp_path / "data"), ["a", "b"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        ws.submit_annotation(
            slot="A",
            review_id=ids[0],
            annotator_id="ann-a",
            choices=_build_choices(),
        )
        assert ws.get_eligible_count("A") == 1
        assert ws.get_eligible_count("B") == 1

    def test_invalid_slot_raises(self, tmp_path):
        _seed_collection(str(tmp_path / "data"), ["a"])
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        with pytest.raises(ValueError, match="slot"):
            ws.get_eligible_count("X")


# ---------------------------------------------------------------------------
# H — language_mix round-trip and state revalidation
# ---------------------------------------------------------------------------

class TestLanguageMixRoundTrip:
    def test_hinglish_roundtrip(self, tmp_path):
        """Frozen workflow: A persists in submission, B materializes on record."""
        ids = _seed_collection(
            str(tmp_path / "data"),
            ["यह एक हिंदी review है, placement is good"],
        )
        ws = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))

        # A submission: language_mix persists in submission, not on record
        ws.submit_annotation(
            slot="A",
            review_id=ids[0],
            annotator_id="ann-a",
            choices=_build_choices(language_mix=LanguageMix.HINGLISH),
        )
        subs_path = os.path.join(str(tmp_path / "data"), "annotation_submissions.jsonl")
        with open(subs_path, "r", encoding="utf-8") as fh:
            first = json.loads(fh.readline())
        assert first["language_mix"] == "HINGLISH"
        rec = ws.get_record(ids[0])
        assert rec.language_mix is None  # frozen: A doesn't materialize

        # B submission: B materializes language_mix onto record
        updated = ws.submit_annotation(
            slot="B",
            review_id=ids[0],
            annotator_id="ann-b",
            choices=_build_choices(language_mix=LanguageMix.HINGLISH),
        )
        assert updated.language_mix == LanguageMix.HINGLISH

        # Reconstruct from disk and verify
        ws2 = HumanWrittenAnnotationWorkspace(str(tmp_path / "data"))
        rec2 = ws2.get_record(ids[0])
        assert rec2.language_mix == LanguageMix.HINGLISH
