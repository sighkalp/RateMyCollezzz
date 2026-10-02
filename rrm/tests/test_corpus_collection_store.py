"""Tests for rrm.corpus.collection_store module.

Public API tested:
    CollectionSession
    HumanWrittenCollectionStore
"""

from __future__ import annotations

import dataclasses
import json
import os

import pytest

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
# Fixtures / Helpers
# ---------------------------------------------------------------------------


def _make_workflow() -> CorpusWorkflow:
    return CorpusWorkflow(
        AnnotationSubmissionStore(),
        ControlledProtocolTruthStore(),
        DispositionRegister(),
        ReannotationRegister(),
    )


def _collect(
    workflow: CorpusWorkflow,
    text: str = "Good college with nice campus",
    collected_at: str = "2025-01-01T00:00:00Z",
    retain_raw: bool = False,
) -> HumanWrittenIntakeResult:
    return collect_human_written_review(
        workflow=workflow,
        submitted_text=text,
        consent_granted=True,
        collected_at=collected_at,
        collection_method="web_form",
        retain_source_text_raw=retain_raw,
    )


@pytest.fixture
def store(tmp_path):
    return HumanWrittenCollectionStore(str(tmp_path / "collection"))


def _session(store, started_at="2025-01-01T00:00:00Z"):
    return store.start_session(
        started_at=started_at,
        collection_method="local_cli",
    )


# ---------------------------------------------------------------------------
# A — store initialization (Covers requirement 1)
# ---------------------------------------------------------------------------


class TestStoreInit:
    def test_creates_root_dir(self, tmp_path):
        root = str(tmp_path / "new_dir")
        assert not os.path.exists(root)
        HumanWrittenCollectionStore(root)
        assert os.path.isdir(root)

    def test_existing_dir_is_fine(self, tmp_path):
        store = HumanWrittenCollectionStore(str(tmp_path))
        assert store.root_dir == str(tmp_path)

    def test_non_string_root_raises(self):
        with pytest.raises(TypeError, match="root_dir"):
            HumanWrittenCollectionStore(123)

    def test_root_dir_stored(self, tmp_path):
        store = HumanWrittenCollectionStore(str(tmp_path))
        assert store.root_dir == str(tmp_path)


# ---------------------------------------------------------------------------
# B — session creation (Covers requirements 2, 3, 4)
# ---------------------------------------------------------------------------


class TestStartSession:
    def test_returns_collection_session(self, store):
        s = _session(store)
        assert isinstance(s, CollectionSession)

    def test_session_id_is_system_generated(self, store):
        s = _session(store)
        assert s.session_id.startswith("session-")

    def test_manifest_created(self, store):
        _session(store)
        manifest_path = os.path.join(store.root_dir, "session_manifest.json")
        assert os.path.exists(manifest_path)

    def test_initial_record_count_zero(self, store):
        _session(store)
        with open(
            os.path.join(store.root_dir, "session_manifest.json"),
            "r", encoding="utf-8",
        ) as fh:
            manifest = json.load(fh)
        assert manifest["record_count"] == 0

    def test_session_fields(self, store):
        s = _session(store)
        assert s.started_at == "2025-01-01T00:00:00Z"
        assert s.collection_method == "local_cli"

    def test_session_id_unique(self, store):
        s1 = _session(store)
        s2 = _session(store)
        assert s1.session_id != s2.session_id

    def test_blank_started_at_raises(self, store):
        with pytest.raises(ValueError, match="started_at"):
            store.start_session(
                started_at="   ",
                collection_method="local_cli",
            )

    def test_blank_collection_method_raises(self, store):
        with pytest.raises(ValueError, match="collection_method"):
            store.start_session(
                started_at="2025-01-01T00:00:00Z",
                collection_method="",
            )

    def test_non_string_started_at_raises(self, store):
        with pytest.raises(TypeError, match="started_at"):
            store.start_session(
                started_at=123,
                collection_method="local_cli",
            )

    def test_non_string_collection_method_raises(self, store):
        with pytest.raises(TypeError, match="collection_method"):
            store.start_session(
                started_at="2025-01-01T00:00:00Z",
                collection_method=456,
            )

    def test_session_is_frozen(self, store):
        s = _session(store)
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.session_id = "changed"


# ---------------------------------------------------------------------------
# C — append_intake happy path (Covers requirements 5, 6, 7, 8, 9, 10, 11, 26)
# ---------------------------------------------------------------------------


class TestAppendIntakeHappy:
    def test_valid_result_appends(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        store.append_intake(session, result)
        # No exception = success

    def test_records_jsonl_created(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        store.append_intake(session, result)
        assert os.path.exists(
            os.path.join(store.root_dir, "records.jsonl")
        )

    def test_consents_jsonl_created(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        store.append_intake(session, result)
        assert os.path.exists(
            os.path.join(store.root_dir, "consents.jsonl")
        )

    def test_pii_file_absent_when_clean(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        store.append_intake(session, result)
        assert not os.path.exists(
            os.path.join(store.root_dir, "pii_evidence.jsonl")
        )

    def test_pii_file_created_when_pii_present(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="Email me at test@example.com",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        store.append_intake(session, result)
        assert os.path.exists(
            os.path.join(store.root_dir, "pii_evidence.jsonl")
        )

    def test_source_text_raw_not_persisted(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        store.append_intake(session, result)
        with open(
            os.path.join(store.root_dir, "records.jsonl"),
            "r", encoding="utf-8",
        ) as fh:
            record = json.loads(fh.readline())
        assert "source_text_raw" not in record

    def test_consent_has_no_review_text(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        store.append_intake(session, result)
        with open(
            os.path.join(store.root_dir, "consents.jsonl"),
            "r", encoding="utf-8",
        ) as fh:
            consent = json.loads(fh.readline())
        assert "review_text" not in consent
        assert "review_text" not in str(consent)

    def test_pii_evidence_has_no_matched_values(
        self, store
    ):
        wf = _make_workflow()
        session = _session(store)
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="test@example.com",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        store.append_intake(session, result)
        with open(
            os.path.join(store.root_dir, "pii_evidence.jsonl"),
            "r", encoding="utf-8",
        ) as fh:
            pii = json.loads(fh.readline())
        pii_str = json.dumps(pii)
        assert "test@example.com" not in pii_str

    def test_hinglish_utf8_roundtrip(self, store):
        wf = _make_workflow()
        session = _session(store)
        text = "यह एक हिंदी review है, placement is good"
        result = _collect(wf, text=text)
        store.append_intake(session, result)
        with open(
            os.path.join(store.root_dir, "records.jsonl"),
            "r", encoding="utf-8",
        ) as fh:
            record = json.loads(fh.readline())
        assert record["review_text"] == text

    def test_manifest_count_after_single_append(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        store.append_intake(session, result)
        with open(
            os.path.join(store.root_dir, "session_manifest.json"),
            "r", encoding="utf-8",
        ) as fh:
            manifest = json.load(fh)
        assert manifest["record_count"] == 1

    def test_record_count_increments_on_two_appends(self, store):
        wf = _make_workflow()
        session = _session(store)
        r1 = _collect(wf, text="Review one")
        r2 = _collect(wf, text="Review two")
        store.append_intake(session, r1)
        store.append_intake(session, r2)
        with open(
            os.path.join(store.root_dir, "session_manifest.json"),
            "r", encoding="utf-8",
        ) as fh:
            manifest = json.load(fh)
        assert manifest["record_count"] == 2

    def test_record_fields_persisted_correctly(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        store.append_intake(session, result)
        with open(
            os.path.join(store.root_dir, "records.jsonl"),
            "r", encoding="utf-8",
        ) as fh:
            record = json.loads(fh.readline())
        assert record["review_id"] == result.record.review_id
        assert record["review_text"] == "Good college with nice campus"
        assert record["source_type"] == "HUMAN_WRITTEN_RMC"
        assert record["annotation_status"] == "UNANNOTATED"
        assert record["consent_status"] == "CONSENTED"
        assert record["collection_method"] == "web_form"
        assert record["spam"] is None
        assert record["deception"] is None
        assert record["toxicity"] is None
        assert record["advertising"] is None
        assert record["off_topic"] is None
        assert record["pii"] is None
        assert record["language_mix"] is None
        assert record["college_category"] is None


# ---------------------------------------------------------------------------
# D — eligibility rejection (Covers requirements 12–22)
# ---------------------------------------------------------------------------


class TestEligibilityRejection:
    def test_source_text_raw_raises(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf, retain_raw=True)
        with pytest.raises(ValueError, match="source_text_raw"):
            store.append_intake(session, result)

    def test_wrong_source_type_raises(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        bad_record = dataclasses.replace(
            result.record, source_type=SourceType.CONTROLLED_RMC
        )
        bad_result = dataclasses.replace(result, record=bad_record)
        with pytest.raises(ValueError, match="source_type"):
            store.append_intake(session, bad_result)

    def test_non_unannotated_status_raises(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        bad_record = dataclasses.replace(
            result.record, annotation_status=AnnotationStatus.FINAL
        )
        bad_result = dataclasses.replace(result, record=bad_record)
        with pytest.raises(ValueError, match="annotation_status"):
            store.append_intake(session, bad_result)

    def test_non_consented_status_raises(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        bad_record = dataclasses.replace(
            result.record, consent_status="REVOKED"
        )
        bad_result = dataclasses.replace(result, record=bad_record)
        with pytest.raises(ValueError, match="consent_status"):
            store.append_intake(session, bad_result)

    def test_dataset_version_set_raises(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        bad_record = dataclasses.replace(
            result.record, dataset_version="v1"
        )
        bad_result = dataclasses.replace(result, record=bad_record)
        with pytest.raises(ValueError, match="dataset_version"):
            store.append_intake(session, bad_result)

    def test_split_membership_set_raises(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        bad_record = dataclasses.replace(
            result.record, split_membership="train"
        )
        bad_result = dataclasses.replace(result, record=bad_record)
        with pytest.raises(ValueError, match="split_membership"):
            store.append_intake(session, bad_result)

    def test_split_group_id_set_raises(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        bad_record = dataclasses.replace(
            result.record, split_group_id="group-1"
        )
        bad_result = dataclasses.replace(result, record=bad_record)
        with pytest.raises(ValueError, match="split_group_id"):
            store.append_intake(session, bad_result)

    def test_spam_label_set_raises(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        bad_record = dataclasses.replace(result.record, spam=1)
        bad_result = dataclasses.replace(result, record=bad_record)
        with pytest.raises(ValueError, match="spam"):
            store.append_intake(session, bad_result)

    def test_deception_label_set_raises(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        bad_record = dataclasses.replace(result.record, deception=1)
        bad_result = dataclasses.replace(result, record=bad_record)
        with pytest.raises(ValueError, match="deception"):
            store.append_intake(session, bad_result)

    def test_toxicity_label_set_raises(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        bad_record = dataclasses.replace(result.record, toxicity=1)
        bad_result = dataclasses.replace(result, record=bad_record)
        with pytest.raises(ValueError, match="toxicity"):
            store.append_intake(session, bad_result)

    def test_advertising_label_set_raises(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        bad_record = dataclasses.replace(result.record, advertising=1)
        bad_result = dataclasses.replace(result, record=bad_record)
        with pytest.raises(ValueError, match="advertising"):
            store.append_intake(session, bad_result)

    def test_off_topic_label_set_raises(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        bad_record = dataclasses.replace(result.record, off_topic=1)
        bad_result = dataclasses.replace(result, record=bad_record)
        with pytest.raises(ValueError, match="off_topic"):
            store.append_intake(session, bad_result)

    def test_pii_label_set_raises(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        bad_record = dataclasses.replace(result.record, pii=1)
        bad_result = dataclasses.replace(result, record=bad_record)
        with pytest.raises(ValueError, match="pii"):
            store.append_intake(session, bad_result)

    def test_language_mix_set_raises(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        bad_record = dataclasses.replace(
            result.record, language_mix=LanguageMix.HINGLISH
        )
        bad_result = dataclasses.replace(result, record=bad_record)
        with pytest.raises(ValueError, match="language_mix"):
            store.append_intake(session, bad_result)

    def test_college_category_set_raises(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        bad_record = dataclasses.replace(
            result.record, college_category=CollegeCategory.ACADEMICS
        )
        bad_result = dataclasses.replace(result, record=bad_record)
        with pytest.raises(ValueError, match="college_category"):
            store.append_intake(session, bad_result)

    def test_empty_review_text_raises(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        bad_record = dataclasses.replace(
            result.record, review_text=""
        )
        bad_result = dataclasses.replace(result, record=bad_record)
        with pytest.raises(ValueError, match="review_text"):
            store.append_intake(session, bad_result)


# ---------------------------------------------------------------------------
# E — cross-artifact consistency (Covers requirements 23, 24, 25)
# ---------------------------------------------------------------------------


class TestCrossArtifactConsistency:
    def test_consent_review_id_mismatch_raises(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        bad_consent = dataclasses.replace(
            result.consent_artifact,
            review_id="consent-" + "x" * 32,
        )
        bad_result = dataclasses.replace(
            result, consent_artifact=bad_consent
        )
        with pytest.raises(ValueError, match="review_id"):
            store.append_intake(session, bad_result)

    def test_pii_evidence_review_id_mismatch_raises(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="test@example.com",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        bad_pii = dataclasses.replace(
            result.pii_evidence,
            review_id="review-" + "x" * 32,
        )
        bad_result = dataclasses.replace(result, pii_evidence=bad_pii)
        with pytest.raises(ValueError, match="review_id"):
            store.append_intake(session, bad_result)

    def test_pii_evidence_id_mismatch_raises(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = collect_human_written_review(
            workflow=wf,
            submitted_text="test@example.com",
            consent_granted=True,
            collected_at="2025-01-01T00:00:00Z",
            collection_method="web_form",
        )
        bad_record = dataclasses.replace(
            result.record,
            pii_evidence_id="evid-" + "x" * 32,
        )
        bad_result = dataclasses.replace(result, record=bad_record)
        with pytest.raises(ValueError, match="pii_evidence_id"):
            store.append_intake(session, bad_result)


# ---------------------------------------------------------------------------
# F — duplicate protection (Covers requirement 12)
# ---------------------------------------------------------------------------


class TestDuplicateProtection:
    def test_duplicate_review_id_raises(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        store.append_intake(session, result)
        with pytest.raises(ValueError, match="already exists"):
            store.append_intake(session, result)

    def test_different_review_ids_both_accepted(self, store):
        wf = _make_workflow()
        session = _session(store)
        r1 = _collect(wf, text="Review one")
        r2 = _collect(wf, text="Review two")
        store.append_intake(session, r1)
        store.append_intake(session, r2)
        with open(
            os.path.join(store.root_dir, "records.jsonl"),
            "r", encoding="utf-8",
        ) as fh:
            lines = [l.strip() for l in fh if l.strip()]
        assert len(lines) == 2
        ids = [json.loads(l)["review_id"] for l in lines]
        assert ids[0] == r1.record.review_id
        assert ids[1] == r2.record.review_id


# ---------------------------------------------------------------------------
# G — record_count after rejection (Covers requirement 27 — Blocker F-A)
# ---------------------------------------------------------------------------


class TestRecordCountAfterRejection:
    def test_count_unchanged_after_rejection(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        store.append_intake(session, result)
        # Capture current count
        with open(
            os.path.join(store.root_dir, "session_manifest.json"),
            "r", encoding="utf-8",
        ) as fh:
            manifest = json.load(fh)
        assert manifest["record_count"] == 1

        # Attempt invalid append
        bad_result = _collect(wf, retain_raw=True)
        with pytest.raises(ValueError):
            store.append_intake(session, bad_result)

        # Count must still be 1
        with open(
            os.path.join(store.root_dir, "session_manifest.json"),
            "r", encoding="utf-8",
        ) as fh:
            manifest = json.load(fh)
        assert manifest["record_count"] == 1


# ---------------------------------------------------------------------------
# H — session consistency
# ---------------------------------------------------------------------------


class TestSessionConsistency:
    def test_session_mismatch_raises(self, store):
        wf = _make_workflow()
        session1 = _session(store)
        session2 = store.start_session(
            started_at="2025-01-01T00:00:00Z",
            collection_method="local_cli",
        )
        assert session1.session_id != session2.session_id
        result = _collect(wf)
        with pytest.raises(ValueError, match="Session mismatch"):
            store.append_intake(session1, result)

    def test_append_without_manifest_raises(self, tmp_path):
        wf = _make_workflow()
        store = HumanWrittenCollectionStore(
            str(tmp_path / "empty_store")
        )
        result = _collect(wf)
        fake_session = CollectionSession(
            session_id="session-unknown",
            started_at="2025-01-01T00:00:00Z",
            collection_method="local_cli",
        )
        with pytest.raises(RuntimeError, match="session manifest"):
            store.append_intake(fake_session, result)

    def test_manifest_survives_append(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        store.append_intake(session, result)
        with open(
            os.path.join(store.root_dir, "session_manifest.json"),
            "r", encoding="utf-8",
        ) as fh:
            manifest = json.load(fh)
        assert manifest["session_id"] == session.session_id
        assert manifest["record_count"] == 1


# ---------------------------------------------------------------------------
# I — type validation
# ---------------------------------------------------------------------------


class TestTypeValidation:
    def test_non_session_raises(self, store):
        wf = _make_workflow()
        session = _session(store)
        result = _collect(wf)
        with pytest.raises(TypeError, match="session"):
            store.append_intake("not a session", result)

    def test_non_result_raises(self, store):
        session = _session(store)
        with pytest.raises(TypeError, match="result"):
            store.append_intake(session, "not a result")
