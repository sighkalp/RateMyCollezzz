"""Restricted internal Gate-C collection persistence.

This module provides append-only local storage for Human-Written RMC
collection-stage records.  It is NOT portable export.  It is NOT
Gate-D export.  It is NOT dataset freeze.

Architectural note:
    RRM UNDERSTANDS. TRUST DECIDES.

    This module is operational collection infrastructure only.  It
    persists safe collection-stage evidence for later annotation.
    It does NOT make moderation, trust, or policy decisions.

Public API:
    CollectionSession
    HumanWrittenCollectionStore
"""

from __future__ import annotations

import dataclasses
import json
import os
import uuid
from typing import Optional

from rrm.corpus.models import (
    AnnotationStatus,
    CanonicalRecord,
    SourceType,
)
from rrm.corpus.intake import HumanWrittenIntakeResult


# ---------------------------------------------------------------------------
# Session type
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CollectionSession:
    """Immutable collection session metadata.

    Attributes
    ----------
    session_id : str
        Unique session identifier.
    started_at : str
        ISO 8601 timestamp of session start.
    collection_method : str
        How the collection was conducted (e.g., "local_cli").
    """

    session_id: str
    started_at: str
    collection_method: str


# ---------------------------------------------------------------------------
# Collection store
# ---------------------------------------------------------------------------

class HumanWrittenCollectionStore:
    """Append-only restricted internal storage for Human-Written RMC records.

    This store persists collection-stage evidence to JSONL files within
    a caller-supplied root directory.  It enforces strict eligibility
    checks to prevent unsafe records from entering local collection.

    Attributes
    ----------
    root_dir : str
        Root directory for collection artifacts.
    """

    def __init__(self, root_dir: str) -> None:
        if not isinstance(root_dir, str):
            raise TypeError(
                f"root_dir must be str, got {type(root_dir).__name__}"
            )
        self.root_dir = root_dir
        self._ensure_dirs()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        """Create root directory if it does not exist."""
        os.makedirs(self.root_dir, exist_ok=True)

    def _records_path(self) -> str:
        return os.path.join(self.root_dir, "records.jsonl")

    def _consents_path(self) -> str:
        return os.path.join(self.root_dir, "consents.jsonl")

    def _pii_evidence_path(self) -> str:
        return os.path.join(self.root_dir, "pii_evidence.jsonl")

    def _manifest_path(self) -> str:
        return os.path.join(self.root_dir, "session_manifest.json")

    @staticmethod
    def _serialize_enum(value) -> any:
        """Serialize enums using .value, pass through other types."""
        if hasattr(value, "value"):
            return value.value
        return value

    def _serialize_record(self, record: CanonicalRecord) -> dict:
        """Serialize a CanonicalRecord for restricted internal storage.

        source_text_raw is intentionally omitted.
        """
        data = {}
        for field_name in dataclasses.fields(record):
            value = getattr(record, field_name.name)
            if field_name.name == "source_text_raw":
                # Never persist raw source text in collection store
                continue
            data[field_name.name] = self._serialize_enum(value)
        return data

    @staticmethod
    def _serialize_consent(artifact) -> dict:
        """Serialize a ConsentArtifact."""
        return {
            "review_id": artifact.review_id,
            "consent_status": artifact.consent_status,
            "collected_at": artifact.collected_at,
            "method": artifact.method,
            "evidence_ref": artifact.evidence_ref,
        }

    @staticmethod
    def _serialize_pii_evidence(summary) -> dict:
        """Serialize a PIIEvidenceSummary (no matched values)."""
        return {
            "evidence_id": summary.evidence_id,
            "review_id": summary.review_id,
            "categories": list(summary.categories),
            "match_count": summary.match_count,
            "redaction_status": summary.redaction_status,
            "created_at": summary.created_at,
        }

    def _read_existing_review_ids(self) -> set:
        """Read review_ids already present in records.jsonl."""
        path = self._records_path()
        if not os.path.exists(path):
            return set()
        ids = set()
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "review_id" in obj:
                        ids.add(obj["review_id"])
                except json.JSONDecodeError:
                    continue
        return ids

    def _read_manifest(self) -> Optional[dict]:
        """Read current session manifest, if it exists."""
        path = self._manifest_path()
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write_manifest_atomic(self, manifest: dict) -> None:
        """Write session manifest with atomic replace."""
        path = self._manifest_path()
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)

    def _append_jsonl(self, path: str, obj: dict) -> None:
        """Append a single JSON object as a line to a JSONL file."""
        with open(path, "a", encoding="utf-8") as fh:
            line = json.dumps(obj, ensure_ascii=False)
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_session(
        self,
        started_at: str,
        collection_method: str,
    ) -> CollectionSession:
        """Start a new collection session.

        Parameters
        ----------
        started_at : str
            ISO 8601 timestamp of session start.
        collection_method : str
            How the collection was conducted.

        Returns
        -------
        CollectionSession
            The new session descriptor.
        """
        if not isinstance(started_at, str):
            raise TypeError(
                f"started_at must be str, got {type(started_at).__name__}"
            )
        if not isinstance(collection_method, str):
            raise TypeError(
                f"collection_method must be str, "
                f"got {type(collection_method).__name__}"
            )
        if not started_at.strip():
            raise ValueError("started_at must be a non-empty string.")
        if not collection_method.strip():
            raise ValueError(
                "collection_method must be a non-empty string."
            )

        session_id = "session-" + uuid.uuid4().hex

        # Write initial manifest atomically
        manifest = {
            "session_id": session_id,
            "started_at": started_at,
            "collection_method": collection_method,
            "record_count": 0,
        }
        self._write_manifest_atomic(manifest)

        return CollectionSession(
            session_id=session_id,
            started_at=started_at,
            collection_method=collection_method,
        )

    def append_intake(
        self,
        session: CollectionSession,
        result: HumanWrittenIntakeResult,
    ) -> None:
        """Append an intake result to the collection store.

        Parameters
        ----------
        session : CollectionSession
            The active collection session.
        result : HumanWrittenIntakeResult
            The intake result to persist.

        Raises
        ------
        TypeError
            If session or result has the wrong type.
        ValueError
            If the record fails eligibility checks, source_text_raw is
            present, review_id already exists, or cross-artifact
            consistency fails.
        """
        if not isinstance(session, CollectionSession):
            raise TypeError(
                f"session must be CollectionSession, "
                f"got {type(session).__name__}"
            )
        if not isinstance(result, HumanWrittenIntakeResult):
            raise TypeError(
                f"result must be HumanWrittenIntakeResult, "
                f"got {type(result).__name__}"
            )

        record = result.record
        consent = result.consent_artifact
        pii_evidence = result.pii_evidence

        # --- source_text_raw guard -------------------------------------------
        if record.source_text_raw is not None:
            raise ValueError(
                "Cannot append record with source_text_raw. "
                "C2.2 collection path requires retain_source_text_raw=False."
            )

        # --- eligibility checks ----------------------------------------------
        if record.source_type != SourceType.HUMAN_WRITTEN_RMC:
            raise ValueError(
                f"Record source_type must be HUMAN_WRITTEN_RMC, "
                f"got {record.source_type}"
            )
        if record.annotation_status != AnnotationStatus.UNANNOTATED:
            raise ValueError(
                f"Record annotation_status must be UNANNOTATED, "
                f"got {record.annotation_status}"
            )
        if record.consent_status != "CONSENTED":
            raise ValueError(
                f"Record consent_status must be 'CONSENTED', "
                f"got {record.consent_status!r}"
            )
        if not record.review_text or not record.review_text.strip():
            raise ValueError("Record review_text must be non-empty.")
        if record.dataset_version is not None:
            raise ValueError(
                "Record dataset_version must be None for collection."
            )
        if record.split_membership is not None:
            raise ValueError(
                "Record split_membership must be None for collection."
            )
        if record.split_group_id is not None:
            raise ValueError(
                "Record split_group_id must be None for collection."
            )
        # All six task labels must remain None
        for label_name in (
            "spam", "deception", "toxicity",
            "advertising", "off_topic", "pii",
        ):
            if getattr(record, label_name) is not None:
                raise ValueError(
                    f"Record {label_name} must be None for collection, "
                    f"got {getattr(record, label_name)!r}"
                )
        if record.language_mix is not None:
            raise ValueError(
                "Record language_mix must be None for collection."
            )
        if record.college_category is not None:
            raise ValueError(
                "Record college_category must be None for collection."
            )

        # --- cross-artifact consistency --------------------------------------
        if consent.review_id != record.review_id:
            raise ValueError(
                f"ConsentArtifact.review_id ({consent.review_id}) does not "
                f"match record.review_id ({record.review_id})."
            )

        if pii_evidence is not None:
            if pii_evidence.review_id != record.review_id:
                raise ValueError(
                    f"PIIEvidenceSummary.review_id ({pii_evidence.review_id}) "
                    f"does not match record.review_id ({record.review_id})."
                )
            if record.pii_evidence_id != pii_evidence.evidence_id:
                raise ValueError(
                    f"Record pii_evidence_id ({record.pii_evidence_id}) does "
                    f"not match PIIEvidenceSummary.evidence_id "
                    f"({pii_evidence.evidence_id})."
                )

        # --- duplicate review_id protection ----------------------------------
        existing_ids = self._read_existing_review_ids()
        if record.review_id in existing_ids:
            raise ValueError(
                f"review_id {record.review_id} already exists in "
                f"collection store."
            )

        # --- serialize -------------------------------------------------------
        record_data = self._serialize_record(record)
        consent_data = self._serialize_consent(consent)

        # --- writes ----------------------------------------------------------
        # Append records
        self._append_jsonl(self._records_path(), record_data)
        # Append consent
        self._append_jsonl(self._consents_path(), consent_data)
        # Append PII evidence if present
        if pii_evidence is not None:
            pii_data = self._serialize_pii_evidence(pii_evidence)
            self._append_jsonl(self._pii_evidence_path(), pii_data)

        # --- update manifest atomically (increment after all writes) ---------
        manifest = self._read_manifest()
        if manifest is None:
            raise RuntimeError(
                "No session manifest found. "
                "Call start_session before append_intake."
            )
        if manifest.get("session_id") != session.session_id:
            raise ValueError(
                f"Session mismatch: manifest session_id "
                f"({manifest.get('session_id')}) does not match "
                f"provided session ({session.session_id})."
            )
        manifest["record_count"] = manifest.get("record_count", 0) + 1
        self._write_manifest_atomic(manifest)
