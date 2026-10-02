"""Human-Written independent A/B annotation workspace.

Architectural note:
    RRM UNDERSTANDS. TRUST DECIDES.

    This module provides workspace infrastructure for independent
    double annotation of Human-Written RMC records.  It does NOT
    hide, delete, ban, moderate, or make trust decisions.
    It stores research annotations only.

Public API:
    HumanWrittenAnnotationWorkspace
    AnnotationTask
"""

from __future__ import annotations

import dataclasses
import json
import os
from datetime import datetime, timezone
from typing import Dict, Optional

from rrm.corpus.annotation import (
    AnnotationSubmissionStore,
    ControlledProtocolTruthStore,
    DispositionRegister,
    ReannotationRegister,
)
from rrm.corpus.models import (
    AnnotationStatus,
    AnnotationSubmission,
    CanonicalRecord,
    CollegeCategory,
    LanguageMix,
    SourceType,
)
from rrm.corpus.validation import validate_annotation_submission
from rrm.corpus.workflow import CorpusWorkflow


@dataclasses.dataclass(frozen=True)
class AnnotationTask:
    """Blind annotation task for a single annotator.

    Exposes only what is needed for annotation.  Does NOT expose
    contributor identity, other annotator data, or PII evidence
    internals.

    Attributes
    ----------
    review_id : str
        The record identifier.
    review_text : str
        The safe review text for annotation.
    slot : str
        "A" or "B".
    """

    review_id: str
    review_text: str
    slot: str


class HumanWrittenAnnotationWorkspace:
    """Independent A/B annotation workspace for Human-Written RMC.

    Uses SEQUENTIAL-BUT-BLIND DOUBLE ANNOTATION per the frozen
    canonical workflow: A must submit before B becomes eligible,
    but B never sees A's answers, identity, or metadata.

    Parameters
    ----------
    root_dir : str
        Root directory containing ``records.jsonl`` and annotation
        operational artifacts.

    Notes
    -----
    This workspace is annotation infrastructure only.  It does NOT
    perform adjudication, finalization, moderation, or trust decisions.
    """

    ANNOTATION_GUIDE_VERSION = "1.0"
    _HUMAN_WRITTEN_DECEPTION = -1

    def __init__(self, root_dir: str) -> None:
        if not isinstance(root_dir, str) or not root_dir.strip():
            raise TypeError("root_dir must be a non-empty string")
        self.root_dir = root_dir
        self._records_path = os.path.join(root_dir, "records.jsonl")
        self._submissions_path = os.path.join(
            root_dir, "annotation_submissions.jsonl"
        )
        self._states_path = os.path.join(
            root_dir, "annotation_record_states.jsonl"
        )

        # Fresh workflow components
        self._submission_store = AnnotationSubmissionStore()
        self._truth_store = ControlledProtocolTruthStore()
        self._disposition_register = DispositionRegister()
        self._reannotation_register = ReannotationRegister()
        self._workflow = CorpusWorkflow(
            self._submission_store,
            self._truth_store,
            self._disposition_register,
            self._reannotation_register,
        )
        self._records: Dict[str, CanonicalRecord] = {}

        self._load()

    # ------------------------------------------------------------------
    # Loading / reconstruction — fail closed
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Reconstruct workspace state from disk."""
        self._load_base_records()
        self._apply_latest_states()
        self._replay_submissions()

    def _load_base_records(self) -> None:
        """Load and validate records from records.jsonl. Fail closed."""
        if not os.path.exists(self._records_path):
            return
        seen_ids: Dict[str, int] = {}  # review_id -> lineno
        with open(self._records_path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                # Must be valid JSON object
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    raise ValueError(
                        f"Malformed JSON in {self._records_path} "
                        f"line {lineno}"
                    )
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"Non-object JSON in {self._records_path} "
                        f"line {lineno}"
                    )
                # Must have non-blank review_id
                rid = obj.get("review_id")
                if not isinstance(rid, str) or not rid.strip():
                    raise ValueError(
                        f"Missing or blank review_id in "
                        f"{self._records_path} line {lineno}"
                    )
                # No duplicate review_ids
                if rid in seen_ids:
                    raise ValueError(
                        f"Duplicate review_id '{rid}' in "
                        f"{self._records_path} lines "
                        f"{seen_ids[rid]} and {lineno}"
                    )
                seen_ids[rid] = lineno

                record = self._deserialize_record(obj)
                self._validate_source_record(record)
                self._records[rid] = record
                self._workflow.register_record(record)

    def _apply_latest_states(self) -> None:
        """Apply latest annotation_record_states.jsonl snapshot per review_id.

        Rejects malformed state, unknown review_ids, and privacy violations.
        """
        if not os.path.exists(self._states_path):
            return
        latest: Dict[str, tuple] = {}  # review_id -> (lineno, obj)
        with open(self._states_path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    raise ValueError(
                        f"Malformed JSON in {self._states_path} "
                        f"line {lineno}"
                    )
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"Non-object JSON in {self._states_path} "
                        f"line {lineno}"
                    )
                rid = obj.get("review_id")
                if not isinstance(rid, str) or not rid.strip():
                    raise ValueError(
                        f"Missing or blank review_id in "
                        f"{self._states_path} line {lineno}"
                    )
                if rid not in self._records:
                    raise ValueError(
                        f"State row references unknown review_id "
                        f"'{rid}' in {self._states_path} line {lineno}"
                    )
                # Privacy: source_text_raw must not be set in state
                if "source_text_raw" in obj and obj["source_text_raw"] is not None:
                    raise ValueError(
                        f"State row for '{rid}' in {self._states_path} "
                        f"line {lineno} has non-None source_text_raw"
                    )
                latest[rid] = (lineno, obj)
        for rid, (_, state_obj) in latest.items():
            updated = self._deserialize_record(state_obj)
            # Re-validate the deserialized state against Human-Written contract
            self._validate_source_record(updated)
            self._records[rid] = updated
            self._workflow._update_record(updated)

    def _replay_submissions(self) -> None:
        """Replay annotation_submissions.jsonl into the submission store.

        Fail closed: validate every row, reject duplicates beyond 2,
        reject same annotator for both slots.
        """
        if not os.path.exists(self._submissions_path):
            return
        seen_slots: Dict[str, int] = {}  # review_id -> submission count
        with open(self._submissions_path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    raise ValueError(
                        f"Malformed JSON in {self._submissions_path} "
                        f"line {lineno}"
                    )
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"Non-object JSON in {self._submissions_path} "
                        f"line {lineno}"
                    )
                rid = obj.get("review_id")
                if not isinstance(rid, str) or not rid.strip():
                    raise ValueError(
                        f"Missing or blank review_id in "
                        f"{self._submissions_path} line {lineno}"
                    )
                if rid not in self._records:
                    raise ValueError(
                        f"Submission references unknown review_id "
                        f"'{rid}' in {self._submissions_path} line {lineno}"
                    )

                sub = self._deserialize_submission(obj)

                # Validate via canonical validator
                ok, errors = validate_annotation_submission(sub)
                if not ok:
                    raise ValueError(
                        f"Invalid submission in "
                        f"{self._submissions_path} line {lineno}: "
                        + "; ".join(errors)
                    )

                # deception must be -1 for Human-Written
                if sub.deception != -1:
                    raise ValueError(
                        f"Submission deception={sub.deception} in "
                        f"{self._submissions_path} line {lineno}; "
                        f"Human-Written requires deception=-1"
                    )

                # guide version must match workspace constant
                if sub.annotation_guide_version != self.ANNOTATION_GUIDE_VERSION:
                    raise ValueError(
                        f"Submission guide_version "
                        f"'{sub.annotation_guide_version}' in "
                        f"{self._submissions_path} line {lineno}; "
                        f"expected '{self.ANNOTATION_GUIDE_VERSION}'"
                    )

                # Reject more than 2 submissions for the same review
                count = seen_slots.get(rid, 0)
                if count >= 2:
                    raise ValueError(
                        f"Third submission for '{rid}' in "
                        f"{self._submissions_path} line {lineno}; "
                        f"max 2 allowed"
                    )

                # Reject same annotator for both slots
                if count == 1:
                    first = self._submission_store.get_for_review(rid)
                    if first and first[0].annotator_id == sub.annotator_id:
                        raise ValueError(
                            f"Same annotator '{sub.annotator_id}' for "
                            f"both A and B on '{rid}' in "
                            f"{self._submissions_path} line {lineno}"
                        )

                seen_slots[rid] = count + 1
                self._submission_store.submit(sub)

    def _validate_source_record(self, record: CanonicalRecord) -> None:
        """Validate Human-Written Gate-C source record requirements.

        Rejects any record that does not meet the strict C2.2 contract.
        """
        if record.source_type != SourceType.HUMAN_WRITTEN_RMC:
            raise ValueError(
                f"Record {record.review_id}: source_type must be "
                f"HUMAN_WRITTEN_RMC, got {record.source_type}"
            )
        if record.consent_status != "CONSENTED":
            raise ValueError(
                f"Record {record.review_id}: consent_status must be "
                f"'CONSENTED', got {record.consent_status!r}"
            )
        if not record.review_text or not record.review_text.strip():
            raise ValueError(
                f"Record {record.review_id}: review_text must be "
                f"non-empty"
            )
        if record.source_text_raw is not None:
            raise ValueError(
                f"Record {record.review_id}: source_text_raw must "
                f"be None"
            )
        if record.dataset_version is not None:
            raise ValueError(
                f"Record {record.review_id}: dataset_version must "
                f"be None"
            )
        if record.split_membership is not None:
            raise ValueError(
                f"Record {record.review_id}: split_membership must "
                f"be None"
            )
        if record.split_group_id is not None:
            raise ValueError(
                f"Record {record.review_id}: split_group_id must "
                f"be None"
            )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_record(record: CanonicalRecord) -> dict:
        """Serialize CanonicalRecord to a JSON-safe dict.

        source_text_raw is intentionally omitted for privacy.
        """
        result: dict = {}
        for field in dataclasses.fields(record):
            value = getattr(record, field.name)
            if field.name == "source_text_raw":
                continue
            if hasattr(value, "value"):
                result[field.name] = value.value
            elif isinstance(value, list):
                result[field.name] = list(value)
            else:
                result[field.name] = value
        return result

    @staticmethod
    def _serialize_submission(sub: AnnotationSubmission) -> dict:
        """Serialize AnnotationSubmission to a JSON-safe dict.

        Uses real canonical fields.  Enums serialize by .value.
        """
        result: dict = {}
        for field in dataclasses.fields(sub):
            val = getattr(sub, field.name)
            if hasattr(val, "value"):
                result[field.name] = val.value
            else:
                result[field.name] = val
        return result

    @staticmethod
    def _deserialize_record(obj: dict) -> CanonicalRecord:
        """Deserialize a dict to CanonicalRecord."""
        kwargs: dict = {}
        for field in dataclasses.fields(CanonicalRecord):
            if field.name not in obj:
                continue
            value = obj[field.name]
            if value is None:
                kwargs[field.name] = None
            elif field.name == "source_type" and isinstance(value, str):
                kwargs[field.name] = SourceType(value)
            elif field.name == "annotation_status" and isinstance(value, str):
                kwargs[field.name] = AnnotationStatus(value)
            elif field.name == "language_mix" and isinstance(value, str):
                kwargs[field.name] = LanguageMix(value)
            elif field.name == "college_category" and isinstance(value, str):
                kwargs[field.name] = CollegeCategory(value)
            else:
                kwargs[field.name] = value
        return CanonicalRecord(**kwargs)

    @staticmethod
    def _deserialize_submission(obj: dict) -> AnnotationSubmission:
        """Deserialize a dict to AnnotationSubmission."""
        kwargs: dict = {}
        for field in dataclasses.fields(AnnotationSubmission):
            if field.name not in obj:
                continue
            value = obj[field.name]
            if value is None:
                kwargs[field.name] = None
            elif field.name == "language_mix" and isinstance(value, str):
                kwargs[field.name] = LanguageMix(value)
            elif field.name == "college_category" and isinstance(value, str):
                kwargs[field.name] = CollegeCategory(value)
            else:
                kwargs[field.name] = value
        return AnnotationSubmission(**kwargs)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _append_jsonl(self, path: str, obj: dict) -> None:
        """Append a JSON object as a line to a JSONL file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _persist_submission(self, sub: AnnotationSubmission) -> None:
        """Append an annotation submission to the submissions file."""
        obj = self._serialize_submission(sub)
        self._append_jsonl(self._submissions_path, obj)

    def _persist_record_state(self, record: CanonicalRecord) -> None:
        """Append a record state snapshot to the states file.

        source_text_raw is never persisted.
        """
        obj = self._serialize_record(record)
        self._append_jsonl(self._states_path, obj)

    # ------------------------------------------------------------------
    # Public API — selection
    # ------------------------------------------------------------------

    def select_next(self, slot: str) -> Optional[AnnotationTask]:
        """Select the next review eligible for the requested slot.

        Slot A eligible:
            - valid Human-Written record
            - no A submission
            - record not FINAL/EXCLUDED

        Slot B eligible:
            - A submission exists
            - B submission does not exist
            - record not FINAL/EXCLUDED

        Selection follows stable records.jsonl insertion order.

        Parameters
        ----------
        slot : str
            "A" or "B".

        Returns
        -------
        AnnotationTask or None
            The next eligible task, or None if no records are eligible.
        """
        slot = slot.upper()
        if slot not in ("A", "B"):
            raise ValueError(f"slot must be 'A' or 'B', got {slot!r}")

        for review_id, record in self._records.items():
            if record.source_type != SourceType.HUMAN_WRITTEN_RMC:
                continue

            status = self._workflow.get_workflow_state(record)
            if status in (AnnotationStatus.EXCLUDED, AnnotationStatus.FINAL):
                continue

            if slot == "A":
                a_sub = self._submission_store.get_a(review_id)
                if a_sub is None:
                    return AnnotationTask(
                        review_id=review_id,
                        review_text=record.review_text or "",
                        slot="A",
                    )
            else:
                a_sub = self._submission_store.get_a(review_id)
                b_sub = self._submission_store.get_b(review_id)
                if a_sub is not None and b_sub is None:
                    return AnnotationTask(
                        review_id=review_id,
                        review_text=record.review_text or "",
                        slot="B",
                    )

        return None

    def get_task(self, review_id: str, slot: str) -> AnnotationTask:
        """Get the exact AnnotationTask for a known review_id.

        Enforces slot eligibility.  Returns only review_id, review_text,
        and slot — no other metadata.

        Parameters
        ----------
        review_id : str
            The record to annotate.
        slot : str
            "A" or "B".

        Returns
        -------
        AnnotationTask
            The blind task for this review and slot.

        Raises
        ------
        ValueError
            If review_id is unknown or ineligible for the slot.
        """
        slot = slot.upper()
        if slot not in ("A", "B"):
            raise ValueError(f"slot must be 'A' or 'B', got {slot!r}")

        record = self._records.get(review_id)
        if record is None:
            raise ValueError(f"Unknown review_id: {review_id}")

        if record.source_type != SourceType.HUMAN_WRITTEN_RMC:
            raise ValueError(
                f"Record {review_id} is not HUMAN_WRITTEN_RMC"
            )

        status = self._workflow.get_workflow_state(record)
        if status in (AnnotationStatus.EXCLUDED, AnnotationStatus.FINAL):
            raise ValueError(
                f"Record {review_id} is {status.value}"
            )

        if slot == "A":
            a_sub = self._submission_store.get_a(review_id)
            if a_sub is not None:
                raise ValueError(
                    f"Annotator A already submitted for {review_id}"
                )
        else:
            a_sub = self._submission_store.get_a(review_id)
            b_sub = self._submission_store.get_b(review_id)
            if a_sub is None:
                raise ValueError(
                    f"Annotator A has not submitted for {review_id}; "
                    f"B is not yet eligible"
                )
            if b_sub is not None:
                raise ValueError(
                    f"Annotator B already submitted for {review_id}"
                )

        return AnnotationTask(
            review_id=review_id,
            review_text=record.review_text or "",
            slot=slot,
        )

    # ------------------------------------------------------------------
    # Public API — submission
    # ------------------------------------------------------------------

    def submit_annotation(
        self,
        slot: str,
        review_id: str,
        annotator_id: str,
        choices: dict,
    ) -> CanonicalRecord:
        """Submit a canonical annotation for a Human-Written review.

        Submits through the canonical workflow APIs, persists the
        submission and resulting record state, and returns the
        updated CanonicalRecord.

        Parameters
        ----------
        slot : str
            "A" or "B".
        review_id : str
            The record to annotate.
        annotator_id : str
            Pseudonymous annotator identifier.  Must not be the same
            as the other slot's annotator for the same review.
        choices : dict
            Annotation choices.  Required keys: spam, toxicity,
            advertising, off_topic, pii, language_mix.
            Optional: college_category.
            deception is fixed to -1 for Human-Written and must NOT
            be provided in choices.

        Returns
        -------
        CanonicalRecord
            The updated record after workflow state transition.

        Raises
        ------
        ValueError
            If the submission is invalid (duplicate slot, same annotator
            for both slots, ineligible record, missing keys, etc.).
        TypeError
            If annotator_id is not a non-empty string.
        """
        slot = slot.upper()
        if slot not in ("A", "B"):
            raise ValueError(f"slot must be 'A' or 'B', got {slot!r}")
        if not isinstance(annotator_id, str) or not annotator_id.strip():
            raise TypeError("annotator_id must be a non-empty string")

        # Validate required keys — exact C2.3 annotation dimensions
        required_keys = {
            "spam", "toxicity", "advertising", "off_topic", "pii",
            "language_mix",
        }
        missing = required_keys - set(choices.keys())
        if missing:
            raise ValueError(f"Missing required choices: {missing}")
        if "deception" in choices:
            raise ValueError(
                "deception must NOT be provided in choices; "
                "it is fixed to -1 for Human-Written records"
            )

        # Strict binary label validation: exact int 0 or 1, no bool, no str
        _BINARY_LABELS = ("spam", "toxicity", "advertising", "off_topic", "pii")
        for label in _BINARY_LABELS:
            value = choices[label]
            if type(value) is not int or value not in (0, 1):
                raise ValueError(
                    f"{label} must be int 0 or 1, got {value!r} "
                    f"(type {type(value).__name__})"
                )

        # Enum type validation
        language_mix_val = choices["language_mix"]
        if not isinstance(language_mix_val, LanguageMix):
            raise ValueError(
                f"language_mix must be a LanguageMix enum, "
                f"got {language_mix_val!r} (type "
                f"{type(language_mix_val).__name__})"
            )

        cc_val = choices.get("college_category")
        if cc_val is not None and not isinstance(cc_val, CollegeCategory):
            raise ValueError(
                f"college_category must be CollegeCategory or None, "
                f"got {cc_val!r} (type {type(cc_val).__name__})"
            )

        record = self._records.get(review_id)
        if record is None:
            raise ValueError(f"Unknown review_id: {review_id}")

        if record.source_type != SourceType.HUMAN_WRITTEN_RMC:
            raise ValueError(
                f"Record {review_id} is not HUMAN_WRITTEN_RMC"
            )

        status = self._workflow.get_workflow_state(record)
        if status in (AnnotationStatus.EXCLUDED, AnnotationStatus.FINAL):
            raise ValueError(
                f"Record {review_id} is {status.value}"
            )

        # Build the canonical submission
        submitted_at = datetime.now(timezone.utc).isoformat()

        sub = AnnotationSubmission(
            review_id=review_id,
            annotator_id=annotator_id,
            annotation_guide_version=self.ANNOTATION_GUIDE_VERSION,
            spam=choices["spam"],
            deception=self._HUMAN_WRITTEN_DECEPTION,
            toxicity=choices["toxicity"],
            advertising=choices["advertising"],
            off_topic=choices["off_topic"],
            pii=choices["pii"],
            language_mix=language_mix_val,
            college_category=cc_val,
            annotation_note=None,
            submitted_at=submitted_at,
        )

        # Validate via canonical validator
        ok, errors = validate_annotation_submission(sub)
        if not ok:
            raise ValueError(
                "Invalid annotation submission: " + "; ".join(errors)
            )

        # Enforce slot uniqueness and annotator independence
        if slot == "A":
            existing = self._submission_store.get_a(review_id)
            if existing is not None:
                raise ValueError(
                    f"Annotator A already submitted for {review_id}"
                )
            updated = self._workflow.submit_annotation_a(record, sub)
        else:
            existing = self._submission_store.get_b(review_id)
            if existing is not None:
                raise ValueError(
                    f"Annotator B already submitted for {review_id}"
                )
            a_sub = self._submission_store.get_a(review_id)
            if a_sub is None:
                raise ValueError(
                    f"Annotator A has not submitted for {review_id}"
                )
            if a_sub.annotator_id == annotator_id:
                raise ValueError(
                    f"Annotator {annotator_id} is already Annotator A "
                    f"for {review_id}; A and B must be independent"
                )
            updated = self._workflow.submit_annotation_b(record, sub)

        # Update internal state
        self._records[review_id] = updated

        # Persist submission and record state
        self._persist_submission(sub)
        self._persist_record_state(updated)

        return updated

    # ------------------------------------------------------------------
    # Public API — queries
    # ------------------------------------------------------------------

    def get_record(self, review_id: str) -> Optional[CanonicalRecord]:
        """Return the current CanonicalRecord for a review, or None."""
        return self._records.get(review_id)

    def get_eligible_count(self, slot: str) -> int:
        """Return the number of records eligible for the given slot.

        Parameters
        ----------
        slot : str
            "A" or "B".

        Returns
        -------
        int
            Count of eligible records.
        """
        slot = slot.upper()
        if slot not in ("A", "B"):
            raise ValueError(f"slot must be 'A' or 'B', got {slot!r}")
        count = 0
        for review_id, record in self._records.items():
            if record.source_type != SourceType.HUMAN_WRITTEN_RMC:
                continue
            status = self._workflow.get_workflow_state(record)
            if status in (AnnotationStatus.EXCLUDED, AnnotationStatus.FINAL):
                continue
            if slot == "A":
                a_sub = self._submission_store.get_a(review_id)
                if a_sub is None:
                    count += 1
            else:
                a_sub = self._submission_store.get_a(review_id)
                b_sub = self._submission_store.get_b(review_id)
                if a_sub is not None and b_sub is None:
                    count += 1
        return count
