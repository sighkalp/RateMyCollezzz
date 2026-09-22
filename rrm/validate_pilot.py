from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


DATASET_PATH = Path(__file__).with_name("rmc_pilot_v0.1.jsonl")

REQUIRED_FIELDS = {
    "review_id",
    "review_text",
    "language_mix",
    "college_category",
    "source_type",
    "spam",
    "deception",
    "toxicity",
    "advertising",
    "off_topic",
    "pii",
    "annotation_status",
    "annotator_count",
    "agreement_status",
    "dataset_split",
    "dataset_version",
    "notes",
}

LANGUAGES = {
    "ENGLISH",
    "HINGLISH",
    "ROMAN_HINDI",
    "OTHER",
    "MIXED_OTHER",
}

COLLEGE_CATEGORIES = {
    "ACADEMICS",
    "FACULTY",
    "PLACEMENTS",
    "HOSTEL",
    "INFRASTRUCTURE",
    "FEES",
    "ADMINISTRATION",
    "CAMPUS_LIFE",
    "ADMISSIONS",
    "MULTI_TOPIC",
    "OTHER",
}

SOURCE_TYPES = {
    "PUBLIC_DATASET",
    "CONTROLLED_RESEARCH",
    "HUMAN_WRITTEN_RESEARCH",
    "SYNTHETIC_RESEARCH",
    "PLATFORM_REVIEW",
}

ANNOTATION_STATUSES = {
    "UNANNOTATED",
    "IN_PROGRESS",
    "ANNOTATED",
    "ADJUDICATION_REQUIRED",
    "FINAL",
    "EXCLUDED",
}

AGREEMENT_STATUSES = {
    "NOT_APPLICABLE",
    "SINGLE_ANNOTATOR",
    "AGREEMENT",
    "PARTIAL_AGREEMENT",
    "DISAGREEMENT",
    "ADJUDICATED",
}

DATASET_SPLITS = {
    None,
    "TRAIN",
    "VALIDATION",
    "TEST",
}

BINARY_LABELS = {
    "spam",
    "toxicity",
    "advertising",
    "off_topic",
    "pii",
}


def load_records() -> list[dict]:
    records = []

    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Line {line_number}: invalid JSON: {exc}"
                ) from exc

            records.append(record)

    return records


def validate_record(record: dict, line_number: int) -> list[str]:
    errors: list[str] = []

    missing = REQUIRED_FIELDS - record.keys()

    if missing:
        errors.append(
            f"Line {line_number}: missing fields: {sorted(missing)}"
        )
        return errors

    extra = record.keys() - REQUIRED_FIELDS

    if extra:
        errors.append(
            f"Line {line_number}: unexpected fields: {sorted(extra)}"
        )

    if not isinstance(record["review_id"], str) or not record["review_id"].strip():
        errors.append(f"Line {line_number}: invalid review_id")

    if not isinstance(record["review_text"], str) or not record["review_text"].strip():
        errors.append(f"Line {line_number}: review_text must be non-empty")

    if record["language_mix"] not in LANGUAGES:
        errors.append(
            f"Line {line_number}: invalid language_mix "
            f"{record['language_mix']!r}"
        )

    if (
        record["college_category"] is not None
        and record["college_category"] not in COLLEGE_CATEGORIES
    ):
        errors.append(
            f"Line {line_number}: invalid college_category "
            f"{record['college_category']!r}"
        )

    if record["source_type"] not in SOURCE_TYPES:
        errors.append(
            f"Line {line_number}: invalid source_type "
            f"{record['source_type']!r}"
        )

    for label in BINARY_LABELS:
        if record[label] not in {0, 1}:
            errors.append(
                f"Line {line_number}: {label} must be 0 or 1"
            )

    if record["deception"] not in {-1, 0, 1}:
        errors.append(
            f"Line {line_number}: deception must be -1, 0, or 1"
        )

    if record["annotation_status"] not in ANNOTATION_STATUSES:
        errors.append(
            f"Line {line_number}: invalid annotation_status "
            f"{record['annotation_status']!r}"
        )

    if (
        not isinstance(record["annotator_count"], int)
        or record["annotator_count"] < 0
    ):
        errors.append(
            f"Line {line_number}: annotator_count must be >= 0"
        )

    if record["agreement_status"] not in AGREEMENT_STATUSES:
        errors.append(
            f"Line {line_number}: invalid agreement_status "
            f"{record['agreement_status']!r}"
        )

    if record["dataset_split"] not in DATASET_SPLITS:
        errors.append(
            f"Line {line_number}: invalid dataset_split "
            f"{record['dataset_split']!r}"
        )

    if (
        not isinstance(record["dataset_version"], str)
        or not record["dataset_version"].strip()
    ):
        errors.append(
            f"Line {line_number}: invalid dataset_version"
        )

    if record["notes"] is not None and not isinstance(record["notes"], str):
        errors.append(
            f"Line {line_number}: notes must be string or null"
        )

    return errors


def print_summary(records: list[dict]) -> None:
    print()
    print("=== RMC PILOT SUMMARY ===")
    print("Records:", len(records))

    print()
    print("Languages:")
    for key, value in Counter(
        record["language_mix"] for record in records
    ).most_common():
        print(f"  {key}: {value}")

    print()
    print("Sources:")
    for key, value in Counter(
        record["source_type"] for record in records
    ).most_common():
        print(f"  {key}: {value}")

    print()
    print("Positive labels:")

    for label in sorted(BINARY_LABELS):
        positives = sum(record[label] == 1 for record in records)
        print(f"  {label}: {positives}")

    deception = Counter(record["deception"] for record in records)

    print()
    print("Deception:")
    print("  UNKNOWN (-1):", deception[-1])
    print("  NON-DECEPTIVE (0):", deception[0])
    print("  DECEPTIVE (1):", deception[1])


def main() -> int:
    if not DATASET_PATH.exists():
        print(f"FAIL: dataset does not exist: {DATASET_PATH}")
        return 1

    records = load_records()

    errors: list[str] = []

    seen_ids: set[str] = set()

    for line_number, record in enumerate(records, start=1):
        errors.extend(validate_record(record, line_number))

        review_id = record.get("review_id")

        if review_id in seen_ids:
            errors.append(
                f"Line {line_number}: duplicate review_id {review_id!r}"
            )

        if isinstance(review_id, str):
            seen_ids.add(review_id)

    if errors:
        print("=== RMC PILOT VALIDATION: FAIL ===")

        for error in errors:
            print("-", error)

        return 1

    print("=== RMC PILOT VALIDATION: PASS ===")

    print_summary(records)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())