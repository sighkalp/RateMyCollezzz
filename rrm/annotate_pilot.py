from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "rmc_pilot_v0.1.jsonl"
ANNOTATION_PATH = ROOT / "pilot_annotations.jsonl"

LANGUAGES = {
    "1": "ENGLISH",
    "2": "HINGLISH",
    "3": "ROMAN_HINDI",
    "4": "OTHER",
    "5": "MIXED_OTHER",
}

TOPICS = {
    "1": "ACADEMICS",
    "2": "FACULTY",
    "3": "PLACEMENTS",
    "4": "HOSTEL",
    "5": "INFRASTRUCTURE",
    "6": "FEES",
    "7": "ADMINISTRATION",
    "8": "CAMPUS_LIFE",
    "9": "ADMISSIONS",
    "10": "MULTI_TOPIC",
    "11": "OTHER",
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def ask_binary(name: str) -> int:
    while True:
        value = input(f"{name} [0=no, 1=yes]: ").strip()

        if value in {"0", "1"}:
            return int(value)

        print("Enter only 0 or 1.")


def ask_choice(title: str, choices: dict[str, str]) -> str:
    print()
    print(title)

    for key, value in choices.items():
        print(f"  {key}. {value}")

    while True:
        value = input("Choice: ").strip()

        if value in choices:
            return choices[value]

        print("Choose one of the listed numbers.")


def save_annotation(annotation: dict) -> None:
    with ANNOTATION_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                annotation,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Blind human annotation tool for the RMC pilot."
    )

    parser.add_argument(
        "--annotator",
        required=True,
        help="Anonymous annotator ID, for example A1.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of new records to annotate.",
    )

    args = parser.parse_args()

    records = load_jsonl(DATASET_PATH)
    annotations = load_jsonl(ANNOTATION_PATH)

    completed = {
        (row["annotator_id"], row["review_id"])
        for row in annotations
    }

    pending = [
        row
        for row in records
        if (args.annotator, row["review_id"]) not in completed
    ]

    if args.limit is not None:
        pending = pending[: args.limit]

    print("=== RMC Blind Annotation ===")
    print("Annotator:", args.annotator)
    print("Available records:", len(records))
    print("Already annotated:", len(records) - len([
        row for row in records
        if (args.annotator, row["review_id"]) not in completed
    ]))
    print("This session:", len(pending))

    if not pending:
        print("Nothing left to annotate.")
        return 0

    for index, record in enumerate(pending, start=1):
        print()
        print("=" * 72)
        print(f"SESSION RECORD {index}/{len(pending)}")
        print("Review ID:", record["review_id"])
        print()
        print(record["review_text"])
        print()

        language_mix = ask_choice(
            "LANGUAGE",
            LANGUAGES,
        )

        college_category = ask_choice(
            "PRIMARY TOPIC",
            TOPICS,
        )

        print()
        print("RISK LABELS")
        print("Use the annotation guide. Evaluate each independently.")

        spam = ask_binary("spam")
        toxicity = ask_binary("toxicity")
        advertising = ask_binary("advertising")
        off_topic = ask_binary("off_topic")
        pii = ask_binary("pii")

        print()
        note = input(
            "Optional annotation note (Enter to skip): "
        ).strip()

        annotation = {
            "review_id": record["review_id"],
            "annotator_id": args.annotator,
            "language_mix": language_mix,
            "college_category": college_category,
            "spam": spam,

            # Blind text annotation cannot establish deception.
            "deception": -1,
            "deception_basis": "UNKNOWN_TEXT_ONLY",

            "toxicity": toxicity,
            "advertising": advertising,
            "off_topic": off_topic,
            "pii": pii,
            "annotation_note": note or None,
            "annotation_guide_version": "RMC-v0.1",
            "annotated_at": datetime.now(timezone.utc).isoformat(),
        }

        save_annotation(annotation)

        print("Saved:", record["review_id"])

    print()
    print("Annotation session complete.")
    print("Saved to:", ANNOTATION_PATH)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())