from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANNOTATION_PATH = ROOT / "pilot_annotations.jsonl"

BINARY_LABELS = (
    "spam",
    "toxicity",
    "advertising",
    "off_topic",
    "pii",
)

CATEGORICAL_LABELS = (
    "language_mix",
    "college_category",
)


def load_annotations() -> list[dict]:
    if not ANNOTATION_PATH.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {ANNOTATION_PATH}"
        )

    return [
        json.loads(line)
        for line in ANNOTATION_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def cohen_kappa(values_a: list, values_b: list) -> float | None:
    if len(values_a) != len(values_b):
        raise ValueError("Annotation lists must have equal length.")

    n = len(values_a)

    if n == 0:
        return None

    observed = sum(
        a == b
        for a, b in zip(values_a, values_b)
    ) / n

    categories = set(values_a) | set(values_b)

    count_a = Counter(values_a)
    count_b = Counter(values_b)

    expected = sum(
        (count_a[value] / n) * (count_b[value] / n)
        for value in categories
    )

    if expected == 1:
        # Both annotators assigned exactly one identical category
        # to every record. Agreement is perfect, but standard
        # kappa denominator becomes zero.
        return None

    return (observed - expected) / (1 - expected)


def analyze_field(
    field: str,
    pairs: list[tuple[dict, dict]],
) -> dict:
    values_a = [a[field] for a, _ in pairs]
    values_b = [b[field] for _, b in pairs]

    agreements = [
        a == b
        for a, b in zip(values_a, values_b)
    ]

    disagreement_ids = [
        a["review_id"]
        for (a, b), agreed in zip(pairs, agreements)
        if not agreed
    ]

    raw_agreement = (
        sum(agreements) / len(agreements)
        if agreements
        else 0
    )

    return {
        "field": field,
        "count": len(pairs),
        "agreement_count": sum(agreements),
        "disagreement_count": len(disagreement_ids),
        "raw_agreement": raw_agreement,
        "kappa": cohen_kappa(values_a, values_b),
        "a_distribution": Counter(values_a),
        "b_distribution": Counter(values_b),
        "disagreement_ids": disagreement_ids,
    }


def print_result(result: dict) -> None:
    print()
    print("=" * 68)
    print(result["field"].upper())
    print("=" * 68)

    print("Compared records:", result["count"])
    print("Agreements:", result["agreement_count"])
    print("Disagreements:", result["disagreement_count"])
    print(
        "Raw agreement:",
        f"{result['raw_agreement'] * 100:.1f}%"
    )

    kappa = result["kappa"]

    if kappa is None:
        print(
            "Cohen's kappa: undefined "
            "(insufficient category variation)"
        )
    else:
        print("Cohen's kappa:", f"{kappa:.3f}")

    print("Annotator A distribution:")
    print(" ", dict(result["a_distribution"]))

    print("Annotator B distribution:")
    print(" ", dict(result["b_distribution"]))

    if result["disagreement_ids"]:
        print("Disagreement review IDs:")

        for review_id in result["disagreement_ids"]:
            print(" ", review_id)
    else:
        print("Disagreement review IDs: none")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze agreement between two RMC annotators."
    )

    parser.add_argument(
        "--annotator-a",
        default="A1",
    )

    parser.add_argument(
        "--annotator-b",
        default="A2",
    )

    args = parser.parse_args()

    rows = load_annotations()

    by_annotator: dict[str, dict[str, dict]] = {}

    for row in rows:
        annotator = row["annotator_id"]
        review_id = row["review_id"]

        by_annotator.setdefault(
            annotator,
            {},
        )[review_id] = row

    annotations_a = by_annotator.get(
        args.annotator_a,
        {},
    )

    annotations_b = by_annotator.get(
        args.annotator_b,
        {},
    )

    overlap_ids = sorted(
        set(annotations_a)
        & set(annotations_b)
    )

    if not overlap_ids:
        print("No overlapping reviews found.")
        return 1

    pairs = [
        (
            annotations_a[review_id],
            annotations_b[review_id],
        )
        for review_id in overlap_ids
    ]

    print("=== RMC PILOT AGREEMENT ANALYSIS ===")
    print("Annotator A:", args.annotator_a)
    print("Annotator B:", args.annotator_b)
    print("A annotations:", len(annotations_a))
    print("B annotations:", len(annotations_b))
    print("Overlapping reviews:", len(overlap_ids))

    print()
    print(
        "Deception is intentionally excluded because blind text "
        "annotation cannot establish deception ground truth."
    )

    for field in BINARY_LABELS:
        print_result(
            analyze_field(
                field,
                pairs,
            )
        )

    for field in CATEGORICAL_LABELS:
        print_result(
            analyze_field(
                field,
                pairs,
            )
        )

    print()
    print("=" * 68)
    print("SUMMARY")
    print("=" * 68)

    all_fields = (
        *BINARY_LABELS,
        *CATEGORICAL_LABELS,
    )

    for field in all_fields:
        result = analyze_field(
            field,
            pairs,
        )

        kappa = (
            "N/A"
            if result["kappa"] is None
            else f"{result['kappa']:.3f}"
        )

        print(
            f"{field:18} "
            f"agreement={result['raw_agreement'] * 100:5.1f}% "
            f"kappa={kappa:>6} "
            f"disagreements={result['disagreement_count']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())