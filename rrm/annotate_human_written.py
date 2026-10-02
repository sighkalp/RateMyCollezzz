"""CLI for Human-Written independent A/B annotation workspace.

Architectural note:
    RRM UNDERSTANDS. TRUST DECIDES.

    This CLI is annotation tooling only.  It does NOT moderate,
    ban, delete, or make trust decisions.

Usage:
    python -m rrm.annotate_human_written \
        --root-dir <collection_root> \
        --slot A|B \
        --annotator-id <id> \
        [--review-id <id>]
"""

from __future__ import annotations

import argparse
import sys

from rrm.corpus.annotation_workspace import (
    HumanWrittenAnnotationWorkspace,
    AnnotationTask,
)
from rrm.corpus.models import (
    CollegeCategory,
    LanguageMix,
)


_DEFAULT_ROOT_DIR = "local_research_data/gate_c/human_written"


# ---------------------------------------------------------------------------
# Interactive prompt helpers
# ---------------------------------------------------------------------------

def _prompt_binary(label: str) -> int:
    """Prompt for a 0/1 binary annotation.

    Accepts only '0' or '1'.  Reprompts on invalid input.
    """
    while True:
        raw = input(f"  {label} (0=No, 1=Yes): ").strip()
        if raw in ("0", "1"):
            return int(raw)
        print("  Please enter 0 or 1.")


def _prompt_language_mix() -> LanguageMix:
    """Prompt for language mix using canonical textual names.

    Accepts case-insensitive canonical names after strip.
    """
    canonical = {
        "ENGLISH": LanguageMix.ENGLISH,
        "HINGLISH": LanguageMix.HINGLISH,
        "ROMAN_HINDI": LanguageMix.ROMAN_HINDI,
        "OTHER": LanguageMix.OTHER,
        "MIXED_OTHER": LanguageMix.MIXED_OTHER,
    }
    print("\n  Language mix (enter canonical name):")
    for name in canonical:
        print(f"    {name}")
    while True:
        raw = input("  Language mix: ").strip().upper()
        if raw in canonical:
            return canonical[raw]
        print("  Invalid. Enter one of the canonical names above.")


def _prompt_college_category() -> object:
    """Prompt for college category using canonical textual names + NONE."""
    canonical = {
        "NONE": None,
        "ACADEMICS": CollegeCategory.ACADEMICS,
        "FACULTY": CollegeCategory.FACULTY,
        "PLACEMENTS": CollegeCategory.PLACEMENTS,
        "HOSTEL": CollegeCategory.HOSTEL,
        "INFRASTRUCTURE": CollegeCategory.INFRASTRUCTURE,
        "FEES": CollegeCategory.FEES,
        "ADMINISTRATION": CollegeCategory.ADMINISTRATION,
        "CAMPUS_LIFE": CollegeCategory.CAMPUS_LIFE,
        "ADMISSIONS": CollegeCategory.ADMISSIONS,
        "MULTI_TOPIC": CollegeCategory.MULTI_TOPIC,
    }
    print("\n  College category (enter canonical name or NONE):")
    for name in canonical:
        print(f"    {name}")
    while True:
        raw = input("  College category: ").strip().upper()
        if raw in canonical:
            return canonical[raw]
        print("  Invalid. Enter one of the canonical names above.")


# ---------------------------------------------------------------------------
# Core annotation flow
# ---------------------------------------------------------------------------

def _print_privacy_notice() -> None:
    """Print the annotator privacy instruction."""
    print(
        "\nUse only your assigned pseudonymous annotator ID.\n"
        "Do not enter your real name, email, phone number, student ID,\n"
        "or other personal identity information.\n"
    )


def _print_task(task: AnnotationTask) -> None:
    """Display only the blind task data."""
    print(f"Review ID: {task.review_id}")
    print(f"Review text: {task.review_text}")


def _collect_annotation() -> dict:
    """Interactively collect the seven C2.3 annotation choices.

    Exact input sequence:
        spam (0/1)
        toxicity (0/1)
        advertising (0/1)
        off_topic (0/1)
        pii (0/1)
        language_mix (canonical name)
        college_category (canonical name or NONE)
    """
    print("\nAnnotate each dimension (enter 0 or 1):")
    choices: dict = {}
    choices["spam"] = _prompt_binary("Spam")
    choices["toxicity"] = _prompt_binary("Toxic/abusive")
    choices["advertising"] = _prompt_binary("Advertising")
    choices["off_topic"] = _prompt_binary("Off-topic")
    choices["pii"] = _prompt_binary("Contains PII")
    choices["language_mix"] = _prompt_language_mix()
    choices["college_category"] = _prompt_college_category()
    return choices


def _run_annotation(
    workspace: HumanWrittenAnnotationWorkspace,
    slot: str,
    annotator_id: str,
    review_id: str | None,
) -> int:
    """Run one annotation submission.

    One review per invocation.  No repeat/continue loop.

    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """
    # Select task
    if review_id is not None:
        try:
            task = workspace.get_task(review_id, slot)
        except ValueError as exc:
            print(f"No eligible review: {exc}", file=sys.stderr)
            return 1
    else:
        task = workspace.select_next(slot)
        if task is None:
            print(f"No eligible review for slot {slot}.")
            return 0

    # Display privacy notice, task, and deception policy
    _print_privacy_notice()
    print(f"Review ID: {task.review_id}")
    print(f"Review text: {task.review_text}")
    print("deception = UNKNOWN (-1) by Human-Written policy")

    # Collect choices
    choices = _collect_annotation()

    # Submit through workspace
    try:
        updated = workspace.submit_annotation(
            slot=slot,
            review_id=task.review_id,
            annotator_id=annotator_id,
            choices=choices,
        )
    except (ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Annotation saved.\nreview_id: {updated.review_id}\nslot: {slot}")
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Entry point for the Human-Written annotation CLI.

    Parameters
    ----------
    argv : list, optional
        Command-line arguments.  Defaults to sys.argv[1:].

    Returns
    -------
    int
        Exit code (0 = success).
    """
    parser = argparse.ArgumentParser(
        prog="annotate-human-written",
        description=(
            "Human-Written independent A/B annotation tooling. "
            "One annotation per invocation."
        ),
    )
    parser.add_argument(
        "--root-dir",
        default=_DEFAULT_ROOT_DIR,
        help=(
            "Collection root directory "
            f"(default: {_DEFAULT_ROOT_DIR})"
        ),
    )
    parser.add_argument(
        "--slot",
        required=True,
        choices=["A", "B"],
        help="Annotation slot: A or B.",
    )
    parser.add_argument(
        "--annotator-id",
        required=True,
        help="Pseudonymous annotator ID.",
    )
    parser.add_argument(
        "--review-id",
        default=None,
        help="Specific review to annotate (optional).",
    )

    args = parser.parse_args(argv)

    # Blank annotator ID rejection
    if not args.annotator_id.strip():
        print(
            "Error: --annotator-id must not be blank.",
            file=sys.stderr,
        )
        return 1

    try:
        workspace = HumanWrittenAnnotationWorkspace(args.root_dir)
    except (ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return _run_annotation(
        workspace=workspace,
        slot=args.slot,
        annotator_id=args.annotator_id,
        review_id=args.review_id,
    )


if __name__ == "__main__":
    sys.exit(main())
