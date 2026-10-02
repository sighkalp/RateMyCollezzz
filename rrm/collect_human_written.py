"""Genuine Human-Written RMC collection CLI.

This module provides a minimal local command-line interface for collecting
genuine Human-Written RMC reviews from consenting contributors through
the existing C2.1 intake adapter.

Architectural note:
    RRM UNDERSTANDS. TRUST DECIDES.

    This module is a collection interface only.  It does not annotate,
    label, moderate, or make trust decisions.

Usage:
    python -m rrm.collect_human_written [--root-dir PATH]

Public API:
    run_collection_cli
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

from rrm.corpus.annotation import (
    AnnotationSubmissionStore,
    ControlledProtocolTruthStore,
    DispositionRegister,
    ReannotationRegister,
)
from rrm.corpus.collection_store import (
    HumanWrittenCollectionStore,
)
from rrm.corpus.intake import collect_human_written_review
from rrm.corpus.workflow import CorpusWorkflow


# ---------------------------------------------------------------------------
# Consent and PII warning text
# ---------------------------------------------------------------------------

_CONSENT_STATEMENT = """\
I voluntarily agree that the review text I submit may be used for \
research and development of the RateMyCollezzz review-risk model. \
I understand that my real name is not required and that I should not \
include private personal information.\
"""

_PII_WARNING = """\
Do not include private personal information such as personal phone \
numbers, personal email addresses, home addresses, IDs, passwords, or \
other sensitive personal information.\
"""

_CONFIRM_PROMPT = "\nDo you consent? Type YES to confirm: "
_REVIEW_PROMPT = "\nEnter your review: "
_SUCCESS_TEMPLATE = (
    "\nThank you. Your review has been recorded with review_id: {review_id}"
)
_DECLINE_TEMPLATE = (
    "\nCollection declined. No data was collected or stored. Goodbye."
)


# ---------------------------------------------------------------------------
# Collection runner
# ---------------------------------------------------------------------------

def _get_default_root_dir() -> str:
    """Return the default collection storage root directory."""
    return os.path.join("local_research_data", "gate_c", "human_written")


def _get_collection_method() -> str:
    """Return the collection method identifier for this CLI."""
    return "local_cli"


def run_collection_cli(root_dir: str | None = None) -> None:
    """Run a single-review collection session via the CLI.

    Parameters
    ----------
    root_dir : str or None
        Root directory for collection artifacts.  Defaults to
        ``local_research_data/gate_c/human_written``.
    """
    if root_dir is None:
        root_dir = _get_default_root_dir()

    # --- Show consent statement ---------------------------------------------
    print(_CONSENT_STATEMENT)
    consent_input = input(_CONFIRM_PROMPT).strip()

    if consent_input.upper() != "YES":
        print(_DECLINE_TEMPLATE)
        return

    # --- Show PII warning ---------------------------------------------------
    print(_PII_WARNING)

    # --- Enter review -------------------------------------------------------
    submitted_text = input(_REVIEW_PROMPT).strip()

    if not submitted_text:
        print("\nEmpty review. No data was collected. Goodbye.")
        return

    # --- Collect ------------------------------------------------------------
    collected_at = datetime.now(timezone.utc).isoformat()
    collection_method = _get_collection_method()

    workflow = CorpusWorkflow(
        AnnotationSubmissionStore(),
        ControlledProtocolTruthStore(),
        DispositionRegister(),
        ReannotationRegister(),
    )
    store = HumanWrittenCollectionStore(root_dir)

    # Start session
    session = store.start_session(
        started_at=collected_at,
        collection_method=collection_method,
    )

    # Run intake (retain_source_text_raw=False enforced)
    result = collect_human_written_review(
        workflow=workflow,
        submitted_text=submitted_text,
        consent_granted=True,
        collected_at=collected_at,
        collection_method=collection_method,
        retain_source_text_raw=False,
    )

    # Persist to restricted local store
    store.append_intake(session, result)

    # --- Report -------------------------------------------------------------
    print(_SUCCESS_TEMPLATE.format(review_id=result.record.review_id))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Entry point for the collection CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "RateMyCollezzz Human-Written RMC collection CLI. "
            "Collects one genuine review with explicit consent."
        )
    )
    parser.add_argument(
        "--output-dir",
        "--root-dir",
        dest="output_dir",
        default=None,
        help=(
            "Root directory for collection artifacts "
            "(default: local_research_data/gate_c/human_written)"
        ),
    )
    args = parser.parse_args(argv)

    try:
        run_collection_cli(root_dir=args.output_dir)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
