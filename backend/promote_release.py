"""Promote a validated candidate artifact set into the local release store.

Promotion is a dry run unless ``--confirm`` is supplied. A confirmed promotion
copies and validates immutable release artifacts, then atomically switches the
active pointer. Existing releases are never overwritten or deleted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from release_store import ReleaseStore, ReleaseStoreError

BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = BACKEND_DIR.parent / "dataset"
DEFAULT_RELEASE_STORE = BACKEND_DIR / "artifacts"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a candidate and optionally promote it as an immutable, "
            "atomically activated release. The default is a safe dry run."
        )
    )
    parser.add_argument("candidate_dir", help="Candidate artifact directory")
    parser.add_argument(
        "--release-store",
        default=str(DEFAULT_RELEASE_STORE),
        help=f"Release store directory (default: {DEFAULT_RELEASE_STORE})",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_DIR),
        help=f"Dataset root used to verify metadata paths (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually create the release and activate it",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    store = ReleaseStore(args.release_store)
    try:
        plan = store.promote(
            args.candidate_dir,
            confirm=args.confirm,
            dataset_root=args.dataset,
        )
    except ReleaseStoreError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("=" * 72)
    print("RELEASE PROMOTION" + ("" if args.confirm else " (DRY RUN)"))
    print("=" * 72)
    print(f"Candidate:          {plan.source_dir}")
    print(f"Release store:      {store.root}")
    print(f"Release ID:         {plan.release_id}")
    print(f"Release directory:  {plan.release_dir}")
    print(f"Indexed images:     {plan.indexed_images}")
    print(f"Index SHA-256:      {plan.index_sha256}")
    print(f"Metadata SHA-256:   {plan.metadata_sha256}")
    print(f"Previous active:    {plan.previous_release_id or '(none)'}")
    print(f"Release exists:     {'yes' if plan.release_exists else 'no'}")
    print()
    if not args.confirm:
        print("DRY RUN: no release directory or active pointer was changed.")
        print("Re-run with --confirm to create and activate this release.")
    elif plan.already_active:
        print("Release is already active; no changes were made.")
    elif plan.activated:
        print("Release activated atomically. Restart the API to load it.")
    else:
        print("No activation was performed.")
    print("Previous releases are preserved for rollback.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
