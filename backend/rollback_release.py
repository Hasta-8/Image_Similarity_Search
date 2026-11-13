"""Rollback the active release pointer to a previously promoted release.

Rollback changes only the active pointer. Release directories are immutable and
are never modified or deleted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from release_store import ReleaseStore, ReleaseStoreError

BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_RELEASE_STORE = BACKEND_DIR / "artifacts"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Atomically switch the active release to an existing release. "
            "The default is a safe dry run."
        )
    )
    parser.add_argument(
        "release_id",
        nargs="?",
        help="Existing release ID to activate",
    )
    parser.add_argument(
        "--release-store",
        default=str(DEFAULT_RELEASE_STORE),
        help=f"Release store directory (default: {DEFAULT_RELEASE_STORE})",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually switch the active pointer",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List releases and exit",
    )
    return parser


def _print_releases(store: ReleaseStore) -> int:
    try:
        active = store.active_release_id()
        listings = store.list_releases()
    except ReleaseStoreError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Release store: {store.root}")
    print(f"Active release: {active or '(none)'}")
    if not listings:
        print("No releases found.")
        return 0
    print()
    print(f"{'ACTIVE':<7} {'VALID':<6} {'RELEASE ID':<30} {'IMAGES':>7}  PROMOTED")
    for listing in listings:
        marker = "*" if listing.release_id == active else ""
        if listing.info is not None:
            print(
                f"{marker:<7} {'yes':<6} {listing.release_id:<30} "
                f"{listing.info.indexed_images:>7}  {listing.info.promoted_at_utc or ''}"
            )
        else:
            print(f"{marker:<7} {'NO':<6} {listing.release_id:<30} {0:>7}  INVALID")
            for error in listing.errors:
                print(f"        - {error}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    store = ReleaseStore(args.release_store)

    if args.list:
        return _print_releases(store)
    if not args.release_id:
        parser.error("release_id is required unless --list is used")

    try:
        plan = store.rollback(args.release_id, confirm=args.confirm)
    except ReleaseStoreError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("=" * 72)
    print("RELEASE ROLLBACK" + ("" if args.confirm else " (DRY RUN)"))
    print("=" * 72)
    print(f"Release store:      {store.root}")
    print(f"Target release:     {plan.target_release_id}")
    print(f"Target directory:   {plan.target_dir}")
    print(f"Previous active:    {plan.previous_release_id or '(none)'}")
    print()
    if not args.confirm:
        print("DRY RUN: active pointer was not changed.")
        if plan.already_active:
            print("Target release is already active; no changes are required.")
        else:
            print("Re-run with --confirm to activate the target release.")
    elif plan.already_active:
        print("Target release is already active; no changes were made.")
    elif plan.activated:
        print("Active pointer rolled back atomically. Restart the API to load it.")
    else:
        print("No activation was performed.")
    print("Release artifacts were not modified or deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
