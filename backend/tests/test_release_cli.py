"""CLI safety tests for promotion and rollback."""

from __future__ import annotations

import promote_release
import rollback_release
from release_store import ReleaseStore
from tests.artifact_fixtures import build_candidate, make_dataset


def test_promotion_cli_defaults_to_dry_run(tmp_path, capsys):
    dataset = make_dataset(tmp_path / "dataset")
    candidate, _ = build_candidate(tmp_path / "candidate")
    store_dir = tmp_path / "store"

    code = promote_release.main(
        [
            str(candidate),
            "--release-store",
            str(store_dir),
            "--dataset",
            str(dataset),
        ]
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "DRY RUN" in output
    assert "rel-" in output
    assert not store_dir.exists()


def test_promotion_cli_requires_confirm_to_activate(tmp_path, capsys):
    dataset = make_dataset(tmp_path / "dataset")
    candidate, _ = build_candidate(tmp_path / "candidate")
    store_dir = tmp_path / "store"

    code = promote_release.main(
        [
            str(candidate),
            "--release-store",
            str(store_dir),
            "--dataset",
            str(dataset),
            "--confirm",
        ]
    )

    output = capsys.readouterr().out
    store = ReleaseStore(store_dir)
    assert code == 0
    assert "activated" in output.lower()
    assert store.active_release_id() is not None


def test_rollback_cli_lists_and_dry_runs(tmp_path, capsys):
    dataset = make_dataset(tmp_path / "dataset")
    candidate, _ = build_candidate(tmp_path / "candidate")
    store = ReleaseStore(tmp_path / "store")
    plan = store.promote(candidate, confirm=True, dataset_root=dataset)

    list_code = rollback_release.main(
        ["--release-store", str(store.root), "--list"]
    )
    list_output = capsys.readouterr().out
    assert list_code == 0
    assert plan.release_id in list_output
    assert "Active release:" in list_output

    rollback_code = rollback_release.main(
        [plan.release_id, "--release-store", str(store.root)]
    )
    rollback_output = capsys.readouterr().out
    assert rollback_code == 0
    assert "DRY RUN" in rollback_output
    assert "already active" in rollback_output.lower()
    assert store.active_release_id() == plan.release_id
