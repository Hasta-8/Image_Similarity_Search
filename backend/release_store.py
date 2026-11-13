"""Immutable artifact releases with atomic activation and rollback.

The release store is intentionally local-filesystem oriented. Artifacts are
copied into an immutable ``releases/<release_id>/`` directory, validated, and
then activated by atomically replacing a small pointer file. Runtime loading
only trusts a release after the pointer and every artifact have been validated.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from artifact_manifest import (
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    METADATA_FILENAME,
    load_manifest,
    semantic_manifest_sha256,
    utc_now_iso,
    write_manifest,
)
from artifact_paths import DatasetPathError, DatasetPathResolver
from artifact_validation import (
    ArtifactValidationError,
    ValidationResult,
    validate_candidate_artifacts,
    validate_release_artifacts,
)
from dataset_fingerprint import sha256_file
from faiss_index import load_faiss_index, load_metadata

RELEASE_ID_PREFIX = "rel-"
RELEASE_ID_RE = re.compile(r"^rel-[0-9a-f]{24}$")
POINTER_SCHEMA_VERSION = "1.0"
RELEASES_DIRNAME = "releases"
ACTIVE_POINTER_FILENAME = "ACTIVE.json"
LOCK_FILENAME = ".release.lock"
RELEASE_TOOL_VERSION = "3.1"
RELEASE_ID_DOMAIN = "imgsimisearch:release:v2"

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None
    import msvcrt


class ReleaseStoreError(RuntimeError):
    """Raised when a release operation cannot proceed safely."""


@dataclass
class ReleaseInfo:
    release_id: str
    release_dir: Path
    manifest_sha256: str
    index_sha256: str
    metadata_sha256: str
    semantic_manifest_sha256: str
    indexed_images: int
    promoted_at_utc: Optional[str]
    previous_release_id: Optional[str]


@dataclass
class PromotionPlan:
    source_dir: Path
    release_id: str
    release_dir: Path
    index_sha256: str
    metadata_sha256: str
    semantic_manifest_sha256: str
    source_manifest_sha256: str
    indexed_images: int
    previous_release_id: Optional[str]
    release_exists: bool
    dry_run: bool = True
    activated: bool = False
    already_active: bool = False


@dataclass
class RollbackPlan:
    target_release_id: str
    target_dir: Path
    previous_release_id: Optional[str]
    dry_run: bool = True
    activated: bool = False
    already_active: bool = False


@dataclass
class RuntimeArtifacts:
    release_id: str
    release_dir: Path
    index: Any
    metadata: Dict[int, str]
    manifest: Mapping[str, Any]
    semantic_manifest_sha256: str
    index_sha256: str
    metadata_sha256: str
    path_resolver: DatasetPathResolver


@dataclass
class ReleaseListing:
    release_id: str
    release_dir: Path
    valid: bool
    errors: List[str] = field(default_factory=list)
    info: Optional[ReleaseInfo] = None


class _FileLock:
    def __init__(self, path: Path, *, exclusive: bool = True):
        self.path = path
        self.exclusive = exclusive
        self._handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "a+b")
        if fcntl is not None:
            fcntl.flock(
                self._handle.fileno(),
                fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH,
            )
        else:  # pragma: no cover - Windows fallback
            self._handle.seek(0)
            mode = msvcrt.LK_LOCK if self.exclusive else msvcrt.LK_RLCK
            msvcrt.locking(self._handle.fileno(), mode, 1)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if fcntl is not None:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            else:  # pragma: no cover - Windows fallback
                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self._handle.close()
            self._handle = None
        return False


def release_id_for(
    index_sha256: str, metadata_sha256: str, semantic_manifest_sha256: str
) -> str:
    inputs = (index_sha256, metadata_sha256, semantic_manifest_sha256)
    if not all(isinstance(value, str) for value in inputs):
        raise ReleaseStoreError("Release ID inputs must be SHA-256 strings")
    digest = hashlib.sha256(
        (
            f"{RELEASE_ID_DOMAIN}\0{index_sha256}\0{metadata_sha256}\0"
            f"{semantic_manifest_sha256}"
        ).encode("ascii")
    ).hexdigest()
    return f"{RELEASE_ID_PREFIX}{digest[:24]}"


def validate_release_id(release_id: str) -> str:
    if not isinstance(release_id, str) or not RELEASE_ID_RE.match(release_id):
        raise ReleaseStoreError(f"Invalid release ID: {release_id!r}")
    return release_id


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseStoreError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    except ReleaseStoreError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseStoreError(f"Could not read {path}: {exc}") from exc


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _copy_file(source: Path, destination: Path) -> None:
    shutil.copyfile(source, destination)
    with open(destination, "rb+") as handle:
        os.fsync(handle.fileno())


def _manifest_sha256(path: Path) -> str:
    return sha256_file(path)


class ReleaseStore:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root).expanduser().absolute()
        self.releases_dir = self.root / RELEASES_DIRNAME
        self.pointer_path = self.root / ACTIVE_POINTER_FILENAME
        self.lock_path = self.root / LOCK_FILENAME

    def _lock(self, *, exclusive: bool = True) -> _FileLock:
        return _FileLock(self.lock_path, exclusive=exclusive)

    def _release_dir(self, release_id: str) -> Path:
        validate_release_id(release_id)
        return self.releases_dir / release_id

    def _ensure_layout(self) -> None:
        self.releases_dir.mkdir(parents=True, exist_ok=True)

    def _load_pointer(self) -> Optional[dict]:
        if not self.pointer_path.is_file():
            return None
        payload = _read_json(self.pointer_path)
        if not isinstance(payload, dict):
            raise ReleaseStoreError("Active release pointer must be a JSON object")
        if payload.get("schema_version") != POINTER_SCHEMA_VERSION:
            raise ReleaseStoreError("Unsupported active release pointer schema_version")
        validate_release_id(payload.get("release_id"))
        for key in (
            "index_sha256",
            "metadata_sha256",
            "semantic_manifest_sha256",
            "source_manifest_sha256",
        ):
            value = payload.get(key)
            if not isinstance(value, str) or not re.match(r"^[0-9a-f]{64}$", value):
                raise ReleaseStoreError(f"Active pointer has invalid {key}")
        previous = payload.get("previous_release_id")
        if previous is not None:
            validate_release_id(previous)
        return payload

    def active_release_id(self) -> Optional[str]:
        payload = self._load_pointer()
        return payload["release_id"] if payload else None

    def _release_info(
        self, release_id: str, directory: Path, validation: ValidationResult
    ) -> ReleaseInfo:
        manifest_path = directory / MANIFEST_FILENAME
        manifest = load_manifest(manifest_path)
        release_manifest = manifest.get("release", {})
        return ReleaseInfo(
            release_id=release_id,
            release_dir=directory,
            manifest_sha256=_manifest_sha256(manifest_path),
            index_sha256=validation.details["index_sha256"],
            metadata_sha256=validation.details["metadata_sha256"],
            semantic_manifest_sha256=validation.details["semantic_manifest_sha256"],
            indexed_images=int(validation.details["index_ntotal"] or 0),
            promoted_at_utc=(
                release_manifest.get("promoted_at_utc")
                if isinstance(release_manifest, dict)
                else None
            ),
            previous_release_id=(
                release_manifest.get("previous_release_id")
                if isinstance(release_manifest, dict)
                else None
            ),
        )

    def _validated_release(
        self,
        release_id: str,
        expected_index_sha256: Optional[str] = None,
        expected_metadata_sha256: Optional[str] = None,
        expected_embedding_dim: Optional[int] = None,
    ) -> tuple[Path, ValidationResult]:
        directory = self._release_dir(release_id)
        if directory.is_symlink():
            raise ReleaseStoreError(f"Release directory must not be a symlink: {directory}")
        if not directory.is_dir():
            raise ReleaseStoreError(f"Release does not exist: {release_id}")

        validation = validate_release_artifacts(
            directory, expected_embedding_dim=expected_embedding_dim
        )
        if not validation.valid:
            raise ReleaseStoreError(
                f"Release {release_id} is invalid:\n"
                + "\n".join(f"- {error}" for error in validation.errors)
            )

        manifest = load_manifest(directory / MANIFEST_FILENAME)
        manifest_release_id = manifest.get("release", {}).get("release_id")
        if manifest_release_id != release_id:
            raise ReleaseStoreError(
                f"Release manifest ID {manifest_release_id!r} does not match "
                f"directory ID {release_id!r}"
            )
        if expected_index_sha256 is not None and (
            validation.details["index_sha256"] != expected_index_sha256
        ):
            raise ReleaseStoreError("Release index checksum does not match pointer")
        if expected_metadata_sha256 is not None and (
            validation.details["metadata_sha256"] != expected_metadata_sha256
        ):
            raise ReleaseStoreError("Release metadata checksum does not match pointer")
        return directory, validation

    def list_releases(self) -> List[ReleaseListing]:
        if not self.releases_dir.is_dir():
            return []
        listings: List[ReleaseListing] = []
        for entry in sorted(self.releases_dir.iterdir(), key=lambda item: item.name):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if not RELEASE_ID_RE.match(entry.name):
                continue
            validation = validate_release_artifacts(entry)
            if validation.valid:
                listings.append(
                    ReleaseListing(
                        release_id=entry.name,
                        release_dir=entry,
                        valid=True,
                        info=self._release_info(entry.name, entry, validation),
                    )
                )
            else:
                listings.append(
                    ReleaseListing(
                        release_id=entry.name,
                        release_dir=entry,
                        valid=False,
                        errors=list(validation.errors),
                    )
                )
        return listings

    def _validate_candidate_dataset_paths(
        self, source: Path, dataset_root: Optional[str | os.PathLike]
    ) -> None:
        if dataset_root is None:
            return
        try:
            resolver = DatasetPathResolver(dataset_root)
            metadata = load_metadata(str(source / METADATA_FILENAME))
            resolver.metadata_paths(metadata, must_exist=True)
        except DatasetPathError as exc:
            raise ReleaseStoreError(
                f"Candidate metadata path validation failed: {exc}"
            ) from exc

    def plan_promotion(
        self,
        candidate_dir: str | os.PathLike,
        *,
        dataset_root: Optional[str | os.PathLike] = None,
    ) -> PromotionPlan:
        source = Path(candidate_dir).expanduser().absolute()
        validation = validate_candidate_artifacts(source)
        if not validation.valid:
            raise ReleaseStoreError(
                "Candidate artifact validation failed:\n"
                + "\n".join(f"- {error}" for error in validation.errors)
            )
        self._validate_candidate_dataset_paths(source, dataset_root)

        manifest_path = source / MANIFEST_FILENAME
        source_manifest_sha = _manifest_sha256(manifest_path)
        candidate_manifest = load_manifest(manifest_path)
        semantic_sha = semantic_manifest_sha256(candidate_manifest)
        index_sha = validation.details["index_sha256"]
        metadata_sha = validation.details["metadata_sha256"]
        release_id = release_id_for(index_sha, metadata_sha, semantic_sha)
        release_dir = self._release_dir(release_id)
        release_exists = release_dir.exists()

        if release_exists:
            _, existing_validation = self._validated_release(
                release_id,
                expected_index_sha256=index_sha,
                expected_metadata_sha256=metadata_sha,
            )
            existing_semantic_sha = existing_validation.details.get(
                "semantic_manifest_sha256"
            )
            if existing_semantic_sha != semantic_sha:
                raise ReleaseStoreError(
                    "Candidate semantic manifest conflicts with existing release "
                    f"{release_id}: candidate={semantic_sha}, "
                    f"release={existing_semantic_sha}. Refusing to reuse the "
                    "existing release."
                )

        active = self._load_pointer()
        previous = active["release_id"] if active else None
        return PromotionPlan(
            source_dir=source,
            release_id=release_id,
            release_dir=release_dir,
            index_sha256=index_sha,
            metadata_sha256=metadata_sha,
            semantic_manifest_sha256=semantic_sha,
            source_manifest_sha256=source_manifest_sha,
            indexed_images=int(validation.details["index_ntotal"] or 0),
            previous_release_id=previous,
            release_exists=release_exists,
        )

    def _build_release_manifest(
        self,
        candidate_manifest: Mapping[str, Any],
        *,
        release_id: str,
        source_manifest_sha256: str,
        semantic_manifest_sha256: str,
        previous_release_id: Optional[str],
        promoted_at_utc: str,
    ) -> dict:
        manifest = copy.deepcopy(dict(candidate_manifest))
        manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
        manifest["artifact_role"] = "production"
        manifest["promotion_status"] = "promoted"
        manifest["release"] = {
            "release_id": release_id,
            "promoted_at_utc": promoted_at_utc,
            "source_manifest_sha256": source_manifest_sha256,
            "semantic_manifest_sha256": semantic_manifest_sha256,
            "previous_release_id": previous_release_id,
            "tool_version": RELEASE_TOOL_VERSION,
        }
        return manifest

    def promote(
        self,
        candidate_dir: str | os.PathLike,
        *,
        confirm: bool = False,
        dataset_root: Optional[str | os.PathLike] = None,
    ) -> PromotionPlan:
        if not confirm:
            return self.plan_promotion(candidate_dir, dataset_root=dataset_root)

        with self._lock(exclusive=True):
            plan = self.plan_promotion(candidate_dir, dataset_root=dataset_root)
            plan.dry_run = False
            active = self._load_pointer()
            if active and active["release_id"] == plan.release_id:
                plan.already_active = True
                plan.release_exists = True
                return plan

            self._ensure_layout()
            if not plan.release_dir.exists():
                staging = Path(
                    tempfile.mkdtemp(
                        prefix=f".{plan.release_id}.tmp-", dir=str(self.releases_dir)
                    )
                )
                try:
                    _copy_file(plan.source_dir / INDEX_FILENAME, staging / INDEX_FILENAME)
                    _copy_file(
                        plan.source_dir / METADATA_FILENAME, staging / METADATA_FILENAME
                    )
                    if sha256_file(staging / INDEX_FILENAME) != plan.index_sha256:
                        raise ReleaseStoreError("Copied index checksum changed")
                    if sha256_file(staging / METADATA_FILENAME) != plan.metadata_sha256:
                        raise ReleaseStoreError("Copied metadata checksum changed")

                    current_source_manifest_sha = _manifest_sha256(
                        plan.source_dir / MANIFEST_FILENAME
                    )
                    if current_source_manifest_sha != plan.source_manifest_sha256:
                        raise ReleaseStoreError(
                            "Candidate manifest changed during promotion"
                        )
                    candidate_manifest = load_manifest(plan.source_dir / MANIFEST_FILENAME)
                    release_manifest = self._build_release_manifest(
                        candidate_manifest,
                        release_id=plan.release_id,
                        source_manifest_sha256=plan.source_manifest_sha256,
                        semantic_manifest_sha256=plan.semantic_manifest_sha256,
                        previous_release_id=plan.previous_release_id,
                        promoted_at_utc=utc_now_iso(),
                    )
                    write_manifest(release_manifest, staging / MANIFEST_FILENAME)

                    validation = validate_release_artifacts(staging)
                    if not validation.valid:
                        raise ReleaseStoreError(
                            "Staged release validation failed:\n"
                            + "\n".join(f"- {error}" for error in validation.errors)
                        )
                    if plan.release_dir.exists():
                        raise ReleaseStoreError(
                            f"Release appeared during promotion: {plan.release_id}"
                        )
                    os.replace(staging, plan.release_dir)
                    _fsync_directory(self.releases_dir)
                except Exception:
                    shutil.rmtree(staging, ignore_errors=True)
                    raise
            else:
                self._validated_release(
                    plan.release_id,
                    expected_index_sha256=plan.index_sha256,
                    expected_metadata_sha256=plan.metadata_sha256,
                )

            release_manifest = load_manifest(plan.release_dir / MANIFEST_FILENAME)
            release_info = release_manifest.get("release", {})
            plan.source_manifest_sha256 = release_info["source_manifest_sha256"]
            plan.semantic_manifest_sha256 = release_info[
                "semantic_manifest_sha256"
            ]
            self._write_pointer(
                release_id=plan.release_id,
                index_sha256=plan.index_sha256,
                metadata_sha256=plan.metadata_sha256,
                semantic_manifest_sha256=plan.semantic_manifest_sha256,
                source_manifest_sha256=plan.source_manifest_sha256,
                previous_release_id=plan.previous_release_id,
                operation="promote",
            )
            plan.activated = True
            return plan

    def _write_pointer(
        self,
        *,
        release_id: str,
        index_sha256: str,
        metadata_sha256: str,
        semantic_manifest_sha256: str,
        source_manifest_sha256: str,
        previous_release_id: Optional[str],
        operation: str,
    ) -> None:
        payload = {
            "schema_version": POINTER_SCHEMA_VERSION,
            "release_id": release_id,
            "index_sha256": index_sha256,
            "metadata_sha256": metadata_sha256,
            "semantic_manifest_sha256": semantic_manifest_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "previous_release_id": previous_release_id,
            "activated_at_utc": utc_now_iso(),
            "operation": operation,
        }
        _atomic_write_json(self.pointer_path, payload)

    def plan_rollback(self, target_release_id: str) -> RollbackPlan:
        validate_release_id(target_release_id)
        directory, validation = self._validated_release(target_release_id)
        active = self._load_pointer()
        active_id = active["release_id"] if active else None
        return RollbackPlan(
            target_release_id=target_release_id,
            target_dir=directory,
            previous_release_id=active_id,
            dry_run=True,
            already_active=active_id == target_release_id,
        )

    def rollback(self, target_release_id: str, *, confirm: bool = False) -> RollbackPlan:
        if not confirm:
            return self.plan_rollback(target_release_id)

        with self._lock(exclusive=True):
            plan = self.plan_rollback(target_release_id)
            plan.dry_run = False
            if plan.already_active:
                return plan

            directory, validation = self._validated_release(target_release_id)
            manifest_path = directory / MANIFEST_FILENAME
            manifest = load_manifest(manifest_path)
            release_manifest = manifest.get("release", {})
            self._write_pointer(
                release_id=target_release_id,
                index_sha256=validation.details["index_sha256"],
                metadata_sha256=validation.details["metadata_sha256"],
                semantic_manifest_sha256=release_manifest[
                    "semantic_manifest_sha256"
                ],
                source_manifest_sha256=release_manifest["source_manifest_sha256"],
                previous_release_id=plan.previous_release_id,
                operation="rollback",
            )
            plan.activated = True
            return plan

    def load_active_release(
        self,
        dataset_root: str | os.PathLike,
        *,
        expected_embedding_dim: Optional[int] = None,
    ) -> RuntimeArtifacts:
        with self._lock(exclusive=False):
            pointer = self._load_pointer()
            if pointer is None:
                raise ReleaseStoreError("No active release pointer exists")
            release_id = pointer["release_id"]
            directory, validation = self._validated_release(
                release_id,
                expected_index_sha256=pointer["index_sha256"],
                expected_metadata_sha256=pointer["metadata_sha256"],
                expected_embedding_dim=expected_embedding_dim,
            )
            manifest = load_manifest(directory / MANIFEST_FILENAME)
            release_info = manifest.get("release", {})
            source_sha = release_info.get("source_manifest_sha256")
            if source_sha != pointer["source_manifest_sha256"]:
                raise ReleaseStoreError(
                    "Release source manifest checksum does not match active pointer"
                )
            semantic_sha = validation.details["semantic_manifest_sha256"]
            if release_info.get("semantic_manifest_sha256") != semantic_sha:
                raise ReleaseStoreError(
                    "Release semantic manifest checksum does not match its content"
                )
            if semantic_sha != pointer["semantic_manifest_sha256"]:
                raise ReleaseStoreError(
                    "Release semantic manifest checksum does not match active pointer"
                )

            metadata = load_metadata(str(directory / METADATA_FILENAME))
            try:
                resolver = DatasetPathResolver(dataset_root)
                resolver.metadata_paths(metadata, must_exist=True)
            except DatasetPathError as exc:
                raise ReleaseStoreError(f"Release metadata path validation failed: {exc}") from exc

            index = load_faiss_index(str(directory / INDEX_FILENAME))
            return RuntimeArtifacts(
                release_id=release_id,
                release_dir=directory,
                index=index,
                metadata=metadata,
                manifest=manifest,
                semantic_manifest_sha256=semantic_sha,
                index_sha256=validation.details["index_sha256"],
                metadata_sha256=validation.details["metadata_sha256"],
                path_resolver=resolver,
            )
