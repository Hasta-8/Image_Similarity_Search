# Image Similarity Search Backend

FastAPI service that embeds an uploaded image with CLIP and returns the most
similar dataset images from a FAISS index. It also contains the tooling to
build, validate, release, roll back, and evaluate that index.

Run every command in this file from `backend/`. The modules use flat imports,
so the working directory matters.

## Installation

Python 3.10 is required (the pins in `requirements.txt` target it).

```bash
pip install -r requirements.txt       # runtime
pip install -r requirements-dev.txt   # runtime + httpx for the API tests
```

## Running the API

```bash
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

This is for local development. For a persistent host, use a process manager
with a single worker (each worker loads its own CLIP model). See `../deploy/`
for a local single-origin demo launcher and Nginx/systemd templates.

### Endpoints

| Endpoint | Behaviour |
|---|---|
| `POST /search-image` | Multipart upload, field `file`. Returns the top 5 matches. |
| `GET /health` | Liveness. Always 200; reports `runtime_mode`, `active_release_id`, and counts. |
| `GET /ready` | 200 only when a validated release is loaded, otherwise 503. |
| `GET /images/<path>` | Static files from the dataset directory. |
| `GET /thumbnails/<path>` | Downscaled JPEG preview of a dataset image (see Thumbnails below). |
| `GET /` | JSON welcome message, or the React app when `IMGSIMI_FRONTEND_DIR` is set. |
| `/docs`, `/redoc`, `/openapi.json` | 404 unless `IMGSIMI_ENABLE_DOCS=1`. |

Search response:

```json
{
  "results": [
    {"id": 7488443327725773495, "score": 1.0, "path": "_RSB8594.JPG", "image_url": "/images/_RSB8594.JPG", "thumbnail_url": "/thumbnails/_RSB8594.JPG"}
  ]
}
```

`path` is always dataset-relative (never an absolute host path) and `image_url`
is a URL-encoded relative `/images/...` URL. `thumbnail_url` is the same path
under `/thumbnails/`. If a result file is missing or
resolves outside the dataset, the result is still returned, but its `path` is
`"Unknown"` (or a bare file name in legacy mode) and its `image_url` and
`thumbnail_url` are `null`.

Error responses:

| Code | Cause |
|---|---|
| 400 | Not an image content type, empty file, or an image that cannot be decoded |
| 408 | Request body not received within the read timeout |
| 413 | Body larger than the upload limit (checked before and while reading) |
| 429 | All search slots are busy; includes `Retry-After: 5` |
| 503 | Artifacts not loaded, or the embedding model failed |
| 504 | Query embedding exceeded the search timeout |

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `IMGSIMI_DATASET_DIR` | `../dataset` | Dataset root, also served at `/images` |
| `IMGSIMI_RELEASE_STORE` | `./artifacts` | Release store containing `ACTIVE.json` and `releases/` |
| `IMGSIMI_FRONTEND_DIR` | unset | Built React app to serve at `/`; must contain `index.html` |
| `IMGSIMI_MAX_UPLOAD_BYTES` | `16777216` | Upload size limit |
| `IMGSIMI_BODY_READ_TIMEOUT_SECONDS` | `30` | Request body read timeout |
| `IMGSIMI_MAX_CONCURRENT_SEARCHES` | `2` | Concurrent query embeddings |
| `IMGSIMI_SEARCH_TIMEOUT_SECONDS` | `60` | Per-query embedding timeout |
| `IMGSIMI_WARMUP_MODEL` | `0` | Load CLIP at startup |
| `IMGSIMI_THUMBNAIL_SIZE` | `384` | Longest side of previews in pixels (16–2048) |
| `IMGSIMI_THUMBNAIL_CACHE_MB` | `64` | In-memory preview cache budget (`0` disables caching) |
| `IMGSIMI_ENABLE_DOCS` | `0` | Enable the OpenAPI docs routes |

Invalid numeric values stop the server at import time. A timed-out search keeps
its slot until the worker thread actually finishes, so abandoned inference still
counts against the concurrency limit.

CORS allows only `http://localhost:3000` (the CRA dev server), and there is no
environment variable to change it. Production deployments are expected to be
same-origin.

## Embeddings

`embeddings.py` is the single CLIP service used by both indexing and search:
`openai/clip-vit-base-patch32`, 512-dimensional, float32, L2-normalized. Images
that cannot be decoded raise an error; there are no random placeholder
embeddings. The model is loaded on first use unless `IMGSIMI_WARMUP_MODEL=1`.

The model is pinned to Hugging Face commit
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` (`DEFAULT_MODEL_REVISION`). If the
upstream repository changes, the weights and preprocessing used here do not.
Rebuilds record this revision in the manifest. At startup the API refuses to
serve a release built with a different model or revision (see below). To change
the model or revision, update the constant, then rebuild and promote a new
release.

## Thumbnails

Dataset originals can be 4–14 MB each, which is slow to load on a phone. The
results grid uses `GET /thumbnails/<path>` instead. It returns a JPEG whose
longest side is `IMGSIMI_THUMBNAIL_SIZE`, and it never upscales.

- **Generation:** a thumbnail is made on first request in a worker thread
  (`thumbnails.py`). JPEG draft decoding keeps this fast even for large photos.
- **Caching:** results are kept in an in-memory LRU cache bounded by
  `IMGSIMI_THUMBNAIL_CACHE_MB`. Cache keys include the source file's
  modification time and size, so an edited image gets a fresh thumbnail.
  Responses send `Cache-Control: public, max-age=86400`.
- **Orientation:** the EXIF orientation tag is applied, so previews display
  the same way browsers show the originals. Embeddings are not affected.
- **Path safety:** paths use the same dataset-root confinement as search
  results. Missing, non-image, undecodable, traversal and symlink-escape paths
  all return 404.

## Building a Candidate Index

`rebuild_index.py` does a deterministic full rebuild into a new directory. It
never touches the active release or the legacy files.

```bash
python rebuild_index.py --dataset ../dataset --output-dir /path/to/new-candidate
python rebuild_index.py --validate-only /path/to/new-candidate
```

Options: `--batch-size`, `--device auto|cpu|cuda|cuda:N`, `--log-level`. The
output directory must not already exist.

A candidate contains `faiss_index.bin`, `metadata.json`, and `manifest.json`.
The manifest records the dataset fingerprint (`dataset_fingerprint.py`), model
and revision, preprocessing, normalization, ID strategy, index details, and
per-image failures. Image IDs come from `stable_ids.py`: a SHA-256 of the
normalized dataset-relative path, truncated to a positive 63-bit integer, so
the same image gets the same ID across rebuilds.

`index_images.py` is the older append-based indexer. It writes to whatever
paths it is given, so do not use it to produce releases.

## Release Promotion and Rollback

Promotion and rollback are dry runs unless `--confirm` is passed. Existing
releases are never overwritten or deleted.

```bash
# Validate the candidate and show the planned release.
python promote_release.py /path/to/new-candidate --dataset ../dataset

# Copy it into an immutable release and atomically activate it.
python promote_release.py /path/to/new-candidate --dataset ../dataset --confirm

# List releases (* marks the active one).
python rollback_release.py --list

# Preview, then switch the active pointer to an existing release.
python rollback_release.py <release-id>
python rollback_release.py <release-id> --confirm
```

Both tools accept `--release-store` to work on a store other than `./artifacts`.
Restart the API after any change. It reads `ACTIVE.json` only at startup.

### Release identity

A release ID is derived from the index checksum, the metadata checksum, and a
canonical checksum of the semantic manifest fields (dataset identity, model and
revision, preprocessing, normalization, ID strategy, index metadata, failures).
Two candidates with identical files but different provenance therefore get
different IDs. Promoting a candidate whose release already exists checks that
the provenance matches and rejects it if it does not.

### Startup, readiness, and legacy fallback

- **`ACTIVE.json` exists:** the API loads that release, re-validates it
  against the dataset, and checks that the manifest's model name and revision
  match the pinned query encoder. If any check fails, the API refuses to serve: it stays
  up for `/health`, but `/ready` and `/search-image` return 503. It never falls
  back to anything else.
- **No `ACTIVE.json`:** the API loads `faiss_index.bin` and `metadata.json` from
  this directory. These were built by an early version that used random
  placeholder embeddings, so their results are meaningless. Search still works
  for backward compatibility. Results are reduced to a dataset-relative path or
  bare file name with a relative `/images/...` URL. `/ready` returns 503 in
  this mode.

Keep the legacy files. They are the baseline to restore if the first release is
ever removed, because there is no earlier release for `rollback_release.py` to
target. That manual procedure is in `../deploy/README.md`.

## Evaluating Retrieval Quality

```bash
python evaluate_retrieval.py \
  --candidate-dir /path/to/candidate \
  --dataset ../dataset \
  --ground-truth ../evaluation/ground_truth.jsonl \
  --ks 1,5,10
```

Requires human-labelled query/relevance pairs, and none exist yet. Metric
definitions are in `retrieval_metrics.py`, and the label format is in
`../evaluation/README.md`.

## Testing

```bash
python -m pytest -q
```

The suite includes integration tests that load the real CLIP model, so a full
CPU run takes roughly 15–20 minutes. Run a single file for faster feedback, for
example `python -m pytest -q tests/test_release_store.py`. Tests use temporary
directories and do not modify `artifacts/` or the legacy files.

Manual checks against a running server on port 8000:

```bash
python test_client.py ../dataset/_RSB8594.JPG
bash test_search.sh ../dataset/_RSB8594.JPG
```
