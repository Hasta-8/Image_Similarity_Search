# 🔍 Image Similarity Search

Upload an image and get the most visually similar images from a dataset.
Built with CLIP embeddings, a FAISS vector index, a FastAPI backend, and a
React frontend.

> **Status:** A CLIP-built, validated release is active
> (`backend/artifacts/ACTIVE.json`), so the API serves real CLIP similarity
> results over the 70-image dataset. On the public INRIA Holidays benchmark the
> pipeline scores **mAP 0.844** (see [Benchmark Results](#-benchmark-results)).

## 🌟 Features

- **🧠 CLIP embeddings**: one shared embedding service
  (`openai/clip-vit-base-patch32`, 512-d, L2-normalized) is used for both
  indexing and search, so the two always match.
- **⚡ FAISS search**: exact cosine-similarity search via inner product over
  normalized vectors, returning the top 5 matches with scores.
- **🧪 Reproducible index builds**: deterministic candidate rebuilds with
  stable image IDs, a dataset content fingerprint, and a manifest recording
  model, preprocessing, and failures. The CLIP model is pinned to an exact
  revision, and the API won't serve an index built with a different one.
- **📦 Immutable releases**: candidates are validated, copied into
  read-only release directories, and activated or rolled back by atomically
  swapping a pointer file. An invalid active release causes the API to refuse
  traffic instead of serving bad results.
- **🛡️ Public-demo hardening**: upload size cap, body-read timeout,
  concurrency cap, per-query timeout, path-traversal-safe image serving, and
  API docs off by default.
- **💡 React UI**: upload an image and view the matches in a grid. It can be
  served from the same origin as the API.

## 🧠 How It Works

```
Upload → React → POST /search-image → CLIP embedding → FAISS top-5 → JSON → image grid
```

1. **Build:** `rebuild_index.py` embeds every dataset image with CLIP and
   writes a *candidate* (`faiss_index.bin`, `metadata.json`,
   `manifest.json`) to a new directory.
2. **Validate and promote:** `promote_release.py` validates the candidate
   against the dataset and, with `--confirm`, creates an immutable release in
   `backend/artifacts/releases/<release-id>/` and points `ACTIVE.json` at it.
3. **Serve:** at startup the API loads and re-validates the active release.
   Each query is embedded with the same CLIP service and searched in FAISS.
   Results contain dataset-relative paths and relative `/images/...` URLs.

## 📁 Project Structure

```
ImgSimiSearch/
├── backend/
│   ├── main.py                 # FastAPI app: /search-image, /health, /ready, /images
│   ├── embeddings.py           # Shared CLIP embedding service
│   ├── faiss_index.py          # FAISS helpers
│   ├── rebuild_index.py        # Deterministic candidate build + --validate-only
│   ├── promote_release.py      # Validate and activate a candidate (dry run unless --confirm)
│   ├── rollback_release.py     # List releases / roll back (dry run unless --confirm)
│   ├── release_store.py        # Immutable releases and atomic activation
│   ├── artifact_*.py           # Manifest, validation, safe dataset path handling
│   ├── stable_ids.py           # Path-derived stable 63-bit image IDs
│   ├── dataset_fingerprint.py  # Dataset discovery and content hashing
│   ├── evaluate_retrieval.py   # Precision/Recall/AP/MRR@K against labelled data
│   ├── request_limits.py       # Upload size and body-read timeout middleware
│   ├── index_images.py         # Legacy append-based indexer (not for releases)
│   ├── artifacts/              # Release store (ACTIVE.json + releases/), not in git
│   ├── faiss_index.bin, metadata.json  # Legacy artifacts, local only (see note below)
│   ├── requirements.txt, requirements-dev.txt
│   └── tests/
├── frontend/                   # Create React App UI (App.js, api.js, SearchBox.jsx)
├── dataset/                    # Images being searched (served at /images/...), not in git
├── deploy/                     # Local demo launcher + Nginx/systemd examples
└── evaluation/                 # Benchmark tools (INRIA Holidays) and ground-truth schema
```

The dataset images and generated index artifacts are not committed. To run a
fresh clone, put your images in `dataset/`, then build and promote an index
(see below).

## 🏗️ Setup

Requires **Python 3.10** and Node.js.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-dev.txt   # runtime deps + test deps

cd frontend && npm install && cd ..
```

## ▶️ Running

### Development (two processes)

```bash
# Terminal 1: API on :8000 (run from backend/ so imports resolve)
cd backend
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2: React dev server on :3000 (proxies API and /images to :8000)
cd frontend
npm start
```

Open http://localhost:3000. To enable interactive API docs, start the API with
`IMGSIMI_ENABLE_DOCS=1` and visit http://localhost:8000/docs.

### Local single-origin demo

One FastAPI process serves the built React app, the API, and the images on
`http://127.0.0.1:8080`:

```bash
cd frontend && REACT_APP_API_BASE_URL= GENERATE_SOURCEMAP=false npm run build && cd ..

export IMGSIMI_PYTHON="$PWD/.venv/bin/python"   # interpreter with backend deps
deploy/local-demo/demo.sh start    # waits for /health; warns if /ready is not 200
deploy/local-demo/demo.sh status
deploy/local-demo/demo.sh stop
```

It binds to `127.0.0.1` only. See [deploy/README.md](deploy/README.md) for the
Nginx and systemd examples (templates only; nothing is installed for you).

## 🔌 API

| Endpoint | Description |
|---|---|
| `POST /search-image` | Multipart upload (field `file`). Returns `{"results": [{id, score, path, image_url}, ...]}` with the top 5 matches. |
| `GET /health` | Liveness. Always 200; reports runtime mode, active release ID, and vector counts. |
| `GET /ready` | 200 only when a validated release is loaded; otherwise 503. |
| `GET /images/<path>` | Serves dataset images. |

```bash
curl -X POST -F "file=@dataset/_RSB8594.JPG" http://127.0.0.1:8000/search-image
python backend/test_client.py path/to/query.jpg
```

Error codes: `400` invalid or non-image upload, `413` upload too large, `408`
body read stalled, `429` search capacity busy (with `Retry-After`), `503`
artifacts or model unavailable, `504` query timed out.

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `IMGSIMI_DATASET_DIR` | `<repo>/dataset` | Dataset root |
| `IMGSIMI_RELEASE_STORE` | `backend/artifacts` | Release store |
| `IMGSIMI_FRONTEND_DIR` | unset | Built React app to serve at `/` (single origin) |
| `IMGSIMI_MAX_UPLOAD_BYTES` | `16777216` (16 MiB) | Upload size cap |
| `IMGSIMI_BODY_READ_TIMEOUT_SECONDS` | `30` | Max time to receive the request body |
| `IMGSIMI_MAX_CONCURRENT_SEARCHES` | `2` | Concurrent CLIP inferences |
| `IMGSIMI_SEARCH_TIMEOUT_SECONDS` | `60` | Per-query inference timeout |
| `IMGSIMI_WARMUP_MODEL` | `0` | Load CLIP at startup so the first query isn't slow |
| `IMGSIMI_ENABLE_DOCS` | `0` | Serve `/docs`, `/redoc`, `/openapi.json` |

Frontend: `REACT_APP_API_BASE_URL` (empty = same origin). See
[frontend/README.md](frontend/README.md).

## 📦 Rebuilding and Releasing the Index

Run from `backend/`. Promotion and rollback are dry runs unless `--confirm`
is given, and existing releases are never overwritten.

```bash
# 1. Build a candidate into a NEW, non-existing directory
python rebuild_index.py --dataset ../dataset --output-dir /path/to/new-candidate
python rebuild_index.py --validate-only /path/to/new-candidate

# 2. Preview, then activate
python promote_release.py /path/to/new-candidate --dataset ../dataset
python promote_release.py /path/to/new-candidate --dataset ../dataset --confirm

# 3. Inspect or roll back
python rollback_release.py --list
python rollback_release.py <release-id> --confirm
```

Restart the API after promoting or rolling back; it does not hot-reload
releases. Details: [backend/README.md](backend/README.md).

> **Legacy artifacts:** `backend/faiss_index.bin` and `backend/metadata.json`
> were built by an early version that used random placeholder embeddings, so
> their results are meaningless. They are used only when no `ACTIVE.json`
> exists, and in that mode `/ready` returns 503. They are kept as the
> pre-release rollback baseline. Don't use them for real searches.

## 🧪 Testing

```bash
# Backend (from backend/). Loads the real CLIP model, so the full suite is slow (~15+ min on CPU).
cd backend && python -m pytest -q

# Frontend
cd frontend && CI=true npm test -- --watchAll=false
```

## 📏 Benchmark Results

Retrieval quality was measured on **INRIA Holidays** (Jégou, Douze & Schmid,
ECCV 2008): 1,491 holiday photos in 500 groups of the same scene or object.
The first photo of each group is the query, and the rest of its group are the
correct matches.

| Metric (500 queries, gallery of 1,491) | Result |
|---|---|
| **mAP, official INRIA scorer** | **0.844** |
| mAP, standard (non-interpolated) AP | 0.863 |
| Top-1 result correct | 85.6% |
| Recall@10 | 0.940 |
| MRR | 0.898 |
| *Reference: INRIA's 2008 systems, same scorer* | *0.446 / 0.746* |

| Per-query latency (CPU, after warm-up) | p50 | p95 |
|---|---|---|
| JPEG decode | 61 ms | 135 ms |
| CLIP embedding | 208 ms | 279 ms |
| FAISS exact search (full ranking) | 1.7 ms | 2.0 ms |
| **Total** | **270 ms** | **408 ms** |

Setup: `openai/clip-vit-base-patch32` at the pinned revision, exact
`IndexFlatIP` search, CPU only (8 threads), no fine-tuning, no rotation
correction. Indexing the 1,491 photos took 6.5 minutes (about 7.4 images/s
for CLIP alone) with 2.6 GB peak memory. CLIP accounts for about 75% of query
time, and exact FAISS search costs under 2 ms at this scale.

These figures come from a single run on one machine. They measure the
pipeline on a public benchmark, not on the 70-image demo dataset.
Reproduction steps are in [evaluation/README.md](evaluation/README.md).

`backend/evaluate_retrieval.py` computes Precision@K, Recall@K, AP@K and MRR@K
for any labelled JSONL ground truth, including your own dataset.

## 🚀 Roadmap

- Apply EXIF orientation and re-measure; label the demo dataset for a domain-specific score
- Containerize (Docker) and add CI
- GPU inference for the search API (index builds already accept `--device cuda`)
- Approximate-nearest-neighbour index (HNSW/IVF) for larger datasets
- Authentication and per-client rate limiting for public deployment
- Multimodal (text → image) search

## 📝 License

MIT License: free to use, modify, and distribute.
