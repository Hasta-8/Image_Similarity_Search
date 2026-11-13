# Deployment templates (not installed or activated)

The repository does not identify a production host, domain, frontend host, or
service account. These examples are environment-neutral starting points, not a
claim that ImgSimiSearch is deployed. Replace every placeholder and validate the
result against the actual host before use.

## Local single-origin demo (no tunnel)

`deploy/local-demo/demo.sh` runs one FastAPI process that serves the production
React build, `/search-image`, `/health`, `/ready`, and `/images/...` on
`http://127.0.0.1:8080`. It starts no tunnel and installs nothing.

The script uses `<repo>/.venv/bin/python` by default. To use a different
Python 3.10 interpreter with the backend requirements installed, set
`IMGSIMI_PYTHON` to its path. Other overrides (port, dataset, release store, state
directory, limits) are listed at the top of `demo.sh`. The PID file and server
log go to `~/.local/state/imgsimisearch-demo/` by default.

```bash
cd frontend && REACT_APP_API_BASE_URL= GENERATE_SOURCEMAP=false npm run build && cd ..
export IMGSIMI_PYTHON=/path/to/venv/bin/python
deploy/local-demo/demo.sh start    # waits for /health; warns if /ready is not 200
deploy/local-demo/demo.sh status
deploy/local-demo/demo.sh stop
```

Request limits applied by the backend (environment, with defaults):
`IMGSIMI_MAX_UPLOAD_BYTES` (16 MiB, 413 above it), `IMGSIMI_BODY_READ_TIMEOUT_SECONDS`
(30, 408 when a body stalls), `IMGSIMI_MAX_CONCURRENT_SEARCHES` (2, 429 when busy),
`IMGSIMI_SEARCH_TIMEOUT_SECONDS` (60, 504). API docs are off unless
`IMGSIMI_ENABLE_DOCS=1`. The Nginx example above uses the same 16 MiB limit.

## Recommended topology

For a single Linux host, the example topology is:

```text
Browser --HTTPS--> reverse proxy --static files--> frontend/build
                         |
                         +-- /search-image, /health, /ready, /images/* -->
                             FastAPI on 127.0.0.1:8000
```

This keeps the frontend and API same-origin, so the CRA build can use an empty
`REACT_APP_API_BASE_URL`. The Nginx example is plain HTTP and does not configure
TLS, DNS, firewall rules, or an actual hostname. If the selected platform
provides its own ingress/reverse proxy, use that instead of installing Nginx.

## Backend process manager

`systemd` is a suitable example for a persistent Linux host, but the target is
not known and no service was installed, enabled, or started. The template is
`deploy/systemd/imgsimisearch.service`; its sample environment is
`deploy/systemd/imgsimisearch.env.example`.

The backend must be launched with `backend/` as its working directory because
the current modules use flat imports. The intended command is:

```text
<VENV_ROOT>/bin/uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1
```

One worker avoids loading a separate CLIP model for each Uvicorn worker. The
example uses systemd's non-root dynamic user with persistent state/cache
directories. Before installing the unit, replace `__APP_ROOT__` and
`__VENV_ROOT__` with actual absolute paths. The service identity needs:

- read/traverse access to the application code and dataset;
- read access to the pinned CLIP model cache, or network access on first model
  load;
- write access to the configured release store and model cache under its
  `StateDirectory` and `CacheDirectory` paths;
- no write access to the production legacy index/metadata during runtime.

Use absolute installation and virtual-environment paths without spaces (for
example, `/opt/imgsimisearch`); if the selected host requires spaces, escape
them using systemd unit-file syntax before verification.

The current model is loaded lazily. Pre-provision the exact manifest model
revision under `HF_HOME` for offline startup/search, or explicitly allow the
service to reach the model host. Do not put credentials in the frontend or unit
file; use a protected host-side environment file if a future deployment needs
credentials.

The supplied environment example points the release store into the unit's
`StateDirectory` and the Hugging Face cache into its `CacheDirectory`. Replace
the dataset placeholder with a real absolute path and grant the dynamic service
identity read/traverse access to that tree.

After replacing placeholders in a host-specific copy, validate the unit and use
the suggested operator steps (none of these have been run on any host). From
the repository root:

```bash
sudo install -d -o root -g root -m 0750 /etc/imgsimisearch
sudo install -o root -g root -m 0640 /path/to/adapted/imgsimisearch.env \
  /etc/imgsimisearch/imgsimisearch.env
sudo install -o root -g root -m 0644 /path/to/adapted/imgsimisearch.service \
  /etc/systemd/system/imgsimisearch.service
sudo systemd-analyze verify /etc/systemd/system/imgsimisearch.service
sudo systemctl daemon-reload
sudo systemctl enable imgsimisearch.service
sudo systemctl start imgsimisearch.service
sudo systemctl status imgsimisearch.service
journalctl -u imgsimisearch.service
sudo systemctl stop imgsimisearch.service
```

Do not copy these commands into an environment until the installation paths,
data directories, and release activation plan are confirmed.

## Frontend build and same-origin proxy

From `frontend/`, copy `.env.production.example` to `.env.production`, leave
`REACT_APP_API_BASE_URL` empty for the recommended same-origin layout, and run
`npm run build`. CRA embeds the variable at build time. See
`frontend/README.md` for local proxy behavior and alternative-origin caveats.

`deploy/nginx/imgsimisearch.conf.example` serves the CRA build and proxies the
search, health, readiness, and image paths to the backend. It sets request and
response timeouts and disables response buffering for image responses. Its
16 MiB `client_max_body_size` and 30 s `client_body_timeout` match the backend
defaults (`IMGSIMI_MAX_UPLOAD_BYTES`, `IMGSIMI_BODY_READ_TIMEOUT_SECONDS`). If
you change one side, change the other too.

After replacing placeholders, validate with `nginx -t` before installing or
reloading the configuration. No reverse proxy was installed or activated here.

## Health and release expectations

- `/health` is liveness and can return HTTP 200 while the app is in legacy mode.
- `/ready` returns 503 unless a validated release is active. Use it as the
  load-balancer or monitoring readiness check.
- In this checkout, `backend/artifacts/ACTIVE.json` points to release
  `rel-f5e126789ed1beaae68ed38d` (70 vectors). A new host gets its own release
  store (see `IMGSIMI_RELEASE_STORE` in the systemd environment example) and
  needs a release promoted into it before `/ready` returns 200.
- Promotion and rollback only take effect after the application restarts.

## Backup and rollback runbook

Before any separately approved promotion, create a unique timestamped backup on
a durable filesystem outside the release store. Preserve the current legacy
FAISS index and metadata, any existing `ACTIVE.json`, and the complete current
release-store tree when one exists. If the release store is absent, record that
absence instead of creating a fake previous release. Compare SHA-256 hashes of
all copied files with their sources before proceeding. Do not overwrite older
backup directories.

An operator can adapt this outline after selecting a persistent backup root; it
is not executable until `BACKUP_ROOT` is chosen and the timestamped path is
verified unique:

```bash
BACKUP_DIR="$BACKUP_ROOT/imgsimisearch-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -- "$BACKUP_DIR"
cp -- backend/faiss_index.bin backend/metadata.json "$BACKUP_DIR/"
if [ -e backend/artifacts ]; then
  cp -a -- backend/artifacts "$BACKUP_DIR/artifacts"
else
  printf '%s\n' 'No pre-activation release store existed.' \
    > "$BACKUP_DIR/release-store-absent.txt"
fi
sha256sum backend/faiss_index.bin backend/metadata.json \
  "$BACKUP_DIR/faiss_index.bin" "$BACKUP_DIR/metadata.json"
```

Promotion remains dry-run by default. Review the planned release ID and all
artifact/manifest checksums before any separately approved `--confirm` run.
When a prior validated production release exists, first run rollback in dry-run
mode and verify its release ID/checksums; only execute the confirmed rollback
after the appropriate approval. Release directories are immutable and must not
be deleted as part of rollback.

If the first production activation has no prior active release, there is no
release ID for the rollback CLI to target. After stopping only the identified
backend, preserve the new pointer by moving it into the timestamped backup while
holding the release lock, then restart the backend to reload the unchanged
legacy artifacts:

```bash
flock -x "$RELEASE_STORE/.release.lock" \
  mv -- "$RELEASE_STORE/ACTIVE.json" \
  "$BACKUP_DIR/ACTIVE.json-first-activation-rollback"
```

This first-activation recovery procedure is a pointer removal from the active
store, not a deletion of the failed immutable release. Verify the service
manager and paths for the actual host before using it.

This applies to the current local checkout: the active release was the first
activation (`previous_release_id: null`), so returning to the legacy artifacts
means removing `backend/artifacts/ACTIVE.json` as above and restarting. A
verified pre-activation backup of the legacy files was taken outside the
repository.

`RELEASE_STORE` and `BACKUP_DIR` in the command are operator-supplied absolute
paths; verify they refer to the active store and a unique backup destination.
