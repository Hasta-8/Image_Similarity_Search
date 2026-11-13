# Image Similarity Search Frontend

Create React App UI for ImgSimiSearch. You choose an image, it is sent to the
backend's `POST /search-image`, and the matches are shown in a grid with their
similarity scores.

| File | Role |
|---|---|
| `src/App.js` | Page layout, loading and error state, results grid |
| `src/components/SearchBox.jsx` | File picker and submit |
| `src/api.js` | API base URL, search request, image URL resolution |

## Local development

Start the backend on port 8000 (from `backend/`):

```bash
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Then the dev server:

```bash
cd frontend
npm install
npm start          # http://localhost:3000
```

The app requests relative paths (`/search-image`, `/images/...`, `/thumbnails/...`). In
development, the `proxy` field in `package.json` forwards them to
`http://localhost:8000`. Production builds do not use this proxy.

## API URL configuration

`src/api.js` reads one build-time variable, `REACT_APP_API_BASE_URL`.

- **Empty or unset (recommended):** requests go to the same origin. Use this
  when the backend serves the build itself (`IMGSIMI_FRONTEND_DIR`, as in
  `deploy/local-demo/demo.sh`) or a reverse proxy routes `/search-image`,
  `/health`, `/ready`, `/images/`, and `/thumbnails/` to the backend.
- **Absolute origin** (for example `https://api.example.com`): only when the
  API is deliberately hosted elsewhere. The backend's CORS allowlist is fixed
  to `http://localhost:3000`, so this setup also needs a backend change.

To build for production:

```bash
cp .env.production.example .env.production   # keep the value empty for same-origin
npm run build                                # output in build/
```

CRA bakes the value into the generated JavaScript at build time. Never put
secrets in frontend environment files.

## Backend response contract

```json
{
  "results": [
    {
      "id": 7488443327725773495,
      "score": 0.8123,
      "path": "nested/photo.jpg",
      "image_url": "/images/nested/photo.jpg",
      "thumbnail_url": "/thumbnails/nested/photo.jpg"
    }
  ]
}
```

`image_url` is a relative, already URL-encoded path (for example
`/images/nested%20dir/caf%C3%A9.jpg`). `getImageSrc` keeps it relative for
same-origin use and resolves it against `REACT_APP_API_BASE_URL` when that is
an absolute origin. This preserves nested directories, spaces, and Unicode
names. If `image_url` is `null`, the image cannot be shown. If `image_url` is
missing entirely, the app falls back to `path`.

The grid displays `thumbnail_url` (a small JPEG preview, via `getThumbnailSrc`)
and links each preview to the full-size `image_url`. If a result has no
`thumbnail_url`, or the preview fails to load, the full image is shown
instead.

Non-2xx responses surface as an error message containing the status code and
the backend's detail text (for example 413 for an oversized upload or 429 when
the server is busy).

## Verification

```bash
CI=true npm test -- --watchAll=false   # UI flow and API/image URL handling
npm run build
```

The tests don't replace a browser check against the real deployment's proxy
setup.
