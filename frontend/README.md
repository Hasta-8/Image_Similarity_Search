# Image Similarity Search Frontend

React frontend for image similarity search using CLIP embeddings and FAISS.

## Running the Application

1. **Start the backend server:**
   ```bash
   uvicorn backend.main:app --reload
   ```

2. **Start the frontend (from project root):**
   ```bash
   cd frontend
   npm start
   ```

3. **Open the application:**
   Open [http://localhost:3000](http://localhost:3000) in your browser.

## API Integration

The app will POST uploads to `/search-image` via the CRA proxy to `http://localhost:8000`.

### Expected Backend Response Format

The frontend expects the backend to return JSON in the following format:

```json
{
  "results": [
    { "id": 12, "score": 0.9234, "path": "/static/images/photo1.jpg" },
    { "id": 5, "score": 0.8765, "path": "/static/images/photo2.jpg" }
  ]
}
```

### Image Source Handling

The `getImageSrc()` function in `src/api.js` handles image source URLs:

- If `result.path` starts with `http` or `data:`, it returns the path as-is
- If `result.b64` is present, it returns a data URI: `data:image/jpeg;base64,{b64}`
- Otherwise, it constructs a URL: `http://localhost:8000{path}` (ensuring leading slash)
