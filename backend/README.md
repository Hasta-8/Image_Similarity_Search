# Image Similarity Search Backend

FastAPI backend for image similarity search using CLIP embeddings and FAISS.

## Installation

```bash
pip install -r requirements.txt
```

## Indexing Images

Index all images in a folder recursively:

```bash
python backend/index_images.py --folder /path/to/images --index-path backend/faiss_index.bin --meta-path backend/metadata.json
```

## Running the API

Start the FastAPI server:

```bash
uvicorn backend.main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`

## Testing

Test the search endpoint:

```bash
bash backend/test_search.sh path/to/image.jpg
```

Or use the Python client:

```bash
python backend/test_client.py path/to/image.jpg
```

## CLIP Integration

To integrate CLIP embeddings, replace the placeholder functions marked with `TODO` comments:
- `backend/index_images.py`: `get_image_embedding()` function
- `backend/main.py`: `get_image_embedding()` function




