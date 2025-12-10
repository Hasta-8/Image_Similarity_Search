"""
FastAPI Backend for Image Similarity Search

Uses FAISS for vector search and CLIP for image embeddings.
"""

import os
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from io import BytesIO
import numpy as np

# Import FAISS helper functions
from faiss_index import (
    load_faiss_index,
    create_faiss_index,
    search_index,
    load_metadata
)

app = FastAPI(title="Image Similarity Search API", version="1.0.0")

# Mount static file directory for dataset images
app.mount("/images", StaticFiles(directory="../dataset"), name="images")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for cached index and metadata
index = None
metadata: Dict[int, str] = {}
index_path = "faiss_index.bin"
metadata_path = "metadata.json"
embedding_dim = 512  # CLIP vision encoder output dimension


def get_image_embedding(file_bytes_or_path) -> np.ndarray:
    """
    Compute embedding for an image from file bytes or file path.
    
    TODO: Replace this placeholder with actual CLIP embedding generation.
    This function should:
    1. Accept either file bytes (bytes) or file path (str)
    2. Load/process the image
    3. Use CLIP model to generate embedding
    4. Return a 1-D float32 numpy array of dimension 512 (or appropriate CLIP dimension)
    
    Example implementation (to be replaced):
        from embeddings import generate_embedding
        
        if isinstance(file_bytes_or_path, bytes):
            # Handle file bytes
            image = Image.open(BytesIO(file_bytes_or_path)).convert('RGB')
            embedding = generate_embedding(image)
        else:
            # Handle file path
            embedding = generate_embedding(file_bytes_or_path)
        
        return embedding.flatten().astype('float32')
    
    Args:
        file_bytes_or_path: Either bytes (image file content) or str (file path)
    
    Returns:
        1-D float32 numpy array of embedding dimension
    """
    # TODO: Implement actual CLIP embedding generation here
    # For now, return a dummy embedding to allow testing
    return np.random.rand(embedding_dim).astype('float32')


def load_index_and_metadata():
    """
    Load FAISS index and metadata, creating new index if files don't exist.
    Caches results in global variables.
    """
    global index, metadata
    
    # Load or create index
    if os.path.exists(index_path):
        try:
            index = load_faiss_index(index_path)
            print(f"Loaded FAISS index from {index_path} ({index.ntotal} vectors)")
        except Exception as e:
            print(f"Error loading index: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load FAISS index: {str(e)}"
            )
    else:
        print(f"Index file not found at {index_path}, creating new index...")
        index = create_faiss_index(embedding_dim)
        print(f"Created new FAISS index with dimension {embedding_dim}")
    
    # Load metadata
    metadata.clear()
    metadata.update(load_metadata(metadata_path))
    print(f"Loaded metadata: {len(metadata)} entries")


@app.on_event("startup")
async def startup_event():
    """Load index and metadata on startup."""
    try:
        load_index_and_metadata()
        print(f"API ready! Index contains {index.ntotal if index else 0} vectors.")
    except Exception as e:
        print(f"Warning: Could not load index/metadata on startup: {e}")
        print("API will start but search endpoints may not work until index is created.")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to Image Similarity Search API"}


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns status and basic statistics.
    """
    return {
        "status": "ok",
        "index_loaded": index is not None,
        "indexed_images": index.ntotal if index else 0,
        "metadata_count": len(metadata),
        "index_file_exists": os.path.exists(index_path),
        "metadata_file_exists": os.path.exists(metadata_path)
    }


@app.post("/search-image")
async def search_image(file: UploadFile = File(...)):
    """
    Search for similar images using FAISS.
    
    Accepts an image file and returns the top 5 most similar images
    with their IDs, similarity scores, file paths, and image URLs.
    
    Args:
        file: Image file uploaded via multipart/form-data
    
    Returns:
        JSON response with search results:
        {
            "results": [
                {"id": <int>, "score": <float>, "path": "<path>", "image_url": "<url>"},
                ...
            ]
        }
    """
    global index, metadata
    
    # Check if index is loaded
    if index is None:
        try:
            load_index_and_metadata()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Index not available: {str(e)}"
            )
    
    if index.ntotal == 0:
        raise HTTPException(
            status_code=400,
            detail="No images indexed yet. Please index some images first."
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (jpg, png, etc.)"
        )
    
    try:
        # Read file bytes
        file_bytes = await file.read()
        
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file provided")
        
        # Compute embedding using placeholder function
        # TODO: Replace with actual CLIP embedding function
        query_embedding = get_image_embedding(file_bytes)
        
        # Validate embedding shape
        if query_embedding.ndim != 1:
            query_embedding = query_embedding.flatten()
        
        if query_embedding.shape[0] != embedding_dim:
            raise HTTPException(
                status_code=500,
                detail=f"Embedding dimension mismatch: expected {embedding_dim}, got {query_embedding.shape[0]}"
            )
        
        # Search index using FAISS helper
        search_results = search_index(index, query_embedding, top_k=5)
        
        # Map IDs to file paths using metadata and build response
        results = []
        for image_id, score in search_results:
            file_path = metadata.get(image_id, "Unknown")
            
            # Build image URL using basename of the path
            if file_path != "Unknown":
                basename = os.path.basename(file_path)
                image_url = f"http://localhost:8000/images/{basename}"
            else:
                image_url = None
            
            results.append({
                "id": int(image_id),  # Ensure JSON serializable int
                "score": float(score),  # Ensure JSON serializable float
                "path": file_path,
                "image_url": image_url
            })
        
        return {"results": results}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing search request: {str(e)}"
        )
