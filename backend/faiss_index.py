"""
FAISS Index Management Module

This module provides functions for creating, managing, and searching FAISS indices
optimized for cosine similarity search using normalized vectors.
"""

import os
import json
import faiss
import numpy as np
from typing import List, Tuple, Optional, Dict


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize vectors to unit L2 norm (required for cosine similarity with IndexFlatIP).
    
    Args:
        vectors: Input vectors of shape (n, dim) or (dim,)
    
    Returns:
        Normalized vectors with same shape, dtype float32
    """
    # Ensure float32
    vectors = vectors.astype('float32')
    
    # Handle 1D case
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        if norm > 0:
            return (vectors / norm).astype('float32')
        return vectors
    
    # Handle 2D case - normalize each row
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1.0, norms)
    normalized = vectors / norms
    return normalized.astype('float32')


def create_faiss_index(dim: int) -> faiss.Index:
    """
    Create and return a FAISS index suitable for cosine similarity search.
    
    Uses IndexFlatIP (Inner Product) internally. Since vectors are normalized
    to unit L2 norm, inner product equals cosine similarity.
    The index is wrapped with IndexIDMap to support custom integer IDs.
    
    Args:
        dim: Dimension of the embedding vectors
    
    Returns:
        A FAISS index wrapped with IndexIDMap
    
    Example:
        >>> index = create_faiss_index(512)
        >>> print(type(index))  # <class 'faiss.swigfaiss.IndexIDMap'>
    """
    # Create IndexFlatIP for inner product (cosine similarity with normalized vectors)
    inner_product_index = faiss.IndexFlatIP(dim)
    
    # Wrap with IndexIDMap to support custom integer IDs
    index = faiss.IndexIDMap(inner_product_index)
    
    return index


def add_embeddings(index: faiss.Index, embeddings: np.ndarray, ids: List[int]) -> None:
    """
    Add embeddings to the index and associate them with the provided integer IDs.
    
    Embeddings are normalized to unit L2 norm before adding to ensure cosine similarity
    works correctly with IndexFlatIP.
    
    Args:
        index: FAISS index (should be IndexIDMap wrapped)
        embeddings: Embedding vectors of shape (n, dim), dtype float32
        ids: List of integer IDs corresponding to each embedding
    
    Raises:
        ValueError: If embeddings and ids have mismatched lengths
        AssertionError: If index is not IndexIDMap wrapped
    
    Example:
        >>> index = create_faiss_index(512)
        >>> embeddings = np.random.rand(10, 512).astype('float32')
        >>> ids = list(range(10))
        >>> add_embeddings(index, embeddings, ids)
    """
    # Validate inputs
    if len(embeddings) != len(ids):
        raise ValueError(f"Number of embeddings ({len(embeddings)}) must match number of ids ({len(ids)})")
    
    # Ensure embeddings are float32
    embeddings = embeddings.astype('float32')
    
    # Normalize embeddings to unit L2 norm (required for cosine similarity)
    normalized_embeddings = normalize_vectors(embeddings)
    
    # Convert ids to numpy array of int64 (FAISS requirement)
    id_array = np.array(ids, dtype='int64')
    
    # Add to index with IDs
    index.add_with_ids(normalized_embeddings, id_array)


def save_faiss_index(index: faiss.Index, index_path: str) -> None:
    """
    Persist the FAISS index to disk.
    
    Args:
        index: FAISS index to save
        index_path: Path where the index will be saved
    
    Example:
        >>> index = create_faiss_index(512)
        >>> save_faiss_index(index, "my_index.bin")
    """
    faiss.write_index(index, index_path)


def load_faiss_index(index_path: str) -> faiss.Index:
    """
    Load and return the FAISS index from disk.
    
    Args:
        index_path: Path to the saved index file
    
    Returns:
        Loaded FAISS index
    
    Raises:
        FileNotFoundError: If the index file doesn't exist
    
    Example:
        >>> index = load_faiss_index("my_index.bin")
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")
    
    index = faiss.read_index(index_path)
    return index


def search_index(
    index: faiss.Index,
    query_emb: np.ndarray,
    top_k: int = 5
) -> List[Tuple[int, float]]:
    """
    Search the index for the top_k most similar vectors to the query embedding.
    
    The query embedding is normalized to unit L2 norm before searching.
    Returns results sorted by cosine similarity (best to worst).
    
    Args:
        index: FAISS index to search
        query_emb: Query embedding vector of shape (dim,) or (1, dim), dtype float32
        top_k: Number of top results to return
    
    Returns:
        List of (id, score) tuples sorted by score (best to worst).
        Score is the cosine similarity (inner product of normalized vectors).
    
    Example:
        >>> index = create_faiss_index(512)
        >>> query = np.random.rand(512).astype('float32')
        >>> results = search_index(index, query, top_k=5)
        >>> for id, score in results:
        ...     print(f"ID: {id}, Similarity: {score}")
    """
    # Ensure float32
    query_emb = query_emb.astype('float32')
    
    # Reshape to (1, dim) if needed
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    elif query_emb.ndim == 2 and query_emb.shape[0] != 1:
        raise ValueError(f"Query embedding must be shape (dim,) or (1, dim), got {query_emb.shape}")
    
    # Normalize query embedding to unit L2 norm
    normalized_query = normalize_vectors(query_emb)
    
    # Search the index
    # IndexFlatIP returns inner product scores (which equals cosine similarity for normalized vectors)
    scores, indices = index.search(normalized_query, top_k)
    
    # Convert to list of (id, score) tuples
    # scores[0] and indices[0] because we searched with a single query vector
    results = [
        (int(idx), float(score))
        for idx, score in zip(indices[0], scores[0])
        if idx != -1  # FAISS returns -1 for invalid results
    ]
    
    return results


def save_metadata(metadata: Dict[int, str], meta_path: str) -> None:
    """
    Save the metadata mapping (integer IDs -> file paths) as JSON.
    
    Integer keys are converted to strings for JSON compatibility.
    
    Args:
        metadata: Dictionary mapping integer IDs to file paths
        meta_path: Path where the metadata will be saved
    
    Example:
        >>> metadata = {0: "images/cat.jpg", 1: "images/dog.jpg"}
        >>> save_metadata(metadata, "metadata.json")
    """
    # Convert int keys to strings for JSON compatibility
    json_metadata = {str(k): v for k, v in metadata.items()}
    
    with open(meta_path, 'w') as f:
        json.dump(json_metadata, f, indent=2)


def load_metadata(meta_path: str) -> Dict[int, str]:
    """
    Load and return the metadata mapping (integer IDs -> file paths) from JSON.
    
    If the file doesn't exist, returns an empty dictionary.
    String keys are converted back to integers.
    
    Args:
        meta_path: Path to the metadata JSON file
    
    Returns:
        Dictionary mapping integer IDs to file paths, or empty dict if file doesn't exist
    
    Example:
        >>> metadata = load_metadata("metadata.json")
        >>> print(metadata)  # {0: "images/cat.jpg", 1: "images/dog.jpg"}
    """
    if not os.path.exists(meta_path):
        return {}
    
    with open(meta_path, 'r') as f:
        json_metadata = json.load(f)
    
    # Convert string keys back to integers
    metadata = {int(k): v for k, v in json_metadata.items()}
    
    return metadata


if __name__ == "__main__":
    """
    Example usage: Create an index, add dummy embeddings, and save both index and metadata.
    """
    import tempfile
    
    # Configuration
    dim = 512
    index_path = "example_index.bin"
    meta_path = "example_metadata.json"
    
    print("=== FAISS Index Example ===\n")
    
    # Step 1: Create a new FAISS index
    print("1. Creating FAISS index...")
    index = create_faiss_index(dim)
    print(f"   Created index with dimension {dim}\n")
    
    # Step 2: Generate dummy embeddings
    print("2. Generating dummy embeddings...")
    embedding1 = np.random.rand(dim).astype('float32')
    embedding2 = np.random.rand(dim).astype('float32')
    embeddings = np.stack([embedding1, embedding2])
    print(f"   Generated {len(embeddings)} embeddings of shape {embeddings.shape}\n")
    
    # Step 3: Define IDs and file paths
    print("3. Setting up IDs and metadata...")
    ids = [100, 200]  # Example IDs
    file_paths = {
        100: "images/photo1.jpg",
        200: "images/photo2.jpg"
    }
    print(f"   IDs: {ids}")
    print(f"   File paths: {file_paths}\n")
    
    # Step 4: Add embeddings to index
    print("4. Adding embeddings to index...")
    add_embeddings(index, embeddings, ids)
    print(f"   Added {index.ntotal} vectors to index\n")
    
    # Step 5: Save index
    print("5. Saving FAISS index...")
    save_faiss_index(index, index_path)
    print(f"   Saved index to {index_path}\n")
    
    # Step 6: Save metadata
    print("6. Saving metadata...")
    save_metadata(file_paths, meta_path)
    print(f"   Saved metadata to {meta_path}\n")
    
    # Step 7: Demonstrate loading
    print("7. Loading index and metadata...")
    loaded_index = load_faiss_index(index_path)
    loaded_metadata = load_metadata(meta_path)
    print(f"   Loaded index with {loaded_index.ntotal} vectors")
    print(f"   Loaded metadata: {loaded_metadata}\n")
    
    # Step 8: Demonstrate search
    print("8. Performing similarity search...")
    query_embedding = np.random.rand(dim).astype('float32')
    results = search_index(loaded_index, query_embedding, top_k=2)
    print(f"   Query embedding shape: {query_embedding.shape}")
    print("   Search results:")
    for id, score in results:
        file_path = loaded_metadata.get(id, "Unknown")
        print(f"     ID {id}: {file_path} (similarity: {score:.4f})")
    
    print("\n=== Example completed successfully! ===")

