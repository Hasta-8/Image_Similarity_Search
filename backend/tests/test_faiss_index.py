"""
Unit tests for faiss_index module using pytest.
"""

import os
import tempfile
import pytest
import numpy as np
from faiss_index import (
    create_faiss_index,
    add_embeddings,
    search_index,
    save_faiss_index,
    load_faiss_index,
    normalize_vectors
)


@pytest.fixture
def random_seed():
    """Set random seed for deterministic tests."""
    np.random.seed(0)
    yield
    # Reset seed after test (optional, but good practice)


@pytest.fixture
def test_dim():
    """Test embedding dimension."""
    return 8


@pytest.fixture
def test_embeddings(test_dim):
    """Generate deterministic test embeddings."""
    np.random.seed(0)
    # Generate 3 random embeddings
    embeddings = np.random.rand(3, test_dim).astype('float32')
    # Normalize them
    embeddings = normalize_vectors(embeddings)
    return embeddings


@pytest.fixture
def test_ids():
    """Test IDs for embeddings."""
    return [1, 2, 3]


def test_create_faiss_index(test_dim):
    """Test FAISS index creation."""
    index = create_faiss_index(test_dim)
    
    assert index is not None
    assert index.ntotal == 0  # Initially empty
    assert index.d == test_dim  # Correct dimension


def test_add_embeddings(test_dim, test_embeddings, test_ids):
    """Test adding embeddings to FAISS index."""
    index = create_faiss_index(test_dim)
    
    # Add embeddings
    add_embeddings(index, test_embeddings, test_ids)
    
    # Verify embeddings were added
    assert index.ntotal == len(test_ids)
    assert index.ntotal == 3


def test_search_index_exact_match(test_dim, test_embeddings, test_ids):
    """Test searching with an exact match (one of the added vectors)."""
    np.random.seed(0)
    
    index = create_faiss_index(test_dim)
    add_embeddings(index, test_embeddings, test_ids)
    
    # Use the first embedding as query (should match id=1)
    query_vector = test_embeddings[0].copy()
    
    # Search
    results = search_index(index, query_vector, top_k=3)
    
    # Verify results
    assert len(results) == 3
    
    # Top result should be id=1 (exact match)
    top_id, top_score = results[0]
    assert top_id == 1
    
    # Score should be very close to 1.0 (cosine similarity of vector with itself)
    assert top_score > 0.99, f"Expected score > 0.99, got {top_score}"
    
    # Verify all expected IDs are in results
    result_ids = [id for id, _ in results]
    assert set(result_ids) == set(test_ids)


def test_search_index_different_vector(test_dim, test_embeddings, test_ids):
    """Test searching with a different vector."""
    np.random.seed(0)
    
    index = create_faiss_index(test_dim)
    add_embeddings(index, test_embeddings, test_ids)
    
    # Create a different query vector
    query_vector = np.random.rand(test_dim).astype('float32')
    query_vector = normalize_vectors(query_vector)
    
    # Search
    results = search_index(index, query_vector, top_k=2)
    
    # Verify we get results
    assert len(results) >= 1
    assert len(results) <= 2
    
    # Verify results are sorted by score (best to worst)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    
    # Verify all scores are between -1 and 1 (cosine similarity range)
    for _, score in results:
        assert -1.0 <= score <= 1.0


def test_save_load_roundtrip(test_dim, test_embeddings, test_ids):
    """Test saving and loading FAISS index maintains functionality."""
    np.random.seed(0)
    
    # Create index and add embeddings
    original_index = create_faiss_index(test_dim)
    add_embeddings(original_index, test_embeddings, test_ids)
    
    # Use first embedding as query
    query_vector = test_embeddings[0].copy()
    
    # Search original index
    original_results = search_index(original_index, query_vector, top_k=3)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Save index
        save_faiss_index(original_index, tmp_path)
        
        # Verify file exists
        assert os.path.exists(tmp_path)
        
        # Load index
        loaded_index = load_faiss_index(tmp_path)
        
        # Verify loaded index has same properties
        assert loaded_index.ntotal == original_index.ntotal
        assert loaded_index.d == original_index.d
        
        # Search loaded index with same query
        loaded_results = search_index(loaded_index, query_vector, top_k=3)
        
        # Verify results are identical
        assert len(loaded_results) == len(original_results)
        
        for (orig_id, orig_score), (loaded_id, loaded_score) in zip(
            original_results, loaded_results
        ):
            assert orig_id == loaded_id
            # Scores should be very close (allowing for floating point precision)
            assert abs(orig_score - loaded_score) < 1e-5, \
                f"Score mismatch: {orig_score} vs {loaded_score}"
        
        # Verify top result matches
        assert original_results[0][0] == loaded_results[0][0]
        assert original_results[0][0] == 1  # Should match id=1
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_search_with_top_k_larger_than_index(test_dim, test_embeddings, test_ids):
    """Test searching with top_k larger than number of indexed vectors."""
    np.random.seed(0)
    
    index = create_faiss_index(test_dim)
    add_embeddings(index, test_embeddings, test_ids)
    
    query_vector = test_embeddings[0].copy()
    
    # Request more results than available
    results = search_index(index, query_vector, top_k=10)
    
    # Should return at most 3 results (number of indexed vectors)
    assert len(results) <= 3
    assert len(results) == 3  # Should return all available


def test_normalize_vectors():
    """Test vector normalization."""
    np.random.seed(0)
    
    # Test 1D vector
    vec_1d = np.random.rand(8).astype('float32')
    normalized = normalize_vectors(vec_1d)
    
    assert normalized.shape == vec_1d.shape
    assert normalized.dtype == np.float32
    norm = np.linalg.norm(normalized)
    assert abs(norm - 1.0) < 1e-5, f"Expected norm ~1.0, got {norm}"
    
    # Test 2D array
    vec_2d = np.random.rand(3, 8).astype('float32')
    normalized_2d = normalize_vectors(vec_2d)
    
    assert normalized_2d.shape == vec_2d.shape
    assert normalized_2d.dtype == np.float32
    # Check each row is normalized
    for row in normalized_2d:
        norm = np.linalg.norm(row)
        assert abs(norm - 1.0) < 1e-5, f"Expected norm ~1.0, got {norm}"




