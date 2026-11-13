"""
Image Indexing Script

Recursively walks a folder, computes embeddings for all images, and adds them to a FAISS index.

Example Usage:
    python index_images.py --folder ../dataset

    This will:
    - Scan the ../dataset folder recursively for image files (.jpg, .jpeg, .png)
    - Print how many images will be indexed before processing
    - Generate normalized CLIP embeddings in bounded batches
    - Add them to a FAISS index (default: faiss_index.bin)
    - Save metadata (default: metadata.json)

Full example with all options:
    python index_images.py --folder ../dataset --index-path faiss_index.bin \
        --meta-path metadata.json --batch-size 16

The embedding dimension is fixed at 512 for openai/clip-vit-base-patch32.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List

import numpy as np

from embeddings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBEDDING_DIM,
    EmbeddingError,
    generate_embedding,
    generate_embeddings_batch,
)
from faiss_index import (
    create_faiss_index,
    load_faiss_index,
    save_faiss_index,
    add_embeddings,
    load_metadata,
    save_metadata,
    normalize_vectors
)

logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}


def get_image_embedding(image_path: str) -> np.ndarray:
    """Compute a normalized CLIP embedding for an image file.

    This compatibility wrapper delegates to the shared embedding service. The
    indexing loop uses the batch API directly; this helper remains for callers
    that need a single vector.
    """
    return generate_embedding(image_path).reshape(-1)


def _validate_embedding_matrix(
    embeddings: np.ndarray,
    expected_count: int,
    embedding_dim: int,
) -> np.ndarray:
    """Validate a batch of unit-normalized float32 embeddings."""
    if not isinstance(embeddings, np.ndarray):
        raise ValueError(
            f"Embedding output must be a numpy array, got {type(embeddings).__name__}"
        )
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    expected_shape = (expected_count, embedding_dim)
    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Embedding shape mismatch: expected {expected_shape}, got {embeddings.shape}"
        )
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("Embedding batch contains non-finite values")
    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("Embedding batch contains a zero-norm vector")
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError(
            f"Embedding batch is not unit-normalized (norms {norms.min():.6f}"
            f"..{norms.max():.6f})"
        )
    return embeddings


def find_image_files(folder_path: str) -> List[str]:
    """
    Recursively find all image files in the given folder.
    
    Args:
        folder_path: Root folder to search
    
    Returns:
        List of image file paths
    """
    image_files = []
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    # Recursively walk through the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix in IMAGE_EXTENSIONS:
                image_files.append(str(file_path))
    
    return sorted(image_files)  # Sort for consistent ordering


def index_images(
    folder_path: str,
    index_path: str,
    meta_path: str,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """
    Index all images in a folder by computing CLIP embeddings and adding them
    to a FAISS index.

    Args:
        folder_path: Path to folder containing images
        index_path: Path where FAISS index will be saved
        meta_path: Path where metadata JSON will be saved
        embedding_dim: Dimension of embeddings (must be 512 for this checkpoint)
        batch_size: Maximum number of images per CLIP inference batch
    """
    if embedding_dim != DEFAULT_EMBEDDING_DIM:
        raise ValueError(
            f"This CLIP checkpoint produces {DEFAULT_EMBEDDING_DIM}-dimensional "
            f"embeddings; embedding_dim={embedding_dim} is not supported"
        )
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")

    print(f"Scanning folder: {folder_path}")
    image_files = find_image_files(folder_path)

    if not image_files:
        print("No image files found!")
        return

    print(f"Found {len(image_files)} image file(s)")
    print(
        f"Embedding model: openai/clip-vit-base-patch32 "
        f"({embedding_dim}-d), batch_size={batch_size}\n"
    )

    # Load existing metadata to continue ID sequence
    metadata = load_metadata(meta_path)
    if metadata:
        next_id = max(metadata.keys()) + 1
        print(f"Found existing metadata with {len(metadata)} entries")
        print(f"Starting new IDs from {next_id}\n")
    else:
        next_id = 0
        print("No existing metadata found, starting IDs from 0\n")

    # Collect embeddings and IDs
    embeddings_list: List[np.ndarray] = []
    ids_list: List[int] = []
    metadata_dict = metadata.copy()  # Start with existing metadata
    failed_images: List[tuple[str, str]] = []

    print("Processing images in bounded batches...")
    processed_count = 0
    for batch_start in range(0, len(image_files), batch_size):
        batch_paths = image_files[batch_start : batch_start + batch_size]
        batch_embeddings: np.ndarray
        successful_paths: List[str]

        try:
            batch_embeddings = generate_embeddings_batch(
                batch_paths, batch_size=batch_size
            )
            batch_embeddings = _validate_embedding_matrix(
                batch_embeddings, len(batch_paths), embedding_dim
            )
            successful_paths = batch_paths
        except EmbeddingError as batch_error:
            logger.warning(
                "Batch %d failed (%s); retrying images individually",
                batch_start // batch_size + 1,
                batch_error,
            )
            print(
                f"  Batch at image {batch_start + 1} failed; "
                "retrying images individually to isolate invalid files..."
            )
            single_embeddings: List[np.ndarray] = []
            successful_paths = []
            for image_path in batch_paths:
                try:
                    single_embedding = generate_embedding(image_path).reshape(-1)
                    single_embedding = _validate_embedding_matrix(
                        single_embedding.reshape(1, -1), 1, embedding_dim
                    )[0]
                except EmbeddingError as exc:
                    failed_images.append((image_path, str(exc)))
                    print(f"  Skipping {image_path}: {exc}")
                    continue
                single_embeddings.append(single_embedding)
                successful_paths.append(image_path)
            if not single_embeddings:
                continue
            batch_embeddings = np.stack(single_embeddings).astype('float32', copy=False)

        for image_path, embedding in zip(successful_paths, batch_embeddings):
            image_id = next_id
            next_id += 1

            embeddings_list.append(embedding)
            ids_list.append(image_id)
            metadata_dict[image_id] = os.path.abspath(image_path)
            processed_count += 1

        print(
            f"  Processed {processed_count}/{len(image_files)} image(s) "
            f"({len(failed_images)} failed so far)"
        )

    if not embeddings_list:
        print("\nNo embeddings were successfully generated!")
        if failed_images:
            print("Failed images:")
            for image_path, reason in failed_images:
                print(f"  - {image_path}: {reason}")
        return

    print(f"\nSuccessfully processed {len(embeddings_list)} image(s)")
    if failed_images:
        print(f"Failed images ({len(failed_images)}):")
        for image_path, reason in failed_images:
            print(f"  - {image_path}: {reason}")
    print()

    # Stack embeddings into numpy array
    print("Stacking embeddings...")
    embeddings_array = np.stack(embeddings_list).astype('float32')
    print(f"  Embeddings shape: {embeddings_array.shape}\n")

    # Normalize embeddings (required for cosine similarity)
    print("Normalizing embeddings...")
    embeddings_array = normalize_vectors(embeddings_array)
    print("  Embeddings normalized\n")

    # Load or create FAISS index
    print("Loading or creating FAISS index...")
    try:
        index = load_faiss_index(index_path)
        print(f"  Loaded existing index with {index.ntotal} vectors")
    except FileNotFoundError:
        index = create_faiss_index(embedding_dim)
        print(f"  Created new index with dimension {embedding_dim}")
    print()
    
    # Add embeddings to index
    print(f"Adding {len(embeddings_list)} embedding(s) to index...")
    add_embeddings(index, embeddings_array, ids_list)
    print(f"  Index now contains {index.ntotal} total vector(s)\n")
    
    # Save index
    print(f"Saving FAISS index to {index_path}...")
    save_faiss_index(index, index_path)
    print("  Index saved\n")
    
    # Save metadata
    print(f"Saving metadata to {meta_path}...")
    save_metadata(metadata_dict, meta_path)
    print(f"  Metadata saved ({len(metadata_dict)} entries)\n")
    
    # Print summary
    print("=" * 60)
    print("INDEXING SUMMARY")
    print("=" * 60)
    print(f"Folder scanned:     {folder_path}")
    print(f"Images found:       {len(image_files)}")
    print(f"Images indexed:     {len(embeddings_list)}")
    print(f"Images failed:      {len(failed_images)}")
    print(f"Batch size:         {batch_size}")
    print(f"Total in index:     {index.ntotal}")
    print(f"Index file:         {index_path}")
    print(f"Metadata file:      {meta_path}")
    print(f"Metadata entries:   {len(metadata_dict)}")
    print("=" * 60)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Index images in a folder using CLIP embeddings and FAISS"
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to folder containing images (will be searched recursively)"
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="faiss_index.bin",
        help="Path where FAISS index will be saved (default: faiss_index.bin)"
    )
    parser.add_argument(
        "--meta-path",
        type=str,
        default="metadata.json",
        help="Path where metadata JSON will be saved (default: metadata.json)"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help=f"Embedding dimension (fixed at {DEFAULT_EMBEDDING_DIM} for this checkpoint)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Maximum images per CLIP inference batch (default: {DEFAULT_BATCH_SIZE})"
    )

    args = parser.parse_args()
    
    try:
        index_images(
            folder_path=args.folder,
            index_path=args.index_path,
            meta_path=args.meta_path,
            embedding_dim=args.dim,
            batch_size=args.batch_size,
        )
    except KeyboardInterrupt:
        print("\n\nIndexing interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        raise


if __name__ == "__main__":
    main()


