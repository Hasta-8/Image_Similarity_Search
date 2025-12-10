"""
Image Indexing Script

Recursively walks a folder, computes embeddings for all images, and adds them to a FAISS index.

Example Usage:
    python index_images.py --folder ../dataset

    This will:
    - Scan the ../dataset folder recursively for image files (.jpg, .jpeg, .png)
    - Print how many images will be indexed before processing
    - Generate embeddings for each image
    - Add them to a FAISS index (default: faiss_index.bin)
    - Save metadata (default: metadata.json)

Full example with all options:
    python index_images.py --folder ../dataset --index-path faiss_index.bin --meta-path metadata.json --dim 512

Note: The script already supports:
    - Accepting --folder argument (required)
    - Printing the number of images found before indexing
    - Recursive folder scanning
    - Progress updates during processing
"""

import os
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple
from faiss_index import (
    create_faiss_index,
    load_faiss_index,
    save_faiss_index,
    add_embeddings,
    load_metadata,
    save_metadata,
    normalize_vectors
)


# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}


def get_image_embedding(image_path: str) -> np.ndarray:
    """
    Compute embedding for an image file.
    
    TODO: Replace this placeholder with actual CLIP embedding generation.
    This function should:
    1. Load the image from image_path
    2. Use CLIP model to generate embedding
    3. Return a 1-D float32 numpy array of dimension 512 (or appropriate CLIP dimension)
    
    Example implementation (to be replaced):
        from embeddings import generate_embedding
        embedding = generate_embedding(image_path)
        return embedding.flatten()  # Ensure 1-D array
    
    Args:
        image_path: Path to the image file
    
    Returns:
        1-D float32 numpy array of embedding dimension
    """
    # TODO: Implement actual CLIP embedding generation here
    # For now, return a dummy embedding to allow testing
    dim = 512  # CLIP vision encoder dimension
    return np.random.rand(dim).astype('float32')


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
    embedding_dim: int = 512
) -> None:
    """
    Index all images in a folder by computing embeddings and adding them to FAISS index.
    
    Args:
        folder_path: Path to folder containing images
        index_path: Path where FAISS index will be saved
        meta_path: Path where metadata JSON will be saved
        embedding_dim: Dimension of embeddings (default: 512 for CLIP)
    """
    print(f"Scanning folder: {folder_path}")
    image_files = find_image_files(folder_path)
    
    if not image_files:
        print("No image files found!")
        return
    
    print(f"Found {len(image_files)} image file(s)\n")
    
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
    
    # Process each image
    print("Processing images...")
    for i, image_path in enumerate(image_files, 1):
        try:
            # Compute embedding
            embedding = get_image_embedding(image_path)
            
            # Validate embedding shape
            if embedding.ndim != 1:
                print(f"  Warning: Embedding for {image_path} is not 1-D, flattening...")
                embedding = embedding.flatten()
            
            if embedding.shape[0] != embedding_dim:
                print(f"  Error: Embedding dimension mismatch for {image_path}")
                print(f"         Expected {embedding_dim}, got {embedding.shape[0]}")
                continue
            
            # Assign ID
            image_id = next_id
            next_id += 1
            
            # Store embedding and ID
            embeddings_list.append(embedding)
            ids_list.append(image_id)
            
            # Store metadata (use absolute path for consistency)
            abs_path = os.path.abspath(image_path)
            metadata_dict[image_id] = abs_path
            
            # Print progress every 10 images or on last image
            if i % 10 == 0 or i == len(image_files):
                print(f"  Processed {i}/{len(image_files)} images...")
        
        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
            continue
    
    if not embeddings_list:
        print("\nNo embeddings were successfully generated!")
        return
    
    print(f"\nSuccessfully processed {len(embeddings_list)} image(s)\n")
    
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
        default=512,
        help="Embedding dimension (default: 512 for CLIP)"
    )
    
    args = parser.parse_args()
    
    try:
        index_images(
            folder_path=args.folder,
            index_path=args.index_path,
            meta_path=args.meta_path,
            embedding_dim=args.dim
        )
    except KeyboardInterrupt:
        print("\n\nIndexing interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        raise


if __name__ == "__main__":
    main()


