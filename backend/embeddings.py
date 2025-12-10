"""
CLIP Image Embedding Generator

This module provides functions to generate normalized CLIP image embeddings
from image files or PIL Image objects.
"""

import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import Union
from transformers import CLIPProcessor, CLIPModel

# Global variables for model caching
_model = None
_processor = None


def load_clip_model():
    """
    Load CLIP model and processor from HuggingFace.
    Model is cached after first load.
    
    Returns:
        tuple: (model, processor)
    """
    global _model, _processor
    
    if _model is None or _processor is None:
        print("Loading CLIP model from HuggingFace...")
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _model.eval()
        print("CLIP model loaded successfully!")
    
    return _model, _processor


def generate_embedding(image_input: Union[str, Path, Image.Image]) -> np.ndarray:
    """
    Generate a normalized 512-dimensional CLIP image embedding.
    
    Args:
        image_input: Can be:
            - str or Path: Path to an image file
            - PIL.Image.Image: A PIL Image object
    
    Returns:
        np.ndarray: Normalized embedding vector of shape (1, 512) with dtype float32
    
    Raises:
        FileNotFoundError: If image file path doesn't exist
        ValueError: If image cannot be processed
    
    Example:
        >>> embedding = generate_embedding("path/to/image.jpg")
        >>> print(embedding.shape)  # (1, 512)
        >>> print(embedding.dtype)  # float32
    """
    # Load model and processor
    model, processor = load_clip_model()
    
    # Handle different input types
    if isinstance(image_input, (str, Path)):
        # Load image from file path
        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path).convert('RGB')
    elif isinstance(image_input, Image.Image):
        # Use PIL Image directly
        image = image_input.convert('RGB')
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    # Generate embedding
    with torch.no_grad():
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        
        # Get image features from CLIP vision encoder
        image_features = model.get_image_features(**inputs)
        
        # Normalize the embeddings (L2 normalization)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Convert to numpy array
        embedding = image_features.cpu().numpy().astype('float32')
    
    return embedding


def generate_embeddings_batch(image_paths: list[Union[str, Path]]) -> np.ndarray:
    """
    Generate embeddings for multiple images in batch.
    
    Args:
        image_paths: List of image file paths
    
    Returns:
        np.ndarray: Embedding matrix of shape (n_images, 512) with dtype float32
    
    Example:
        >>> embeddings = generate_embeddings_batch(["img1.jpg", "img2.jpg", "img3.jpg"])
        >>> print(embeddings.shape)  # (3, 512)
    """
    model, processor = load_clip_model()
    
    # Load and process all images
    images = []
    for path in image_paths:
        image_path = Path(path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path).convert('RGB')
        images.append(image)
    
    # Process batch
    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt")
        image_features = model.get_image_features(**inputs)
        
        # Normalize the embeddings
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Convert to numpy array
        embeddings = image_features.cpu().numpy().astype('float32')
    
    return embeddings


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python embeddings.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    try:
        embedding = generate_embedding(image_path)
        print(f"Generated embedding for: {image_path}")
        print(f"Shape: {embedding.shape}")
        print(f"Dtype: {embedding.dtype}")
        print(f"Norm: {np.linalg.norm(embedding)}")  # Should be ~1.0 for normalized vectors
        print(f"First 10 values: {embedding[0][:10]}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)




