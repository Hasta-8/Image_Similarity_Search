"""
Test client for /search-image endpoint

Demonstrates how to POST an image file to the FastAPI backend and receive
similarity search results.

Usage:
    1. Start the backend server: uvicorn backend.main:app --reload
    2. Run this script: python backend/test_client.py <image_path>

Example:
    python backend/test_client.py test_image.jpg
"""

import sys
import json
import requests
from pathlib import Path


API_URL = "http://localhost:8000/search-image"


def test_search_image(image_path: str):
    """
    Test the /search-image endpoint by posting an image file.
    
    Args:
        image_path: Path to the image file to search
    """
    # Validate image file exists
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"Error: Image file not found: {image_path}")
        return
    
    print("=" * 60)
    print("Testing /search-image Endpoint")
    print("=" * 60)
    print(f"Image file: {image_path}")
    print(f"API URL: {API_URL}")
    print()
    
    try:
        # Open image file and prepare for multipart/form-data upload
        with open(image_file, 'rb') as f:
            files = {'file': (image_file.name, f, 'image/jpeg')}
            
            print("Sending POST request...")
            response = requests.post(API_URL, files=files)
        
        # Check response status
        print(f"HTTP Status: {response.status_code}")
        print()
        
        if response.status_code == 200:
            # Parse and pretty-print JSON response
            results = response.json()
            
            print("=" * 60)
            print("Search Results")
            print("=" * 60)
            print(json.dumps(results, indent=2))
            print()
            
            # Print formatted results
            if 'results' in results and results['results']:
                print("=" * 60)
                print("Formatted Results")
                print("=" * 60)
                for i, result in enumerate(results['results'], 1):
                    print(f"\nResult {i}:")
                    print(f"  ID:     {result.get('id', 'N/A')}")
                    print(f"  Score:  {result.get('score', 'N/A'):.4f}")
                    print(f"  Path:   {result.get('path', 'N/A')}")
            else:
                print("No results found.")
        else:
            # Print error response
            print("Error Response:")
            try:
                error_data = response.json()
                print(json.dumps(error_data, indent=2))
            except:
                print(response.text)
    
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API.")
        print("Make sure the backend server is running:")
        print("  uvicorn backend.main:app --reload")
    except Exception as e:
        print(f"Error: {e}")


def print_expected_response():
    """Print example of expected response structure."""
    example = {
        "results": [
            {
                "id": 0,
                "score": 0.9234,
                "path": "/absolute/path/to/image1.jpg"
            },
            {
                "id": 5,
                "score": 0.8765,
                "path": "/absolute/path/to/image2.jpg"
            },
            {
                "id": 12,
                "score": 0.8123,
                "path": "/absolute/path/to/image3.jpg"
            }
        ]
    }
    
    print("=" * 60)
    print("Expected Response Structure")
    print("=" * 60)
    print(json.dumps(example, indent=2))
    print()
    print("Fields:")
    print("  - id: Integer ID of the matched image (from FAISS index)")
    print("  - score: Cosine similarity score (0.0 to 1.0, higher is more similar)")
    print("  - path: File path of the matched image (from metadata.json)")
    print("=" * 60)
    print()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python backend/test_client.py <image_path>")
        print()
        print_expected_response()
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_search_image(image_path)


if __name__ == "__main__":
    main()




