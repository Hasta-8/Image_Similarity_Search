#!/bin/bash

# Test script for /search-image endpoint using curl
# 
# Usage:
#   1. Start the backend server: uvicorn backend.main:app --reload
#   2. Run this script: bash backend/test_search.sh <image_path>
#
# Example:
#   bash backend/test_search.sh test_image.jpg

set -e

# Default image path (you can override by passing as argument)
IMAGE_PATH="${1:-test_image.jpg}"

# API endpoint
API_URL="http://localhost:8000/search-image"

# Check if image file exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: Image file not found: $IMAGE_PATH"
    echo "Usage: bash backend/test_search.sh <image_path>"
    exit 1
fi

echo "Testing /search-image endpoint..."
echo "Image file: $IMAGE_PATH"
echo "API URL: $API_URL"
echo ""

# Make POST request with curl
echo "Sending POST request..."
curl -X POST "$API_URL" \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@$IMAGE_PATH" \
    -w "\n\nHTTP Status: %{http_code}\n" \
    | python3 -m json.tool 2>/dev/null || cat

echo ""
echo "Done!"




