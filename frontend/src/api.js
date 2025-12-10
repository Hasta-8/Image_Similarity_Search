/**
 * API client for image similarity search
 */

// Base URL for the FastAPI backend
export const API_BASE_URL = 'http://localhost:8000';

/**
 * Search for similar images by uploading an image file.
 * 
 * @param {File} file - Image file to search with
 * @returns {Promise<Object>} Parsed JSON response with search results
 * @throws {Error} If the request fails or response is not ok
 */
export async function searchSimilarImages(file) {
  // Build FormData object
  const formData = new FormData();
  formData.append('file', file); // Backend expects 'file' key
  
  try {
    // POST to search endpoint
    const response = await fetch(`${API_BASE_URL}/search-image`, {
      method: 'POST',
      body: formData
    });
    
    // Throw error if response is not ok
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Search failed: ${response.status} ${response.statusText}. ${errorText}`);
    }
    
    // Parse and return JSON
    return await response.json();
  } catch (err) {
    // Re-throw with network error message if it's a network error
    if (err instanceof TypeError && err.message.includes('fetch')) {
      throw new Error('Network or server error — check backend is running');
    }
    throw err;
  }
}

/**
 * Get image source URL for displaying result images.
 * 
 * @param {Object} result - Search result object with path, image_url, and optional b64 field
 * @param {string} backendBase - Base URL for backend (default: uses API_BASE_URL)
 * @returns {string} Image source URL or data URI
 */
export function getImageSrc(result, backendBase = API_BASE_URL) {
  // Prefer image_url if available (from backend response)
  if (result.image_url) {
    return result.image_url;
  }
  
  // If path is already a full URL or data URI, return it
  if (result.path && (result.path.startsWith('http') || result.path.startsWith('data:'))) {
    return result.path;
  }
  
  // If result has base64 data, return data URI
  if (result.b64) {
    return `data:image/jpeg;base64,${result.b64}`;
  }
  
  // Build URL from backend base and path
  const path = result.path || '';
  // Ensure path starts with slash
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${backendBase}${normalizedPath}`;
}
