/**
 * Centralized API URL handling for Create React App.
 *
 * Leave REACT_APP_API_BASE_URL empty for same-origin requests. CRA's
 * development proxy forwards relative API requests to localhost:8000; a
 * production reverse proxy should route the same paths to FastAPI.
 */
export function getApiBaseUrl(environment = process.env) {
  return (environment.REACT_APP_API_BASE_URL || '').trim().replace(/\/+$/, '');
}

export const API_BASE_URL = getApiBaseUrl();

function withLeadingSlash(path) {
  return path.startsWith('/') ? path : `/${path}`;
}

export function buildApiUrl(path, backendBase = API_BASE_URL) {
  const base = (backendBase || '').replace(/\/+$/, '');
  return `${base}${withLeadingSlash(path)}`;
}

function isAbsoluteUrl(value) {
  return /^[a-z][a-z\d+.-]*:/i.test(value) || value.startsWith('//');
}

function encodePath(path) {
  return path
    .split('/')
    .map((segment) => {
      try {
        return encodeURIComponent(decodeURIComponent(segment));
      } catch {
        return encodeURIComponent(segment);
      }
    })
    .join('/');
}

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
    const response = await fetch(buildApiUrl('/search-image'), {
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
  if (!result) return null;

  const imageUrl = result.image_url;
  if (imageUrl) {
    if (isAbsoluteUrl(imageUrl) || imageUrl.startsWith('data:')) return imageUrl;

    const base = (backendBase || '').replace(/\/+$/, '');
    if (base && isAbsoluteUrl(base)) {
      return new URL(imageUrl, `${base}/`).toString();
    }
    // Keep backend-provided relative URLs relative for same-origin proxying.
    return imageUrl;
  }

  if (result.path && isAbsoluteUrl(result.path)) return result.path;
  if (result.path && result.path.startsWith('data:')) return result.path;
  if (result.b64) return `data:image/jpeg;base64,${result.b64}`;

  if (!result.path) return null;
  const relativePath = encodePath(withLeadingSlash(result.path));
  const base = (backendBase || '').replace(/\/+$/, '');
  if (base && isAbsoluteUrl(base)) {
    return new URL(relativePath, `${base}/`).toString();
  }
  return relativePath;
}
