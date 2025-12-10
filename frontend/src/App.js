import React, { useState } from 'react';
import { searchSimilarImages } from './api';
import SearchBox from './components/SearchBox';
import './index.css';

function App() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileSubmit = async (file) => {
    if (!file) return;

    // Clear previous errors and results
    setError('');
    setLoading(true);
    setResults([]);

    try {
      const response = await searchSimilarImages(file);
      setResults(response.results || []);
    } catch (err) {
      setError(err.message || 'Failed to search images');
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header style={{ textAlign: 'center', marginBottom: '30px' }}>
        <h1>Image Similarity Search (MVP)</h1>
      </header>

      {/* SearchBox component */}
      <SearchBox onSearch={handleFileSubmit} disabled={loading} />

      {/* Loading spinner */}
      {loading && (
        <div style={{ textAlign: 'center', margin: '20px 0' }}>
          <div className="loading" aria-label="Loading search results"></div>
        </div>
      )}

      {/* Results section */}
      {!loading && results.length > 0 && (
        <section>
          {/* Error message at top of results */}
          {error && (
            <div style={{ color: '#dc3545', marginBottom: '20px', padding: '10px', backgroundColor: '#f8d7da', border: '1px solid #f5c6cb', borderRadius: '4px' }} role="alert">
              {error}
            </div>
          )}
          <h2 style={{ marginBottom: '10px' }}>Results</h2>
          <p style={{ marginBottom: '20px', color: '#666' }}>
            {results.length} {results.length === 1 ? 'match' : 'matches'} found
          </p>
          <div className="results-grid">
            {results.map((result) => (
              <div key={result.id} style={{ textAlign: 'center' }}>
                {result.image_url ? (
                  <img
                    src={result.image_url}
                    alt={`Match ${result.id}`}
                    className="thumbnail"
                    loading="lazy"
                  />
                ) : (
                  <div 
                    className="thumbnail" 
                    style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'center',
                      backgroundColor: '#f0f0f0',
                      color: '#666',
                      minHeight: '200px'
                    }}
                  >
                    <p>Image not available</p>
                  </div>
                )}
                <p style={{ marginTop: '10px', fontSize: '14px' }}>
                  Score: {result.score?.toFixed(3) || 'N/A'}
                </p>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Error message when no results */}
      {!loading && error && results.length === 0 && (
        <div className="error-message" role="alert">
          {error}
        </div>
      )}

      {/* No results message */}
      {!loading && !error && results.length === 0 && (
        <div style={{ textAlign: 'center', marginTop: '40px', color: '#666' }}>
          <p>No matches found. Upload an image to search.</p>
        </div>
      )}
    </div>
  );
}

export default App;
