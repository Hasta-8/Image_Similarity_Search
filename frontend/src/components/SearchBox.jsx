import React, { useState } from 'react';

/**
 * SearchBox component for image file selection and search trigger.
 * 
 * @param {Object} props
 * @param {Function} props.onSearch - Callback function called with selected File when search is triggered
 * @param {boolean} [props.disabled=false] - Whether the button should be disabled (e.g., during loading)
 */
function SearchBox({ onSearch, disabled = false }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [showWarning, setShowWarning] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    setSelectedFile(file || null);
    setShowWarning(false); // Clear warning when new file is selected
  };

  const handleSearch = () => {
    if (!selectedFile) {
      setShowWarning(true);
      return;
    }

    // Disable button during submission (parent handles loading state)
    onSearch(selectedFile);
  };

  return (
    <div style={{ textAlign: 'center', marginBottom: '20px' }}>
      <div style={{ marginBottom: '10px' }}>
        <label htmlFor="image-upload" style={{ display: 'block', marginBottom: '10px' }}>
          Select an image to search:
        </label>
        <input
          id="image-upload"
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          disabled={disabled}
          aria-label="Upload image file for similarity search"
        />
      </div>

      {showWarning && (
        <div style={{ color: '#dc3545', fontSize: '14px', marginBottom: '10px' }}>
          Please select an image file first.
        </div>
      )}

      <button
        className="button"
        onClick={handleSearch}
        disabled={disabled}
        aria-label="Search for similar images"
      >
        Search Similar Images
      </button>
    </div>
  );
}

export default SearchBox;



