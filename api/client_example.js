/**
 * PlaybookTV Interior Design AI - Client Example
 * Use this in your Modomo app to interact with the API
 */

// Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const API_KEY = process.env.REACT_APP_API_KEY; // Optional: if you add authentication

/**
 * Analyze interior design image
 * @param {File} imageFile - Image file to analyze
 * @returns {Promise<Object>} Analysis results
 */
export async function analyzeInteriorImage(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);

  const headers = {};
  if (API_KEY) {
    headers['X-API-Key'] = API_KEY;
  }

  try {
    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      body: formData,
      headers
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Error analyzing image:', error);
    throw error;
  }
}

/**
 * Detect objects only (faster)
 * @param {File} imageFile - Image file
 * @returns {Promise<Object>} Detection results
 */
export async function detectObjects(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);

  const response = await fetch(`${API_BASE_URL}/detect`, {
    method: 'POST',
    body: formData
  });

  return await response.json();
}

/**
 * Classify style only
 * @param {File} imageFile - Image file
 * @returns {Promise<Object>} Style classification
 */
export async function classifyStyle(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);

  const response = await fetch(`${API_BASE_URL}/classify/style`, {
    method: 'POST',
    body: formData
  });

  return await response.json();
}

/**
 * Check API health
 * @returns {Promise<Object>} Health status
 */
export async function checkHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  return await response.json();
}

/**
 * Get model information
 * @returns {Promise<Object>} Model info
 */
export async function getModelInfo() {
  const response = await fetch(`${API_BASE_URL}/models/info`);
  return await response.json();
}

// ============================================
// REACT COMPONENT EXAMPLE
// ============================================

/**
 * Example React component using the API
 */
import React, { useState } from 'react';

function InteriorAnalyzer() {
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setAnalyzing(true);
    setError(null);

    try {
      const result = await analyzeInteriorImage(file);
      setResults(result);
      console.log('Analysis complete:', result);
    } catch (err) {
      setError(err.message);
      console.error('Analysis failed:', err);
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="interior-analyzer">
      <h2>Interior Design Analysis</h2>

      {/* File upload */}
      <input
        type="file"
        accept="image/*"
        onChange={handleImageUpload}
        disabled={analyzing}
      />

      {/* Loading state */}
      {analyzing && (
        <div className="loading">
          <p>Analyzing image...</p>
          <div className="spinner"></div>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="error">
          <p>Error: {error}</p>
        </div>
      )}

      {/* Results */}
      {results && (
        <div className="results">
          <h3>Analysis Results</h3>

          {/* Style */}
          <div className="style-result">
            <h4>Design Style</h4>
            <p>
              <strong>{results.style.style}</strong>
              ({(results.style.confidence * 100).toFixed(1)}% confidence)
            </p>

            {/* All style probabilities */}
            <div className="style-probabilities">
              {Object.entries(results.style.all_probabilities)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 5)
                .map(([style, prob]) => (
                  <div key={style} className="prob-bar">
                    <span>{style}</span>
                    <div className="bar">
                      <div
                        className="fill"
                        style={{width: `${prob * 100}%`}}
                      ></div>
                    </div>
                    <span>{(prob * 100).toFixed(1)}%</span>
                  </div>
                ))}
            </div>
          </div>

          {/* Detected objects */}
          <div className="detections">
            <h4>Detected Items ({results.detection_count})</h4>
            <ul>
              {results.detections.map((detection, idx) => (
                <li key={idx}>
                  <strong>{detection.item_type}</strong>
                  <span> - {(detection.confidence * 100).toFixed(1)}% confidence</span>
                  <span> - {detection.area_percentage.toFixed(1)}% of image</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Processing time */}
          <p className="processing-time">
            Processed in {results.processing_time_ms.toFixed(0)}ms
          </p>
        </div>
      )}
    </div>
  );
}

export default InteriorAnalyzer;

// ============================================
// VANILLA JAVASCRIPT EXAMPLE
// ============================================

/**
 * Simple vanilla JS example
 */
function setupInteriorAnalyzer() {
  const fileInput = document.getElementById('image-input');
  const resultsDiv = document.getElementById('results');

  fileInput.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Show loading
    resultsDiv.innerHTML = '<p>Analyzing...</p>';

    try {
      const result = await analyzeInteriorImage(file);

      // Display results
      resultsDiv.innerHTML = `
        <h3>Style: ${result.style.style}</h3>
        <p>Confidence: ${(result.style.confidence * 100).toFixed(1)}%</p>

        <h4>Detected Items (${result.detection_count}):</h4>
        <ul>
          ${result.detections.map(d => `
            <li>${d.item_type} (${(d.confidence * 100).toFixed(1)}%)</li>
          `).join('')}
        </ul>
      `;
    } catch (error) {
      resultsDiv.innerHTML = `<p class="error">Error: ${error.message}</p>`;
    }
  });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', setupInteriorAnalyzer);

// ============================================
// TYPESCRIPT VERSION
// ============================================

/**
 * TypeScript interfaces for type safety
 */

interface Detection {
  item_type: string;
  confidence: number;
  bbox: [number, number, number, number];
  area_percentage: number;
}

interface StylePrediction {
  style: string;
  confidence: number;
  all_probabilities: Record<string, number>;
}

interface AnalysisResult {
  detections: Detection[];
  detection_count: number;
  style: StylePrediction;
  processing_time_ms: number;
}

export async function analyzeInteriorImageTS(
  imageFile: File
): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append('file', imageFile);

  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return await response.json();
}
