// Example code for integrating the backend API with the Next.js frontend
// This file is for reference only and should be adapted to your frontend implementation

// Function to upscale an image
export async function upscaleImage(imageFile, options = {}) {
  const {
    scaleFactor = '4',
    outscale = '4.0',
    faceEnhance = false
  } = options;

  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('scale_factor', scaleFactor);
  formData.append('outscale', outscale);
  formData.append('face_enhance', faceEnhance.toString());
  
  try {
    const response = await fetch('http://localhost:8000/upscale', {
      method: 'POST',
      body: formData,
    });
    
    const data = await response.json();
    if (data.success) {
      // Return the base64-encoded image data
      return {
        success: true,
        imageData: `data:image/png;base64,${data.image}`,
        message: data.message
      };
    } else {
      return {
        success: false,
        error: data.error || 'Unknown error occurred'
      };
    }
  } catch (error) {
    console.error('Error upscaling image:', error);
    return {
      success: false,
      error: error.message || 'Failed to connect to the server'
    };
  }
}

// Function to outpaint an image
export async function outpaintImage(imageFile, options = {}) {
  const {
    scaleFactor = '4',
    outscale = '1.0',
    padding = '64'
  } = options;

  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('scale_factor', scaleFactor);
  formData.append('outscale', outscale);
  formData.append('padding', padding);
  
  try {
    const response = await fetch('http://localhost:8000/outpaint', {
      method: 'POST',
      body: formData,
    });
    
    const data = await response.json();
    if (data.success) {
      // Return the base64-encoded image data
      return {
        success: true,
        imageData: `data:image/png;base64,${data.image}`,
        message: data.message
      };
    } else {
      return {
        success: false,
        error: data.error || 'Unknown error occurred'
      };
    }
  } catch (error) {
    console.error('Error outpainting image:', error);
    return {
      success: false,
      error: error.message || 'Failed to connect to the server'
    };
  }
}

// Example usage in a React component
/*
import { useState } from 'react';
import { upscaleImage, outpaintImage } from './api';

export default function ImageProcessor() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setProcessedImage(null);
    setError(null);
  };

  const handleUpscale = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await upscaleImage(selectedFile, {
        scaleFactor: '4',
        outscale: '4.0',
        faceEnhance: false
      });

      if (result.success) {
        setProcessedImage(result.imageData);
      } else {
        setError(result.error);
      }
    } catch (err) {
      setError('Failed to process image');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleOutpaint = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await outpaintImage(selectedFile, {
        scaleFactor: '4',
        outscale: '1.0',
        padding: '64'
      });

      if (result.success) {
        setProcessedImage(result.imageData);
      } else {
        setError(result.error);
      }
    } catch (err) {
      setError('Failed to process image');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      <h1>Image Processor</h1>
      <input type="file" accept="image/*" onChange={handleFileChange} />
      
      <div>
        <button onClick={handleUpscale} disabled={isLoading || !selectedFile}>
          {isLoading ? 'Processing...' : 'Upscale Image'}
        </button>
        <button onClick={handleOutpaint} disabled={isLoading || !selectedFile}>
          {isLoading ? 'Processing...' : 'Outpaint Image'}
        </button>
      </div>
      
      {error && <p style={{ color: 'red' }}>{error}</p>}
      
      {selectedFile && (
        <div>
          <h2>Original Image</h2>
          <img 
            src={URL.createObjectURL(selectedFile)} 
            alt="Original" 
            style={{ maxWidth: '100%', maxHeight: '300px' }} 
          />
        </div>
      )}
      
      {processedImage && (
        <div>
          <h2>Processed Image</h2>
          <img 
            src={processedImage} 
            alt="Processed" 
            style={{ maxWidth: '100%', maxHeight: '500px' }} 
          />
        </div>
      )}
    </div>
  );
}
*/