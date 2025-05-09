# Image Upscaling and Outpainting Backend API

This is the backend API for the Image Upscaling and Outpainting webapp. It uses Real-ESRGAN models to enhance and outpaint images.

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (recommended for faster processing)

### Installation

1. Clone this repository.
2. Ensure you have Python 3.7+ installed.
3. It is recommended to use a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Download the model weights. The application will look for them in `backend/weights/` first, and then in `backend/Real-ESRGAN-master/weights/`. Create the appropriate directory and place the `.pth` files there:
   - [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)
   - [RealESRGAN_x2plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth)

### Running the API

Navigate to the `backend` directory and run the FastAPI application using Uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### Health Check

```
GET /health
```

Returns the status of the API and whether the models are loaded.

### Upscale Image

```
POST /upscale
```

Parameters (form-data):
- `image`: The image file to upscale
- `scale_factor`: The scale factor to use (2 or 4, default: 4)
- `outscale`: The final output scale (default: 4.0)
- `face_enhance`: Whether to enhance faces (true/false, default: false)

### Outpaint Image

```
POST /outpaint
```

Parameters (form-data):
- `image`: The image file to outpaint
- `scale_factor`: The scale factor to use (2 or 4, default: 4)
- `outscale`: The final output scale (default: 1.0)
- `padding`: The padding to add around the image in pixels (default: 64)

## Integration with Frontend

The API is CORS-enabled, so it can be called from any frontend application. The frontend should send requests to the API endpoints and handle the responses accordingly.

Example frontend code (JavaScript):

```javascript
async function upscaleImage(imageFile) {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('scale_factor', '4');
  formData.append('outscale', '4.0');
  
  const response = await fetch('http://localhost:8000/upscale', {
    method: 'POST',
    body: formData,
  });
  
  const data = await response.json();
  if (data.success) {
    // data.image contains the base64-encoded upscaled image
    return `data:image/png;base64,${data.image}`;
  } else {
    throw new Error(data.error);
  }
}
```