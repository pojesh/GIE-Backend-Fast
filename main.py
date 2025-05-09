import os
import sys
import warnings
import cv2
import numpy as np
import torch
import base64
import uuid
from pathlib import Path
from PIL import Image
import io

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --- Compatibility and Path Fixes ---
warnings.filterwarnings("ignore")

# Patch for torchvision compatibility issue
import types
sys.modules['torchvision.transforms.functional_tensor'] = types.SimpleNamespace(
    rgb_to_grayscale=lambda x: x.mean(dim=1, keepdim=True)
)

# Add Real-ESRGAN to path
REALESRGAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Real-ESRGAN-master')
if REALESRGAN_DIR not in sys.path:
    sys.path.append(REALESRGAN_DIR)

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
    from realesrgan.utils import RealESRGANer
except Exception as e:
    print("Error importing Real-ESRGAN modules:", e)
    raise

# --- FastAPI App Setup ---
app = FastAPI(
    title="Galaxy Image Enhancer API",
    description="API for upscaling and outpainting images using Real-ESRGAN and SDXL.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Directories ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# --- Model Initialization ---
upsampler_x4 = None
upsampler_x2 = None

def initialize_model(model_name):
    """Initialize the Real-ESRGAN model"""
    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        scale = 4
        model_path = os.path.join(REALESRGAN_DIR, 'weights', 'RealESRGAN_x4plus.pth')
    elif model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        scale = 2
        model_path = os.path.join(REALESRGAN_DIR, 'weights', 'RealESRGAN_x2plus.pth')
    else:
        raise ValueError(f'Model {model_name} not supported')

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f'Model file {model_path} not found. Please download it first.')

    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=400,  # Use tile to avoid CUDA OOM
        tile_pad=10,
        pre_pad=0,
        half=False  # Use full precision for compatibility
    )
    return upsampler

@app.on_event("startup")
def load_models():
    global upsampler_x4, upsampler_x2
    try:
        upsampler_x4 = initialize_model('RealESRGAN_x4plus')
        upsampler_x2 = initialize_model('RealESRGAN_x2plus')
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please make sure the model weights are downloaded and placed in the correct directory.")
        print("You can download them from:")
        print("- RealESRGAN_x4plus.pth: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth")
        print("- RealESRGAN_x2plus.pth: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth")
        upsampler_x4 = None
        upsampler_x2 = None

@app.get("/", tags=["General"])
async def read_root():
    return {"message": "Welcome to the Galaxy Image Enhancer API"}

@app.get("/health", tags=["General"])
async def health_check():
    global upsampler_x4, upsampler_x2
    return {
        'status': 'ok',
        'models_loaded': {
            'x4plus': upsampler_x4 is not None,
            'x2plus': upsampler_x2 is not None
        }
    }

@app.post("/upscale", tags=["Image Processing"])
async def upscale_image_api(
    image: UploadFile = File(...),
    scale_factor: str = Form("4"),
    outscale: float = Form(4.0),
    face_enhance: str = Form("false")
):
    global upsampler_x4, upsampler_x2
    if upsampler_x4 is None or upsampler_x2 is None:
        return JSONResponse(content={'error': 'Models not loaded. Please check server logs.'}, status_code=500)

    if not image.filename:
        return JSONResponse(content={'error': 'No image selected'}, status_code=400)

    try:
        # Save the uploaded image
        img_uuid = str(uuid.uuid4())
        input_path = os.path.join(UPLOAD_FOLDER, f"{img_uuid}_input.png")
        output_path = os.path.join(RESULT_FOLDER, f"{img_uuid}_output.png")

        img = Image.open(io.BytesIO(await image.read()))
        img.save(input_path)

        img_cv = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img_cv is None:
            return JSONResponse(content={'error': 'Failed to read image'}, status_code=500)

        upsampler = upsampler_x4 if scale_factor == '4' else upsampler_x2

        # Process the image
        output, _ = upsampler.enhance(img_cv, outscale=outscale)

        # Save the output image
        cv2.imwrite(output_path, output)

        # Return the result image as base64
        with open(output_path, 'rb') as f:
            img_data = f.read()
            encoded_img = base64.b64encode(img_data).decode('utf-8')

        return JSONResponse(content={
            'success': True,
            'image': encoded_img,
            'message': 'Image upscaled successfully'
        })
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)
    finally:
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                os.remove(output_path)
        except Exception:
            pass

@app.post("/outpaint", tags=["Image Processing"])
async def outpaint_image_api(
    image: UploadFile = File(...),
    scale_factor: str = Form("4"),
    outscale: float = Form(1.0),
    padding: int = Form(64)
):
    global upsampler_x4, upsampler_x2
    if upsampler_x4 is None or upsampler_x2 is None:
        return JSONResponse(content={'error': 'Models not loaded. Please check server logs.'}, status_code=500)

    if not image.filename:
        return JSONResponse(content={'error': 'No image selected'}, status_code=400)

    try:
        img_uuid = str(uuid.uuid4())
        input_path = os.path.join(UPLOAD_FOLDER, f"{img_uuid}_input.png")
        padded_input_path = os.path.join(UPLOAD_FOLDER, f"{img_uuid}_padded_input.png")
        output_path = os.path.join(RESULT_FOLDER, f"{img_uuid}_output.png")

        img = Image.open(io.BytesIO(await image.read()))
        img.save(input_path)

        img_cv = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img_cv is None:
            return JSONResponse(content={'error': 'Failed to read image'}, status_code=500)

        # Add padding to the image (outpainting)
        h, w = img_cv.shape[:2]
        padded_img_cv = cv2.copyMakeBorder(img_cv, padding, padding, padding, padding, cv2.BORDER_REFLECT)
        cv2.imwrite(padded_input_path, padded_img_cv)

        upsampler = upsampler_x4 if scale_factor == '4' else upsampler_x2

        # Process the padded image
        output, _ = upsampler.enhance(padded_img_cv, outscale=outscale)
        cv2.imwrite(output_path, output)

        with open(output_path, 'rb') as f:
            img_data = f.read()
            encoded_img = base64.b64encode(img_data).decode('utf-8')

        return JSONResponse(content={
            'success': True,
            'image': encoded_img,
            'message': 'Image outpainted and processed successfully'
        })
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)
    finally:
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(padded_input_path):
                os.remove(padded_input_path)
            if os.path.exists(output_path):
                os.remove(output_path)
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)