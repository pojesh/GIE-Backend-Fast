import os
import sys
import warnings
import cv2
from cv2.detail import resultRoi
import numpy as np
import torch
import argparse
import datetime
import base64
import uuid
from pathlib import Path
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
import io
import torch
from diffusers import AutoencoderKL, TCDScheduler
from diffusers import (
    StableDiffusionInpaintPipeline,
    DDIMScheduler, 
    EulerAncestralDiscreteScheduler
)
from diffusers.models.model_loading_utils import load_state_dict
from huggingface_hub import hf_hub_download
import random
import torchvision.transforms as T

#from controlnet_union import ControlNetModel_Union
#from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel 

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import gc
from diffusers import DiffusionPipeline

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
    from realesrgan.utils import RealESRGANer
except Exception as e:
    print("Error importing Real-ESRGAN modules:", e)
    raise

# --- FastAPI App Setup ---
app = FastAPI(
    title="Galaxy Image Enhancer API",
    description="API for upscaling and outpainting images using Real-ESRGAN and SD.",
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

sd_pipe = None
sd_vae = None
sd_controlnet = None

def initiate_sd(device="cuda", model_version="2.0", use_optimized_vae=True):
        device = _setup_device(device)
        model_version = model_version
        sd_pipe = None
        use_optimized_vae = use_optimized_vae
        #self.processor, self.model = self._create_caption_augmentor()
        
        # Scene type detection keywords for better prompting
        scene_keywords = {
            'urban': ['street', 'road', 'car', 'building', 'city', 'traffic', 'sidewalk'],
            'nature': ['tree', 'grass', 'sky', 'field', 'forest', 'mountain', 'landscape'],
            'indoor': ['room', 'wall', 'furniture', 'interior', 'house'],
            'portrait': ['person', 'face', 'woman', 'man', 'people', 'human'],
            'animal': ['cat', 'dog', 'pet', 'animal', 'fur'],
        }
def _setup_device(preferred_device):
        """Setup and verify PyTorch device."""
        if preferred_device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = torch.device("cpu")
            print("⚠️  Using CPU - this will be significantly slower")
        return device

def initialize_model(model_name):
    """Initialize the Real-ESRGAN model"""
    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        scale = 4
        #model_path = os.path.join(REALESRGAN_DIR, 'weights', 'RealESRGAN_x4plus.pth')
        model_path = "weights/RealESRGAN_x4plus.pth"
    elif model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        scale = 2
        #model_path = os.path.join(REALESRGAN_DIR, 'weights', 'RealESRGAN_x2plus.pth')
        model_path = "weights/RealESRGAN_x2plus.pth"
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
        half=True  # Use full precision for compatibility
    )
    return upsampler



@app.on_event("startup")
def load_models():
    global upsampler_x4, upsampler_x2
    try:
        upsampler_x4 = initialize_model('RealESRGAN_x4plus')
        upsampler_x2 = initialize_model('RealESRGAN_x2plus')


        global sd_pipe
        model_id = "stabilityai/stable-diffusion-2-inpainting"
        print("🔄 Loading Stable Diffusion 2 Inpainting...")

        sd_vae = None

        try:
            sd_vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse", 
                torch_dtype=torch.float16
            )
            print("✅ Loaded optimized VAE")
        except Exception as e:
            print(f"⚠️  Failed to load optimized VAE: {e}")

        sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                vae=sd_vae,
                safety_checker=None,
                torch_dtype=torch.float16,
                variant="fp16"
            )

        sd_pipe.to("cuda")
        scheduler_type = "euler_a"
        _set_scheduler(scheduler_type)

        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please make sure the model weights are downloaded and placed in the correct directory.")
        print("You can download them from:")
        print("- RealESRGAN_x4plus.pth: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth")
        print("- RealESRGAN_x2plus.pth: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth")
        upsampler_x4 = None
        upsampler_x2 = None

        sd_pipe = None
        sd_vae = None


@app.get("/", tags=["General"])
async def read_root():
    return {"message": "Welcome to the Galaxy Image Enhancer API"}

@app.get("/health", tags=["General"])
async def health_check():
    global upsampler_x4, upsampler_x2, sd_pipe
    return {
        'status': 'ok',
        'models_loaded': {
            'x4plus': upsampler_x4 is not None,
            'x2plus': upsampler_x2 is not None,
            'sdxl_outpaint': sd_pipe is not None
        }
    }

@app.post("/upscale", tags=["Image Processing"])
async def upscale_image_api(
    image: UploadFile = File(...),
    scale_factor: str = Form("4"),
    outscale: float = Form(4.0)
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

 
def _set_scheduler(scheduler_type):
        """Set the desired scheduler."""
        try:
            if scheduler_type == "ddim":
                sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
                print("📅 Using DDIM scheduler")
            elif scheduler_type == "euler_a":
                sd_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(sd_pipe.scheduler.config)
                print("📅 Using Euler Ancestral scheduler")
        except Exception as e:
            print(f"⚠️  Error setting scheduler: {e}. Using default.")

def analyze_image_content(image):
        """Analyze image content to determine scene type and generate appropriate prompts."""
        # Convert to numpy for analysis
        img_array = np.array(image)
        
        # Simple color analysis
        avg_colors = np.mean(img_array.reshape(-1, 3), axis=0)
        brightness = np.mean(avg_colors)
        
        # Detect dominant colors
        green_dominance = avg_colors[1] > avg_colors[0] and avg_colors[1] > avg_colors[2]
        blue_dominance = avg_colors[2] > avg_colors[0] and avg_colors[2] > avg_colors[1]
        
        # Basic scene classification
        scene_type = "general"
        if green_dominance and brightness > 100:
            scene_type = "nature"
        elif blue_dominance:
            scene_type = "sky"
        elif brightness < 80:
            scene_type = "indoor"
        
        return scene_type, brightness, avg_colors


def generate_smart_prompt(image, custom_prompt=""):
        """Generate intelligent prompts based on image analysis."""
        if custom_prompt.strip():
            return custom_prompt

        #caption = self.get_image_caption(image)
        

        scene_type, brightness, avg_colors = analyze_image_content(image)
        
        # Base quality terms
        base_prompt = "photorealistic, high quality, detailed, 4k resolution, professional photography"
        
        # Scene-specific prompts
        scene_prompts = {
            "nature": "natural landscape, trees, grass, sky, outdoor scenery, environmental continuity",
            "sky": "clear sky, clouds, natural lighting, atmospheric perspective, horizon",
            "indoor": "consistent lighting",
            "general": "seamless extension, consistent style, natural continuation"
        }
        
        # Lighting prompts based on brightness
        if brightness > 150:
            lighting = "bright lighting, daylight, well-lit"
        elif brightness < 80:
            lighting = "ambient lighting, soft shadows, moody atmosphere"
        else:
            lighting = "natural lighting, balanced exposure"
        
        final_prompt = f"{base_prompt}, {scene_prompts.get(scene_type, scene_prompts['general'])}, {lighting}, seamless blending"
        #final_prompt += ", " + caption  # Add caption to the end for more context
        print(f"🎨 Generated prompt for {scene_type} scene: {final_prompt}...")
        return final_prompt

def create_seamless_canvas(image, target_width, target_height):
        """Create canvas with intelligent padding for seamless outpainting."""
        original_width, original_height = image.size
        
        # Calculate optimal scaling
        scale_w = target_width / original_width
        scale_h = target_height / original_height
        scale = min(scale_w, scale_h) * 0.8  # Leave room for outpainting
        
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Ensure minimum size for the original content
        min_dimension = min(target_width, target_height) // 2
        if new_width < min_dimension or new_height < min_dimension:
            scale = min_dimension / min(original_width, original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
        
        # Resize image with high quality
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Center positioning
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        # Create canvas with neutral background
        canvas = Image.new("RGB", (target_width, target_height), (128, 128, 128))
        
        # Use edge extension for better seamless results
        canvas = _create_edge_extended_canvas(
            resized_image, canvas, x_offset, y_offset, target_width, target_height
        )
        
        return canvas, (x_offset, y_offset), (new_width, new_height)

def _create_edge_extended_canvas(image, canvas, x_offset, y_offset, target_width, target_height):
        """Create canvas with intelligent edge extension."""
        img_width, img_height = image.size
        
        # Create extended versions of edges
        edge_size = 50  # Pixels to extend
        
        # Top edge extension
        if y_offset > 0:
            top_strip = image.crop((0, 0, img_width, min(edge_size, img_height//4)))
            top_extended = top_strip.resize((target_width, y_offset), Image.LANCZOS)
            canvas.paste(top_extended, (0, 0))
        
        # Bottom edge extension
        bottom_start = y_offset + img_height
        if bottom_start < target_height:
            bottom_strip = image.crop((0, max(0, img_height-edge_size), img_width, img_height))
            bottom_extended = bottom_strip.resize((target_width, target_height - bottom_start), Image.LANCZOS)
            canvas.paste(bottom_extended, (0, bottom_start))
        
        # Left edge extension
        if x_offset > 0:
            left_strip = image.crop((0, 0, min(edge_size, img_width//4), img_height))
            left_extended = left_strip.resize((x_offset, img_height), Image.LANCZOS)
            canvas.paste(left_extended, (0, y_offset))
        
        # Right edge extension
        right_start = x_offset + img_width
        if right_start < target_width:
            right_strip = image.crop((max(0, img_width-edge_size), 0, img_width, img_height))
            right_extended = right_strip.resize((target_width - right_start, img_height), Image.LANCZOS)
            canvas.paste(right_extended, (right_start, y_offset))
        
        # Paste the original image on top
        canvas.paste(image, (x_offset, y_offset))
        
        return canvas

def ensure_odd_kernel_size(size):
    """Ensure kernel size is odd and at least 3."""
    size = max(3, int(size))
    return size if size % 2 == 1 else size + 1

def create_professional_mask(canvas_size, content_offset, content_size, feather_radius=32):
    """Create a professional mask with gradient transitions - FIXED VERSION."""
    width, height = canvas_size
    x_offset, y_offset = content_offset
    content_width, content_height = content_size
    
    # Create mask with gradient
    mask = np.ones((height, width), dtype=np.float32) * 255
    
    # Create content preservation area
    content_mask = np.zeros((content_height, content_width), dtype=np.float32)
    
    # Add gradient borders to content mask
    border_size = min(feather_radius, content_width//8, content_height//8)
    
    # Create gradient borders
    for i in range(border_size):
        alpha = i / border_size
        # Top border
        if i < content_height:
            content_mask[i, :] = alpha * 255
        # Bottom border  
        if content_height - 1 - i >= 0:
            content_mask[content_height - 1 - i, :] = alpha * 255
        # Left border
        if i < content_width:
            content_mask[:, i] = np.maximum(content_mask[:, i], alpha * 255)
        # Right border
        if content_width - 1 - i >= 0:
            content_mask[:, content_width - 1 - i] = np.maximum(content_mask[:, content_width - 1 - i], alpha * 255)
    
    # Place content mask on main mask
    mask[y_offset:y_offset+content_height, x_offset:x_offset+content_width] = content_mask
    
    # Apply Gaussian blur for smooth transitions - FIXED KERNEL SIZE
    kernel_size = max(3, feather_radius * 2 + 1)
    if kernel_size % 2 == 0:  # Ensure odd kernel size
        kernel_size += 1
    
    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), max(1.0, feather_radius/3))
    
    # Convert back to PIL
    mask_pil = Image.fromarray(mask.astype(np.uint8), mode='L')
    
    return mask_pil

def generate_outpainting(image, mask, prompt="", negative_prompt="", 
                           guidance_scale=7.5, steps=50, seed=None):
        """Perform high-quality outpainting generation."""
        try:
            # Set up generator with seed
            generator = None
            if seed is not None:
                generator = torch.Generator(device="cuda").manual_seed(seed)
                print(f"🎲 Using seed: {seed}")
            
            # Generate smart prompt if not provided
            if not prompt.strip():
                prompt = generate_smart_prompt(image)
            
            # Enhanced negative prompt
            if not negative_prompt.strip():
                negative_prompt = (
                    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, "
                    "cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, "
                    "username, blurry, duplicate, artifacts, seams, visible boundaries, inconsistent lighting, "
                    "unnatural colors, distorted perspective, repetitive patterns, tiling artifacts"
                )
            
            print(f"🎨 Generating with {steps} steps, guidance scale {guidance_scale}")
            print(f"📝 Prompt: {prompt}...")
            
            # Generate with higher quality settings
            result = sd_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                height=image.height,
                width=image.width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                strength=0.95,  # Higher strength for better outpainting
            ).images[0]
            
            return result
            
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            return None

def post_process_result(result, original_canvas):
        """Apply post-processing to improve result quality."""
        try:
            # Convert to numpy for processing
            result_array = np.array(result)
            canvas_array = np.array(original_canvas)
            
            # Enhance contrast and saturation slightly
            result_enhanced = Image.fromarray(result_array)
            
            # Slight contrast enhancement
            enhancer = ImageEnhance.Contrast(result_enhanced)
            result_enhanced = enhancer.enhance(1.1)
            
            # Slight saturation enhancement
            enhancer = ImageEnhance.Color(result_enhanced)
            result_enhanced = enhancer.enhance(1.05)
            
            return result_enhanced
            
        except Exception as e:
            print(f"⚠️  Post-processing failed: {e}")
            return result

def save_result(image, output_dir="outputs", prefix="outpainted"):
        """Save the result with timestamp."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.png"
            output_file = output_path / filename
            
            # Save with maximum quality
            image.save(output_file, format="PNG", optimize=False, compress_level=1)
            print(f"💾 Saved result: {output_file}")
            return str(output_file)
        except Exception as e:
            print(f"❌ Error saving image: {e}")
            return None

def cleanup():
        """Clean up resources and free memory."""
        try:
            gc.collect()

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            print("🧹 Cleaned up resources and freed memory")
        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")
def create_intelligent_canvas(image, target_width, target_height):
    """Create canvas with intelligent positioning and background."""
    original_width, original_height = image.size
    
    # Calculate scaling to fit while maintaining aspect ratio
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    
    # Use different scaling strategy based on aspect ratios
    target_ratio = target_width / target_height
    original_ratio = original_width / original_height
    
    if abs(target_ratio - original_ratio) < 0.1:
        # Similar aspect ratios - scale to fit with small margin
        scale = min(scale_w, scale_h) * 0.85
    else:
        # Different aspect ratios - be more conservative
        scale = min(scale_w, scale_h) * 0.7
    
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Ensure minimum size
    min_dimension = min(target_width, target_height) // 3
    if new_width < min_dimension or new_height < min_dimension:
        scale = min_dimension / min(original_width, original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
    
    # High-quality resize
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Intelligent positioning based on content
    x_offset, y_offset = calculate_optimal_position(
        resized_image, target_width, target_height, new_width, new_height
    )
    
    # Create canvas with neutral background
    canvas = create_neutral_background(target_width, target_height)
    
    # Create seamless background using content-aware fill
    canvas = create_content_aware_background(
        resized_image, canvas, x_offset, y_offset, target_width, target_height
    )
    
    return canvas, (x_offset, y_offset), (new_width, new_height)

def calculate_optimal_position(image, target_width, target_height, img_width, img_height):
    """Calculate optimal position based on image content analysis."""
    # Default to center
    x_offset = (target_width - img_width) // 2
    y_offset = (target_height - img_height) // 2
    
    # Analyze image content for better positioning
    img_array = np.array(image)
    
    # Detect sky (upper blue regions)
    upper_third = img_array[:img_height//3, :, :]
    blue_channel = upper_third[:, :, 2]
    is_sky_heavy = np.mean(blue_channel) > 120 and np.mean(blue_channel) > np.mean(upper_third[:, :, 1])
    
    # Detect ground/landscape (lower regions)
    lower_third = img_array[2*img_height//3:, :, :]
    green_channel = lower_third[:, :, 1]
    brown_channel = (lower_third[:, :, 0] + lower_third[:, :, 1]) // 2
    is_ground_heavy = np.mean(green_channel) > 100 or np.mean(brown_channel) > 100
    
    # Adjust positioning based on content
    if is_sky_heavy and not is_ground_heavy:
        # Sky dominant - position higher to allow ground extension
        y_offset = min(y_offset + (target_height - img_height) // 4, target_height - img_height)
    elif is_ground_heavy and not is_sky_heavy:
        # Ground dominant - position lower to allow sky extension
        y_offset = max(y_offset - (target_height - img_height) // 4, 0)
    
    return x_offset, y_offset

def create_neutral_background(width, height):
    """Create a neutral background color based on common image statistics."""
    # Use a slightly warm neutral gray that works well with most images
    background_color = (118, 118, 112)  # Slightly warm gray
    return Image.new("RGB", (width, height), background_color)

def create_content_aware_background(image, canvas, x_offset, y_offset, target_width, target_height):
    """Create background by mirroring and blending edge content."""
    img_width, img_height = image.size
    img_array = np.array(image)
    canvas_array = np.array(canvas)
    
    # Mirror and blend edges instead of stretching
    mirror_size = min(50, img_width//8, img_height//8)
    
    # Top region
    if y_offset > mirror_size:
        top_strip = img_array[:mirror_size, :, :]
        # Flip vertically and apply fade
        top_mirrored = np.flip(top_strip, axis=0)
        for i in range(mirror_size):
            alpha = (mirror_size - i) / mirror_size * 0.3  # Gentle blend
            y_pos = y_offset - mirror_size + i
            if 0 <= y_pos < target_height:
                canvas_array[y_pos, x_offset:x_offset+img_width] = (
                    alpha * top_mirrored[i] + (1-alpha) * canvas_array[y_pos, x_offset:x_offset+img_width]
                ).astype(np.uint8)
    
    # Bottom region
    if y_offset + img_height + mirror_size < target_height:
        bottom_strip = img_array[-mirror_size:, :, :]
        bottom_mirrored = np.flip(bottom_strip, axis=0)
        for i in range(mirror_size):
            alpha = (mirror_size - i) / mirror_size * 0.3
            y_pos = y_offset + img_height + i
            if 0 <= y_pos < target_height:
                canvas_array[y_pos, x_offset:x_offset+img_width] = (
                    alpha * bottom_mirrored[i] + (1-alpha) * canvas_array[y_pos, x_offset:x_offset+img_width]
                ).astype(np.uint8)
    
    # Left region
    if x_offset > mirror_size:
        left_strip = img_array[:, :mirror_size, :]
        left_mirrored = np.flip(left_strip, axis=1)
        for i in range(mirror_size):
            alpha = (mirror_size - i) / mirror_size * 0.3
            x_pos = x_offset - mirror_size + i
            if 0 <= x_pos < target_width:
                canvas_array[y_offset:y_offset+img_height, x_pos] = (
                    alpha * left_mirrored[:, i] + (1-alpha) * canvas_array[y_offset:y_offset+img_height, x_pos]
                ).astype(np.uint8)
    
    # Right region  
    if x_offset + img_width + mirror_size < target_width:
        right_strip = img_array[:, -mirror_size:, :]
        right_mirrored = np.flip(right_strip, axis=1)
        for i in range(mirror_size):
            alpha = (mirror_size - i) / mirror_size * 0.3
            x_pos = x_offset + img_width + i
            if 0 <= x_pos < target_width:
                canvas_array[y_offset:y_offset+img_height, x_pos] = (
                    alpha * right_mirrored[:, i] + (1-alpha) * canvas_array[y_offset:y_offset+img_height, x_pos]
                ).astype(np.uint8)
    
    # Paste original image
    canvas = Image.fromarray(canvas_array)
    canvas.paste(image, (x_offset, y_offset))
    
    return canvas

def create_advanced_mask(canvas_size, content_offset, content_size, feather_radius=40):
    """Create an advanced mask that better preserves original content."""
    width, height = canvas_size
    x_offset, y_offset = content_offset
    content_width, content_height = content_size
    
    # Create base mask
    mask = np.zeros((height, width), dtype=np.float32)
    
    # Content area should be completely preserved
    content_mask = np.zeros((content_height, content_width), dtype=np.float32)
    
    # Create inner preservation area (80% of content)
    inner_margin_x = int(content_width * 0.1)
    inner_margin_y = int(content_height * 0.1)
    
    content_mask[inner_margin_y:content_height-inner_margin_y, 
                inner_margin_x:content_width-inner_margin_x] = 0  # Completely preserve inner area
    
    # Create gradient only at the very edges
    edge_size = min(feather_radius, content_width//6, content_height//6)
    
    # Smoother gradient at edges
    for i in range(edge_size):
        alpha = np.power(i / edge_size, 2) * 255  # Quadratic falloff for smoother transition
        
        # Apply gradient only to outer edges
        if i < content_height:
            content_mask[i, :] = np.maximum(content_mask[i, :], alpha)
        if i < content_height:
            content_mask[content_height - 1 - i, :] = np.maximum(content_mask[content_height - 1 - i, :], alpha)
        if i < content_width:
            content_mask[:, i] = np.maximum(content_mask[:, i], alpha)
        if i < content_width:
            content_mask[:, content_width - 1 - i] = np.maximum(content_mask[:, content_width - 1 - i], alpha)
    
    # Everything outside content area should be generated
    mask[:] = 255
    mask[y_offset:y_offset+content_height, x_offset:x_offset+content_width] = content_mask
    
    # Apply multiple blur passes for ultra-smooth transitions
    # Ensure kernel sizes are odd and positive
    kernel_size1 = ensure_odd_kernel_size(feather_radius * 2)
    kernel_size2 = ensure_odd_kernel_size(feather_radius)
    
    mask = cv2.GaussianBlur(mask, (kernel_size1, kernel_size1), max(1.0, feather_radius/4))
    mask = cv2.GaussianBlur(mask, (kernel_size2, kernel_size2), max(0.5, feather_radius/8))
    
    return Image.fromarray(mask.astype(np.uint8), mode='L')

def generate_enhanced_prompt(image, custom_prompt=""):
    """Generate more sophisticated prompts based on detailed image analysis."""
    if custom_prompt.strip():
        return custom_prompt
    
    scene_type, brightness, avg_colors = analyze_image_content(image)
    
    # Enhanced base quality terms
    base_prompt = "masterpiece, best quality, highly detailed, photorealistic, 8k uhd, professional photography"
    
    # More sophisticated scene analysis
    scene_prompts = {
        "nature": "natural landscape, lush vegetation, organic terrain, environmental harmony, seamless wilderness",
        "sky": "atmospheric perspective, natural sky gradient, cloud formations, horizon continuity",
        "indoor": "architectural consistency, interior design harmony, ambient lighting",
        "general": "coherent composition, natural extension, stylistic continuity"
    }
    
    # Advanced lighting analysis
    if brightness > 150:
        lighting = "bright natural daylight, high key lighting, vibrant colors"
    elif brightness < 80:
        lighting = "soft ambient lighting, moody atmosphere, subtle shadows"
    else:
        lighting = "balanced natural lighting, photographic exposure"
    
    # Add composition terms
    composition = "wide angle perspective, cinematic framing, natural boundaries"
    
    final_prompt = f"{base_prompt}, {scene_prompts.get(scene_type, scene_prompts['general'])}, {lighting}, {composition}, seamless blending, no artifacts"
    
    return final_prompt

def generate_enhanced_negative_prompt():
    """Generate comprehensive negative prompt to avoid common issues."""
    return (
        "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, "
        "cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, "
        "username, blurry, duplicate, artifacts, seams, visible boundaries, inconsistent lighting, "
        "unnatural colors, distorted perspective, repetitive patterns, tiling artifacts, "
        "stretched pixels, mirrored content, copy-paste artifacts, unnatural transitions, "
        "face distortion, body deformation, architectural inconsistency, "
        "oversaturated, undersaturated, noise, grain, compression artifacts"
    )

def generate_outpainting_enhanced(image, mask, prompt="", negative_prompt="", 
                                guidance_scale=7.5, steps=50, seed=None):
    """Enhanced outpainting with better quality settings."""
    try:
        # Set up generator
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # Generate enhanced prompts
        if not prompt.strip():
            prompt = generate_enhanced_prompt(image)
        
        if not negative_prompt.strip():
            negative_prompt = generate_enhanced_negative_prompt()
        
        # Optimized generation parameters
        result = sd_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            height=image.height,
            width=image.width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            strength=0.99,  # High strength for better outpainting
            num_images_per_prompt=1,
            eta=0.0,  # Deterministic sampling
        ).images[0]
        
        return result
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        return None

def advanced_post_process(result, original_canvas, content_offset, content_size):
    """Advanced post-processing with content-aware enhancements."""
    try:
        result_array = np.array(result)
        x_offset, y_offset = content_offset
        content_width, content_height = content_size
        
        # Extract original content area for reference
        original_content = np.array(original_canvas)[
            y_offset:y_offset+content_height,
            x_offset:x_offset+content_width
        ]
        
        # Blend original content back into result for perfect preservation
        blend_margin = 10
        for i in range(blend_margin):
            alpha = i / blend_margin
            
            # Top edge
            if y_offset + i < result_array.shape[0]:
                result_array[y_offset + i, x_offset:x_offset+content_width] = (
                    alpha * result_array[y_offset + i, x_offset:x_offset+content_width] +
                    (1-alpha) * original_content[i]
                ).astype(np.uint8)
            
            # Bottom edge
            bottom_idx = y_offset + content_height - 1 - i
            if bottom_idx >= 0:
                result_array[bottom_idx, x_offset:x_offset+content_width] = (
                    alpha * result_array[bottom_idx, x_offset:x_offset+content_width] +
                    (1-alpha) * original_content[content_height - 1 - i]
                ).astype(np.uint8)
            
            # Left edge
            if x_offset + i < result_array.shape[1]:
                result_array[y_offset:y_offset+content_height, x_offset + i] = (
                    alpha * result_array[y_offset:y_offset+content_height, x_offset + i] +
                    (1-alpha) * original_content[:, i]
                ).astype(np.uint8)
            
            # Right edge
            right_idx = x_offset + content_width - 1 - i
            if right_idx >= 0:
                result_array[y_offset:y_offset+content_height, right_idx] = (
                    alpha * result_array[y_offset:y_offset+content_height, right_idx] +
                    (1-alpha) * original_content[:, content_width - 1 - i]
                ).astype(np.uint8)
        
        # Restore core content completely
        inner_margin = 20
        if content_width > 2*inner_margin and content_height > 2*inner_margin:
            result_array[
                y_offset+inner_margin:y_offset+content_height-inner_margin,
                x_offset+inner_margin:x_offset+content_width-inner_margin
            ] = original_content[inner_margin:-inner_margin, inner_margin:-inner_margin]
        
        result_enhanced = Image.fromarray(result_array)
        
        # Subtle enhancements
        enhancer = ImageEnhance.Contrast(result_enhanced)
        result_enhanced = enhancer.enhance(1.05)
        
        enhancer = ImageEnhance.Color(result_enhanced)
        result_enhanced = enhancer.enhance(1.02)
        
        # Final smoothing
        result_enhanced = result_enhanced.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return result_enhanced
        
    except Exception as e:
        print(f"⚠️  Advanced post-processing failed: {e}")
        return result


@app.post("/outpaint", tags=["Image Processing"])
async def outpaint_image_api(
    image: UploadFile = File(...),
    target_width: str = Form(1920), # "l2p" for landscape to portrait, "p2l" for portrait to landscape
    target_height: str = Form(1080)
):
    global sd_pipe # Use the SD 1.5 pipeline
    steps = 50
    guidance_scale = 8.0
    seed = None
    prompt = ""
    negative_prompt = ""
    scheduler_type = "eular_a" #ddim
    feather_radius = 45
    output_dir = "outputs"
    no_post_process = False
    target_width = int(target_width)
    target_height = int(target_height)
    
    




    if not image.filename:
        return JSONResponse(content={'error': 'No image selected'}, status_code=400)

    try:
        if 'sd_pipe' not in globals() or sd_pipe is None: # Check for sd_pipe
            return JSONResponse(content={'error': 'SD models not loaded. Please check server logs.'}, status_code=500)

        # === Read the image ===
        img_bytes = await image.read()
        original_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        original_width, original_height = original_image.size
        print(f"📷 Image size: {original_width}x{original_height}")
        print(f"📷 Target size: {target_width}x{target_height}")

        _set_scheduler(scheduler_type)

        # === Generate outpainting ===

        canvas, offset, content_size = create_intelligent_canvas(
            original_image, 
            target_width, 
            target_height
        )

        # Use advanced mask creation
        mask = create_advanced_mask(
            (target_width, target_height),
            offset,
            content_size,
            feather_radius
        )

        result = generate_outpainting_enhanced(
            canvas,
            mask,
            prompt,
            negative_prompt,
            guidance_scale,
            steps,
            seed
        )
        cleanup()
        if result is not None:
            # Advanced Post-process result
            if not no_post_process:
                result = advanced_post_process(result, canvas, offset, content_size)
                print("✨ Applied advanced post-processing")

            output_path = save_result(result, output_dir)
        else:
            print("❌ Generation failed. Check server logs for details.")
            return JSONResponse(content={'error': 'Generation failed. Check server logs for details.'}, status_code=500)

        
        buffered = io.BytesIO()
        result.save(buffered, format="PNG") # Save as PNG to preserve transparency if any
        encoded_img = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return JSONResponse(content={
            'success': True,
            'image': encoded_img,
            'message': 'Image outpainted successfully with SD'
        })

    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")

    except Exception as e:
        # Log the exception for debugging
        print(f"Error during outpainting: {e}")
        return JSONResponse(content={'error': str(e)}, status_code=500)
    finally:
        cleanup()
        pass

if __name__ == "__main__":

    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)