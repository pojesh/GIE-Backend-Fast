import os
import sys
import warnings
import cv2
import numpy as np
import torch
import base64
import uuid
from pathlib import Path
from PIL import Image, ImageDraw
import io
import torch
from diffusers import AutoencoderKL, TCDScheduler
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

#sdxl_pipe = None
#sdxl_vae = None
#sdxl_controlnet = None
sd15_pipe = None
sd15_vae = None
sd15_controlnet = None

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

        # Load SDXL models
        '''config_file = hf_hub_download(
            "xinsir/controlnet-union-sdxl-1.0",
            filename="config_promax.json",
        )
        config = ControlNetModel_Union.load_config(config_file)
        controlnet_model = ControlNetModel_Union.from_config(config)

        model_file = hf_hub_download(
            "xinsir/controlnet-union-sdxl-1.0",
            filename="diffusion_pytorch_model_promax.safetensors",
        )
        state_dict = load_state_dict(model_file)
        loaded_keys = list(state_dict.keys())

        result = ControlNetModel_Union._load_pretrained_model(
            controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0", loaded_keys
        )
        sdxl_controlnet = result[0].to(device="cuda", dtype=torch.float16)

        sdxl_vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        ).to("cuda")

        global sdxl_pipe
        sdxl_pipe = StableDiffusionXLFillPipeline.from_pretrained(
            "SG161222/RealVisXL_V5.0_Lightning",
            torch_dtype=torch.float16,
            vae=sdxl_vae,
            controlnet=sdxl_controlnet,
            variant="fp16",
        ).to("cuda")

        sdxl_pipe.scheduler = TCDScheduler.from_config(sdxl_pipe.scheduler.config)'''

        global sd15_pipe
        controlnet_model_id = "comfyanonymous/ControlNet-v1-1_fp16_safetensors"
        controlnet_filename = "control_v11p_sd15_inpaint_fp16.safetensors"

        if controlnet_filename:
            controlnet_path = hf_hub_download(controlnet_model_id, filename=controlnet_filename)
            sd15_controlnet = ControlNetModel.from_single_file(
                controlnet_path,
                torch_dtype=torch.float16
            )
        else: # If the model_id points directly to a loadable ControlNet model directory
            sd15_controlnet = ControlNetModel.from_pretrained(
                controlnet_model_id,
                torch_dtype=torch.float16
            )    

        sd15_controlnet.to(device="cuda")
        print("SD 1.5 ControlNet loaded.")

        vae_model_id = "stabilityai/sd-vae-ft-mse"
        sd15_vae = AutoencoderKL.from_pretrained(
            vae_model_id,
            torch_dtype=torch.float16
        ).to("cuda")
        print("SD 1.5 VAE loaded.")

        pipeline_model_id = "runwayml/stable-diffusion-v1-5"
        sd15_pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            pipeline_model_id,
            vae=sd15_vae,
            controlnet=sd15_controlnet,
            torch_dtype=torch.float16,
        # variant="fp16", # Often used if the model repo has specific fp16 weights, otherwise torch_dtype handles it
            safety_checker=None, # Optional: disable safety checker if you handle safety elsewhere
            requires_safety_checker=False, # Optional
        ).to("cuda")

        print(f"Stable Diffusion 1.5 Inpaint Pipeline with ControlNet loaded using {pipeline_model_id}.")

        sd15_pipe.scheduler = TCDScheduler.from_config(sd15_pipe.scheduler.config)
        print("TCD Scheduler configured for SD 1.5 pipeline.")

        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please make sure the model weights are downloaded and placed in the correct directory.")
        print("You can download them from:")
        print("- RealESRGAN_x4plus.pth: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth")
        print("- RealESRGAN_x2plus.pth: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth")
        upsampler_x4 = None
        upsampler_x2 = None
        #sdxl_pipe = None
        #sdxl_vae = None
        #sdxl_controlnet = None
        sd15_pipe = None
        sd15_vae = None
        sd15_controlnet = None


@app.get("/", tags=["General"])
async def read_root():
    return {"message": "Welcome to the Galaxy Image Enhancer API"}

@app.get("/health", tags=["General"])
async def health_check():
    global upsampler_x4, upsampler_x2, sd15_pipe
    return {
        'status': 'ok',
        'models_loaded': {
            'x4plus': upsampler_x4 is not None,
            'x2plus': upsampler_x2 is not None,
            'sdxl_outpaint': sd15_pipe is not None
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

@app.post("/outpaint", tags=["Image Processing"])
async def outpaint_image_api(
    image: UploadFile = File(...),
    format: str = Form("l2p") # "l2p" for landscape to portrait, "p2l" for portrait to landscape
):
    global sd15_pipe # Use the SD 1.5 pipeline

    if not image.filename:
        return JSONResponse(content={'error': 'No image selected'}, status_code=400)

    try:
        if 'sd15_pipe' not in globals() or sd15_pipe is None: # Check for sd15_pipe
            return JSONResponse(content={'error': 'SD 1.5 models not loaded. Please check server logs.'}, status_code=500)

        # Read the image
        '''img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        original_width, original_height = img.size

        # Determine target dimensions based on format
        if format == "l2p":  # Landscape to Portrait
            target_width = original_width
            # Ensure new aspect ratio (e.g., 2:3 or 3:4 for portrait from landscape)
            # Example: if original is 16:9, new portrait could be 9:16 (approx original_width * (16/9) for height)
            # This example uses a fixed 1.5 multiplier for height from width
            target_height = int(original_width * 1.5)
            if target_height < original_height: # Should not happen if outpainting vertically
                target_height = original_height
        elif format == "p2l":  # Portrait to Landscape
            target_height = original_height
            # Example: if original is 9:16, new landscape could be 16:9 (approx original_height * (16/9) for width)
            # This example uses a fixed 1.5 multiplier for width from height
            target_width = int(original_height * 1.5)
            if target_width < original_width: # Should not happen if outpainting horizontally
                target_width = original_width
        else:
            return JSONResponse(content={'error': 'Invalid format specified. Use "l2p" or "p2l".'}, status_code=400)

        # Ensure target dimensions are multiples of 64 (good for SD 1.5, common for ControlNet)
        # SD 1.5 typically works well with base 512, so ensure dimensions are reasonable.
        target_width = (target_width // 64) * 64
        target_height = (target_height // 64) * 64
        if target_width == 0: target_width = 64 # Avoid zero dimensions
        if target_height == 0: target_height = 64

        # Create padded image (where original image is placed) and mask (defines outpainting area)
        # Using a neutral gray for padding, can be any color.
        padded_img = Image.new('RGB', (target_width, target_height), (127, 127, 127))
        
        
        # Calculate paste position to center the original image
        paste_x = (target_width - original_width) // 2
        paste_y = (target_height - original_height) // 2

        padded_img.paste(img, (paste_x, paste_y))

        # Create mask: white (255) where we want to outpaint, black (0) where original image is
        mask = Image.new('L', (target_width, target_height), 255) # Start with all white (outpaint everything)
        mask_draw = ImageDraw.Draw(mask)
        # Make the original image area black (don't outpaint this area)
        mask_draw.rectangle(
            [paste_x, paste_y, paste_x + original_width, paste_y + original_height],
            fill=0
        )

        # The control_image for ControlNetInpaint is typically the padded_img itself.
        control_image = padded_img.copy()

        # Prepare prompt
        # Using the same detailed prompt as before.
        prompt_input = "Expand the existing scene realistically and seamlessly, matching the original image’s style, lighting, perspective, and subject matter. Continue the environment naturally with cohesive elements, adding subtle background details and maintaining visual harmony across the entire composition."
        final_prompt = f"{prompt_input} , high quality, detailed" # Removed 4k as SD1.5 native is lower
        negative_prompt = "low quality, blurry, noisy, text, watermark, signature, deformed, disfigured, extra limbs, bad anatomy" # Example negative prompt

        # Perform inference
        # Note: SD 1.5 pipeline does not use pooled_prompt_embeds
        # It typically takes prompt and negative_prompt as strings.
        # The TCDScheduler might allow for very few steps.
        # Ensure `sd15_pipe` is on the correct device (e.g., "cuda")
        
        output_image = sd15_pipe(
            prompt=final_prompt,
            negative_prompt=negative_prompt, # Added negative prompt
            image=padded_img, # The image with the original content and areas to fill
            mask_image=mask,  # Mask indicating where to inpaint (white areas)
            control_image=control_image, # Image for ControlNet conditioning
            num_inference_steps=20, # Increased steps for potentially better SD1.5 quality, adjust as needed with TCD
            guidance_scale=7.5, # Typical guidance scale for SD1.5, adjust as needed
            # strength=0.99, # For inpainting, strength is important. Default is 1.0.
        ).images[0]

        # Composite the original image area from padded_img (which is the original)
        # onto the outpainted result to ensure the original part is pristine.
        # The outpainted image 'output_image' should ideally respect the black areas of the mask,
        # but explicit compositing can ensure the original part is perfectly preserved.
        output_image_rgba = output_image.convert("RGBA")
        original_content_rgba = img.convert("RGBA")

        # Create a new transparent image with target dimensions
        final_composited_image = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
        # Paste the outpainted result
        final_composited_image.paste(output_image_rgba, (0,0))
        # Paste the original image on top of the outpainted area corresponding to original image
        final_composited_image.paste(original_content_rgba, (paste_x, paste_y))'''

        # === Read the image ===
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        original_width, original_height = img.size

        # === Determine target dimensions ===
        if format == "l2p":
            target_width = original_width
            target_height = int(original_width * 1.5)
            if target_height < original_height:
                target_height = original_height
        elif format == "p2l":
            target_height = original_height
            target_width = int(original_height * 1.5)
            if target_width < original_width:
                target_width = original_width
        else:
            return JSONResponse(content={'error': 'Invalid format specified. Use "l2p" or "p2l".'}, status_code=400)

        # Align to nearest multiple of 64
        target_width = max(64, (target_width // 64) * 64)
        target_height = max(64, (target_height // 64) * 64)

        # === Create padded canvas with mirrored edges ===
        paste_x = (target_width - original_width) // 2
        paste_y = (target_height - original_height) // 2

        # === Create padded canvas ===
        # Option 1: Gray padding (fallback)
        padded_img = Image.new('RGB', (target_width, target_height), (127, 127, 127))
        # Option 2: Mirror padding for more realism
        '''img_np = np.array(img)
        padded_img_np = cv2.copyMakeBorder(img_np, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT)
        padded_img = Image.fromarray(padded_img_np)

        paste_x = (target_width - original_width) // 2
        paste_y = (target_height - original_height) // 2
        padded_img.paste(img, (paste_x,paste_y))'''

        # === Create mask ===
        mask = Image.new('L', (target_width, target_height), 255)
        ImageDraw.Draw(mask).rectangle(
            [paste_x, paste_y, paste_x + original_width, paste_y + original_height],
            fill=0
        )

        # === Create edge map for ControlNet conditioning ===
        def create_edge_map(pil_img: Image.Image) -> Image.Image:
            img_np = np.array(pil_img)
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(img_gray, 100, 200)
            return Image.fromarray(edges).convert("RGB")

        control_image = create_edge_map(padded_img)

        # === Prompt ===
        final_prompt = (
            "Expand the scene realistically based on the original image. Match the lighting, perspective, and visual tone. "
            "Continue the environment naturally, ensuring consistency in depth, composition, and background elements. "
            "Use detailed, photorealistic textures and cinematic quality."
        )
        negative_prompt = (
            "low resolution, bad anatomy, blurry, distorted, extra limbs, text, watermark, unrealistic, broken perspective, deformed"
        )

        # === Generate with SD1.5 + ControlNet ===
        generator = torch.manual_seed(42)  # Use random.randint(0, 100000) for different each time

        output_image = sd15_pipe(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            image=padded_img,
            mask_image=mask,
            control_image=control_image,
            num_inference_steps=50,
            guidance_scale=8.0,
            generator=generator
        ).images[0]

        # === Composite original area ===
        output_image_rgba = output_image.convert("RGBA")
        original_content_rgba = img.convert("RGBA")

        final_composited_image = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
        final_composited_image.paste(output_image_rgba, (0, 0))
        final_composited_image.paste(original_content_rgba, (paste_x, paste_y))


        # Convert the final image to base64
        buffered = io.BytesIO()
        final_composited_image.save(buffered, format="PNG") # Save as PNG to preserve transparency if any
        encoded_img = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return JSONResponse(content={
            'success': True,
            'image': encoded_img,
            'message': 'Image outpainted successfully with SD 1.5'
        })

    except Exception as e:
        # Log the exception for debugging
        print(f"Error during outpainting: {e}")
        return JSONResponse(content={'error': str(e)}, status_code=500)
    finally:
        # Clean up temporary files if any were created (though none are explicitly saved here)
        pass

if __name__ == "__main__":

    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)