# app.py
import io
import base64
import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
import numpy as np
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from huggingface_hub import hf_hub_download
from contextlib import asynccontextmanager
import logging

# Import the custom modules
from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
pipe = None
device = None
dtype = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown"""
    global pipe, device, dtype
    
    logger.info("Loading models...")
    
    # Initialize CUDA if available
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    
    # Load ControlNet
    config_file = hf_hub_download(
        "xinsir/controlnet-union-sdxl-1.0",
        filename="config_promax.json",
    )
    
    config = ControlNetModel_Union.load_config(config_file)
    controlnet_model = ControlNetModel_Union.from_config(config)
    
    # Load the state dictionary
    model_file = hf_hub_download(
        "xinsir/controlnet-union-sdxl-1.0",
        filename="diffusion_pytorch_model_promax.safetensors",
    )
    state_dict = load_state_dict(model_file)
    
    # Extract the keys from the state_dict
    loaded_keys = list(state_dict.keys())
    
    # Load pretrained model
    result = ControlNetModel_Union._load_pretrained_model(
        controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0", loaded_keys
    )
    
    model = result[0]
    model = model.to(device=device, dtype=dtype)
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype
    ).to(device)
    
    # Load pipeline
    pipe = StableDiffusionXLFillPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0_Lightning",
        torch_dtype=dtype,
        vae=vae,
        controlnet=model,
        variant="fp16" if dtype == torch.float16 else None,
    ).to(device)
    
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
    
    logger.info("Models loaded successfully!")
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

def prepare_image_and_mask(image, width, height, overlap_percentage=10, resize_option="Full", 
                          custom_resize_percentage=50, alignment="Middle", 
                          overlap_left=True, overlap_right=True, overlap_top=True, overlap_bottom=True):
    """Prepare image and mask for outpainting"""
    target_size = (width, height)

    # Calculate the scaling factor to fit the image within the target size
    scale_factor = min(target_size[0] / image.width, target_size[1] / image.height)
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    
    # Resize the source image to fit within target size
    source = image.resize((new_width, new_height), Image.LANCZOS)

    # Apply resize option using percentages
    if resize_option == "Full":
        resize_percentage = 100
    elif resize_option == "50%":
        resize_percentage = 50
    elif resize_option == "33%":
        resize_percentage = 33
    elif resize_option == "25%":
        resize_percentage = 25
    else:  # Custom
        resize_percentage = custom_resize_percentage

    # Calculate new dimensions based on percentage
    resize_factor = resize_percentage / 100
    new_width = int(source.width * resize_factor)
    new_height = int(source.height * resize_factor)

    # Ensure minimum size of 64 pixels
    new_width = max(new_width, 64)
    new_height = max(new_height, 64)

    # Resize the image
    source = source.resize((new_width, new_height), Image.LANCZOS)

    # Calculate the overlap in pixels based on the percentage
    overlap_x = int(new_width * (overlap_percentage / 100))
    overlap_y = int(new_height * (overlap_percentage / 100))

    # Ensure minimum overlap of 1 pixel
    overlap_x = max(overlap_x, 1)
    overlap_y = max(overlap_y, 1)

    # Calculate margins based on alignment
    if alignment == "Middle":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Left":
        margin_x = 0
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Right":
        margin_x = target_size[0] - new_width
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Top":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = 0
    elif alignment == "Bottom":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = target_size[1] - new_height

    # Adjust margins to eliminate gaps
    margin_x = max(0, min(margin_x, target_size[0] - new_width))
    margin_y = max(0, min(margin_y, target_size[1] - new_height))

    # Create a new background image and paste the resized source image
    background = Image.new('RGB', target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))

    # Create the mask
    mask = Image.new('L', target_size, 255)
    mask_draw = ImageDraw.Draw(mask)

    # Calculate overlap areas
    white_gaps_patch = 2

    left_overlap = margin_x + overlap_x if overlap_left else margin_x + white_gaps_patch
    right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width - white_gaps_patch
    top_overlap = margin_y + overlap_y if overlap_top else margin_y + white_gaps_patch
    bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height - white_gaps_patch
    
    if alignment == "Left":
        left_overlap = margin_x + overlap_x if overlap_left else margin_x
    elif alignment == "Right":
        right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width
    elif alignment == "Top":
        top_overlap = margin_y + overlap_y if overlap_top else margin_y
    elif alignment == "Bottom":
        bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height

    # Draw the mask
    mask_draw.rectangle([
        (left_overlap, top_overlap),
        (right_overlap, bottom_overlap)
    ], fill=0)

    return background, mask

def can_expand(source_width, source_height, target_width, target_height, alignment):
    """Checks if the image can be expanded based on the alignment."""
    if alignment in ("Left", "Right") and source_width >= target_width:
        return False
    if alignment in ("Top", "Bottom") and source_height >= target_height:
        return False
    return True

def process_outpaint(image, width, height, num_inference_steps=8, prompt_input=""):
    """Process the outpainting with default parameters matching the original"""
    global pipe, device, dtype

    logger.info(f"Using device: {device}")
    
    # Default parameters from the original app
    overlap_percentage = 10
    resize_option = "Full"
    custom_resize_percentage = 50
    alignment = "Middle"
    overlap_left = True
    overlap_right = True
    overlap_top = True
    overlap_bottom = True
    
    # Prepare image and mask
    background, mask = prepare_image_and_mask(
        image, width, height, overlap_percentage, resize_option, 
        custom_resize_percentage, alignment, overlap_left, overlap_right, 
        overlap_top, overlap_bottom
    )
    
    # Check if expansion is possible
    if not can_expand(background.width, background.height, width, height, alignment):
        alignment = "Middle"
    
    # Create control net image
    cnet_image = background.copy()
    cnet_image.paste(0, (0, 0), mask)
    
    # Prepare prompt
    final_prompt = f"{prompt_input} , high quality, 4k" if prompt_input else "high quality, 4k"
    
    # Generate image
    with torch.autocast(device_type=device, dtype=dtype):
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(final_prompt, device, True)
        
        # Get the last image from the generator
        result_image = None
        for image in pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            image=cnet_image,
            num_inference_steps=num_inference_steps
        ):
            result_image = image
    
    # Composite the result
    result_image = result_image.convert("RGBA")
    cnet_image.paste(result_image, (0, 0), mask)
    
    return cnet_image

@app.post("/outpaint")
async def outpaint(
    image: UploadFile = File(...),
    target_width: str = Form(...),
    target_height: str = Form(...)
):
    """Outpaint endpoint matching the frontend API expectations"""
    try:
        # Validate dimensions
        try:
            width = int(target_width)
            height = int(target_height)
            if width <= 0 or height <= 0:
                raise ValueError("Dimensions must be positive")
        except ValueError as e:
            return JSONResponse(
                content={"success": False, "error": "Invalid target dimensions"},
                status_code=400
            )
        
        # Read and validate image
        image_data = await image.read()
        try:
            pil_image = Image.open(io.BytesIO(image_data))
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
        except Exception as e:
            return JSONResponse(
                content={"success": False, "error": "Invalid image file"},
                status_code=400
            )
        
        # Check if target dimensions are smaller than original
        if width < pil_image.width or height < pil_image.height:
            return JSONResponse(
                content={
                    "success": False, 
                    "error": f"Target dimensions must be >= original ({pil_image.width}x{pil_image.height})"
                },
                status_code=400
            )
        
        # Process the image
        logger.info(f"Processing outpaint: {pil_image.width}x{pil_image.height} -> {width}x{height}")
        result_image = process_outpaint(pil_image, width, height)
        
        # Convert result to PNG bytes
        output_buffer = io.BytesIO()
        result_image.save(output_buffer, format='PNG')
        output_buffer.seek(0)

        #clear cuda cache
        torch.cuda.empty_cache()
        
        # Encode to base64
        base64_image = base64.b64encode(output_buffer.read()).decode('utf-8')
        
        return JSONResponse(
            content={
                "success": True,
                "image": base64_image,
                "message": f"Outpainted to {width}x{height}"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in outpaint: {str(e)}")
        return JSONResponse(
            content={"success": False, "error": "Internal server error"},
            status_code=500
        )

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": pipe is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)