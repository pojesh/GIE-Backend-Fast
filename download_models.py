import os
import torch
from huggingface_hub import hf_hub_download
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict

from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

def download_sdxl_models():
    print("Downloading SDXL models...")
    
    # Create weights directory if it doesn't exist
    os.makedirs("weights", exist_ok=True)
    
    # Download ControlNet Union model
    try:
        print("Downloading ControlNet Union model...")
        config_file = hf_hub_download(
            "xinsir/controlnet-union-sdxl-1.0",
            filename="config_promax.json",
        )
        
        model_file = hf_hub_download(
            "xinsir/controlnet-union-sdxl-1.0",
            filename="diffusion_pytorch_model_promax.safetensors",
        )
        
        print(f"Downloaded ControlNet Union model to {model_file}")
        
        # Test loading the model
        config = ControlNetModel_Union.load_config(config_file)
        controlnet_model = ControlNetModel_Union.from_config(config)
        state_dict = load_state_dict(model_file)
        loaded_keys = list(state_dict.keys())
        
        result = ControlNetModel_Union._load_pretrained_model(
            controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0", loaded_keys
        )
        
        print("ControlNet Union model loaded successfully!")
    except Exception as e:
        print(f"Error downloading ControlNet Union model: {e}")
    
    # Download VAE
    try:
        print("Downloading VAE model...")
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        print("VAE model loaded successfully!")
    except Exception as e:
        print(f"Error downloading VAE model: {e}")
    
    # Download SDXL pipeline
    try:
        print("Downloading SDXL pipeline...")
        pipeline = StableDiffusionXLFillPipeline.from_pretrained(
            "SG161222/RealVisXL_V5.0_Lightning",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            variant="fp16" if torch.cuda.is_available() else None,
        )
        print("SDXL pipeline loaded successfully!")
    except Exception as e:
        print(f"Error downloading SDXL pipeline: {e}")
    
    print("\nDownload process completed!")
    print("\nIf there were any errors, please check your internet connection and try again.")
    print("You may also need to install additional dependencies. See requirements.txt for details.")

def download_realesrgan_models():
    print("Downloading Real-ESRGAN models...")
    
    # Create weights directory if it doesn't exist
    os.makedirs(os.path.join("Real-ESRGAN-master", "weights"), exist_ok=True)
    
    # URLs for Real-ESRGAN models
    models = {
        "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    }
    
    import requests
    
    for model_name, url in models.items():
        output_path = os.path.join("Real-ESRGAN-master", "weights", model_name)
        
        if os.path.exists(output_path):
            print(f"{model_name} already exists, skipping download.")
            continue
        
        print(f"Downloading {model_name}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Downloaded {model_name} to {output_path}")
        except Exception as e:
            print(f"Error downloading {model_name}: {e}")
    
    print("\nReal-ESRGAN download process completed!")

if __name__ == "__main__":
    print("Galaxy Image Enhancer - Model Downloader")
    print("=======================================")
    print("This script will download the required models for the Galaxy Image Enhancer.")
    print("\nOptions:")
    print("1. Download SDXL models (for outpainting)")
    print("2. Download Real-ESRGAN models (for upscaling)")
    print("3. Download all models")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ")
    
    if choice == "1":
        download_sdxl_models()
    elif choice == "2":
        download_realesrgan_models()
    elif choice == "3":
        download_sdxl_models()
        print("\n" + "-"*50 + "\n")
        download_realesrgan_models()
    elif choice == "4":
        print("Exiting...")
    else:
        print("Invalid choice. Exiting...")