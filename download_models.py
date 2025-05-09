import os
import sys
import requests
from pathlib import Path

def download_file(url, destination):
    """
    Download a file from a URL to a destination path
    """
    print(f"Downloading {url} to {destination}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Download the file
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Downloaded {destination} successfully!")

def main():
    # Define the model URLs and destinations
    models = {
        "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    }
    
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    
    # Create the weights directory if it doesn't exist
    weights_dir = current_dir / "Real-ESRGAN-master" / "weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    # Download each model
    for model_name, model_url in models.items():
        destination = weights_dir / model_name
        
        # Check if the model already exists
        if os.path.exists(destination):
            print(f"{model_name} already exists at {destination}")
            continue
        
        try:
            download_file(model_url, destination)
        except Exception as e:
            print(f"Error downloading {model_name}: {e}")

if __name__ == "__main__":
    main()