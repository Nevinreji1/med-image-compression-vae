import os
import io
import time
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
from torchvision import transforms
import base64
import numpy as np

# Import the model architecture
from vae_medical_compression import MedicalConvVAE, calculate_psnr, device

app = FastAPI(title="VAE Medical Image Compression API")

# Ensure static directory exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global model
latent_dim = 64
print("Loading model for web inference...")
model = MedicalConvVAE(image_channels=1, latent_dim=latent_dim).to(device)
model.eval() # Set to evaluation mode immediately

# Preprocessing transforms (To match the model's expected 64x64 format)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def tensor_to_base64_image(tensor):
    """Converts a PyTorch tensor (1, 64, 64) back to a base64 string for HTML display."""
    img_array = tensor.squeeze().cpu().numpy()
    img_array = (img_array * 255).astype('uint8')
    img = Image.fromarray(img_array, mode='L')
    
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.post("/api/compress")
async def compress_image(file: UploadFile = File(...)):
    try:
        # 1. Read and preprocess the uploaded image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        input_tensor = transform(image).unsqueeze(0).to(device) # Shape: [1, 1, 64, 64]
        
        # Original Image Base64 (Resize it so it matches dimensions exactly for visual parity)
        original_b64 = tensor_to_base64_image(input_tensor)
        
        # 2. Add an artificial delay so the UI micro-animations are visible and satisfying
        time.sleep(1.2)
        
        # 3. Compress (Encode)
        with torch.no_grad():
            mu, _ = model.encode(input_tensor)
            
            # Theoretical storage calculations
            uncompressed_size = input_tensor.numel() * 4 # float32 bytes
            compressed_size = mu.numel() * 4
            compression_ratio = uncompressed_size / compressed_size
            
            # 4. Decompress (Decode)
            reconstructed_tensor = model.decode(mu)
            
            # 5. Generate visualization of the latent bottleneck (compressed representation)
            mu_normalized = (mu - mu.min()) / (mu.max() - mu.min() + 1e-5) # Normalize to 0-1
            mu_8x8 = mu_normalized.view(1, 8, 8) 
            latent_b64 = tensor_to_base64_image(mu_8x8)
            
        # 6. Calculate Fast Metrics
        psnr = calculate_psnr(reconstructed_tensor, input_tensor)
        
        # Convert reconstructed tensor to Base64 image
        reconstructed_b64 = tensor_to_base64_image(reconstructed_tensor)
        
        return JSONResponse(content={
            "success": True,
            "original_image": original_b64,
            "latent_image": latent_b64,
            "reconstructed_image": reconstructed_b64,
            "compression_ratio": round(compression_ratio, 2),
            "original_size": f"{uncompressed_size / 1024:.2f} KB",
            "compressed_size": f"{compressed_size / 1024:.2f} KB",
            "psnr": round(psnr, 2)
        })

    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.get("/")
def read_root():
    # Redirect root to our HTML interface
    return RedirectResponse(url="/static/index.html")

