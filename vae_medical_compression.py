import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.utils as vutils
import numpy as np
import os
import matplotlib.pyplot as plt

# Description: PyTorch Implementation of a Convolutional Variational Autoencoder (VAE)
# tailored for an academic project on Medical Image Compression.

# Check if GPU is available (highly recommended for Deep Learning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------
# 1. Variational Autoencoder (VAE) Architecture
# ---------------------------------------------------------
class MedicalConvVAE(nn.Module):
    def __init__(self, image_channels=1, latent_dim=128):
        """
        Convolutional VAE specifically designed to handle grayscale medical images 
        (e.g., MRI, CT, X-Ray).
        """
        super(MedicalConvVAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # --- ENCODER: Image to Latent Distribution ---
        self.encoder = nn.Sequential(
            # Input: (image_channels, 64, 64) -> Assumes images are resized to 64x64
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1), # Shape: 32x32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),           # Shape: 64x16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),          # Shape: 128x8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),         # Shape: 256x4x4
            nn.ReLU()
        )
        
        # Flattening size: 256 * 4 * 4 = 4096
        self.flatten_size = 256 * 4 * 4
        
        # The encoder maps the flattened vector to Mean and Log Variance
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # --- DECODER: Latent Vector to Image ---
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # Shape: 128x8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Shape: 64x16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # Shape: 32x32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1), # Shape: (image_channels)x64x64
            nn.Sigmoid() # Outputs pixel intensities mapped between 0 and 1
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1) # Flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization Trick: Allows backpropagation by pushing randomness
        from the latent distribution sample to an external standard normal distribution.
        z = mu + sigma * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 4, 4) # Unflatten to match transpose convolution input
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar

# ---------------------------------------------------------
# 2. Loss Function (ELBO)
# ---------------------------------------------------------
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Computes the Evidence Lower Bound (ELBO) objective.
    ELBO = Reconstruction Loss - KL Divergence
    Using Beta-VAE formulation allows us to adjust compression vs quality (hyperparameter Beta).
    """
    # Reconstruction Loss (MSE is often preferred for medical images over BCE to penalize outlier pixel deviance heavily)
    # Using reduction='sum' handles batch aggregation properly
    Recon_Loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence: D_KL(q(z|x) || p(z))
    # Forces the latent distribution towards a standard Unit Gaussian N(0, 1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    return Recon_Loss + (beta * KLD), Recon_Loss, KLD

# ---------------------------------------------------------
# 3. Utility: Evaluation Metrics
# ---------------------------------------------------------
def calculate_psnr(img1, img2):
    """Calculates Peak Signal to Noise Ratio (PSNR). Higher is better."""
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return 100 # Perfect reconstruction
    max_pixel = 1.0 # Assuming images are normalized [0, 1]
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# ---------------------------------------------------------
# 4. Main Script
# ---------------------------------------------------------
def main():
    # --- Hyperparameters ---
    batch_size = 64
    epochs = 2          # Increase to 50/100 for actual projects
    learning_rate = 1e-3
    latent_dim = 64      # Compression variable: Lower = more compression, but worse quality
    image_size = 64
    
    # --- Dataset Loading ---
    print("Setting up Medical Image Dataset...")

    transform = transforms.Compose([
        transforms.Resize(image_size), # Resize constraint for the standard CVAE
        transforms.ToTensor()          # Converts to [0,1] normalization
    ])
    
    # Mocking a dataset with MNIST
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # --- Initialize Model and Optimizer ---
    model = MedicalConvVAE(image_channels=1, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # --- Training Loop ---
    print("\nStarting Training Phase...")
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        recon_loss_total = 0
        kld_loss_total = 0
        
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward Pass
            recon_batch, mu, logvar = model(data)
            
            # Compute Loss
            loss, recon_loss, kld_loss = loss_function(recon_batch, data, mu, logvar)
            
            # Backpropagation
            loss.backward()
            train_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kld_loss_total += kld_loss.item()
            optimizer.step()
            
            # --- Fast demonstration (Process 1 batch only) ---
            break
            
        # Logging Average Metrics per epoch
        n_samples = len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] \t "
              f"Loss: {train_loss / n_samples:.4f} \t "
              f"Recon: {recon_loss_total / n_samples:.4f} \t "
              f"KLD: {kld_loss_total / n_samples:.4f}")
        
    # --- Compression and Evaluation Phase ---
    print("\n--- Model Evaluation & Compression Simulation ---")
    model.eval()
    
    # Select a sample image from the dataset
    sample_data, _ = dataset[0]
    sample_data = sample_data.unsqueeze(0).to(device) # Shape: [1, 1, 64, 64]
    
    with torch.no_grad():
        # 1. ENCODING (Compression Stage)
        mu, logvar = model.encode(sample_data)
        
        # In deployment, 'mu' is saved as the compressed latent representation of the scan.
        # It is usually quantized (float32 -> int8) and subjected to entropy encoding (Huffman) 
        # to achieve the final .vae compressed file.
        compressed_latent = mu.cpu() 
        
        # 2. DECODING (Decompression Stage)
        # Assuming the receiver gets 'compressed_latent', they decompress it using the decoder
        reconstructed_image = model.decode(compressed_latent.to(device))
        
    # --- Metric Output ---
    # Theoretical size in memory based on float32 elements
    uncompressed_size = sample_data.numel() * 4 # 64x64x1 x 4 bytes = 16,384 Bytes
    compressed_size = compressed_latent.numel() * 4 # 64 x 4 bytes = 256 Bytes
    
    print("\n[Compression Results]")
    print(f"Original Scan Shape:  {sample_data.shape}")
    print(f"Compressed Representation Shape: {compressed_latent.shape}")
    print(f"Theoretical Compression Ratio:   {uncompressed_size / compressed_size :.2f}:1")
    
    psnr_score = calculate_psnr(reconstructed_image, sample_data)
    print(f"\n[Quality Metrics]")
    print(f"Reconstruction PSNR:  {psnr_score:.2f} dB")
    
    # --- Visualization ---
    os.makedirs("results", exist_ok=True)
    comparison = torch.cat([sample_data, reconstructed_image])
    vutils.save_image(comparison.cpu(), 'results/compression_demo.png', nrow=1, normalize=False)
    print("\nDemo Complete! Visualization saved to -> results/compression_demo.png")

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading
    import time
    
    print("--- Running Background Inference Check (CLI) ---")
    main()
    
    print("\n===========================================================")
    print("        STARTING WEB DASHBOARD ON http://127.0.0.1:8000 ")
    print("===========================================================")
    
    def open_browser():
        time.sleep(2)
        webbrowser.open("http://127.0.0.1:8000")
        
    # Auto-open browser
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Launch Web Server API
    uvicorn.run("app:app", host="127.0.0.1", port=8000)
