import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Dataset Preparation
class UIDataset(Dataset):
    def __init__(self, base_dir, transform):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for label, subdir in enumerate(os.listdir(base_dir)):
            subdir_path = os.path.join(base_dir, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    if file.endswith((".png", ".jpg", ".jpeg")):
                        self.image_paths.append(os.path.join(subdir_path, file))
                        self.labels.append(subdir)  # Keep UI type as text for guidance

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define image transformations
transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize([0.5], [0.5]),
])

# Load dataset
base_dir = "./data/train/Akin_SAGAN_500/semantic"
dataset = UIDataset(base_dir, transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Step 2: Define the U-Net Architecture
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        middle = self.middle(encoded)
        decoded = self.decoder(middle)
        
        # Crop or resize to match input size
        decoded = F.interpolate(decoded, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return decoded

# Initialize model
model = UNet().to(device)

# Step 3: Diffusion Process (Noise Scheduler)
def noise_schedule(t, beta_start=0.0001, beta_end=0.02):
    """
    Linear noise schedule from `beta_start` to `beta_end`.
    """
    return beta_start + t * (beta_end - beta_start)


def add_noise(x, t, noise=None):
    """
    Adds noise to the input `x` based on the timestep `t` and noise schedule.
    """
    if noise is None:
        noise = torch.randn_like(x)
    
    # Compute beta values for each t
    beta = noise_schedule(t).view(-1, 1, 1, 1)  # Expand to match spatial dimensions
    
    return torch.sqrt(1 - beta) * x + torch.sqrt(beta) * noise


# Step 4: Loss Function and Optimization
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        images, _ = batch
        images = images.to(device)

        # Add noise to images
        t = torch.rand(size=(images.size(0),), device=device)
        noisy_images = add_noise(images, t)

        # Predict denoised image
        predicted_images = model(noisy_images)

        # Calculate loss
        loss = criterion(predicted_images, images)
        epoch_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

# Step 5: Sampling New Images
def sample(model, num_samples=2, types=["splash", "login", "account_creation", "product_description", "product_listing"], img_size=(3, 128, 128)):
    model.eval()
    with torch.no_grad():
        for ui_type in types:
            for i in range(num_samples):
                # Start with random noise
                img = torch.randn((1, *img_size), device=device)

                # Reverse diffusion process
                for t in reversed(range(100)):  # Simulated steps
                    img = model(img)  # Refine the noise
                img = (img.clamp(-1, 1) + 1) / 2  # Rescale to [0, 1]
                save_image(img, f"./stable_diffusion/{ui_type.replace(' ', '_').lower()}_{i}.png")

# Generate images
sample(model, num_samples=2)
