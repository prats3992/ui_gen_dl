import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.io import read_image
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Parameters
image_size = 64
channels = 3  # Assuming RGB semantic images
timesteps = 1000
epochs = 500
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Loader
class UISemanticDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.file_paths = []
        self.labels = []

        # Collect all file paths and their labels
        for label, idx in self.class_to_idx.items():
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for file in os.listdir(label_dir):
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        self.file_paths.append(os.path.join(label_dir, file))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        image = read_image(img_path).float() / 255.0  # Normalize to [0, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

# Conditional UNet Model
class ConditionalUNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(ConditionalUNet, self).__init__()
        self.num_classes = num_classes

        # Embedding for class labels
        self.label_embedding = nn.Embedding(num_classes, image_size * image_size)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels + 1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.middle = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, t, labels):
        # Add label embedding as a channel
        label_embed = self.label_embedding(labels).view(-1, 1, image_size, image_size)
        x = torch.cat([x, label_embed], dim=1)
        
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

# Noise Schedule
def linear_noise_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

# Forward Diffusion Process
def q_sample(x_start, t, noise):
    batch_size = x_start.size(0)
    alphas_t = alphas_cumprod[t].view(batch_size, 1, 1, 1)
    return x_start * torch.sqrt(1 - alphas_t) + noise * torch.sqrt(alphas_t)

# Reverse Diffusion (Denoising)
@torch.no_grad()
def generate_images(model, timesteps, device, num_images, ui_type, class_to_idx):
    x = torch.randn((num_images, channels, image_size, image_size), device=device)
    labels = torch.tensor([class_to_idx[ui_type]] * num_images, device=device)
    for t in reversed(range(timesteps)):
        t_tensor = torch.tensor([t] * num_images, device=device).long()
        pred_noise = model(x, t_tensor, labels)
        
        if t > 0:
            noise = torch.randn_like(x)
            beta_t = beta[t]
            alpha_t = alphas[t]
            alpha_cumprod_t = alphas_cumprod[t]
            alpha_cumprod_prev_t = alphas_cumprod[t - 1]
            
            x = (
                x - pred_noise * torch.sqrt(beta_t) / torch.sqrt(1 - alpha_t)
            ) / torch.sqrt(alpha_cumprod_prev_t / alpha_cumprod_t)
            x = x + noise * torch.sqrt(beta_t)
        else:
            x = (x - pred_noise * torch.sqrt(beta[t])) / torch.sqrt(1 - beta[t])
    return x

# Training Loop
def train_diffusion_model(model, dataloader, optimizer, timesteps, device, class_to_idx):
    for epoch in range(epochs):
        total_loss = 0
        for batch, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            batch, labels = batch.to(device), labels.to(device)
            noise = torch.randn_like(batch)  # Random noise
            t = torch.randint(0, timesteps, (batch.size(0),), device=device).long()
            x_noisy = q_sample(batch, t, noise)
            pred_noise = model(x_noisy, t, labels)
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

# Prepare Dataset
root_dir = "./data/train/Akin_SAGAN_500/semantic"  # Replace with your dataset path
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])
dataset = UISemanticDataset(root_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize Model, Optimizer, and Noise Schedule
num_classes = len(os.listdir(root_dir))
model = ConditionalUNet(channels, channels, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
beta = linear_noise_schedule(timesteps).to(device)
alphas = 1.0 - beta
alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

# Train the Model
train_diffusion_model(model, dataloader, optimizer, timesteps, device, dataset.class_to_idx)

# Generate UI Wireframes
model.eval()
# ui_type = input(f"Enter the UI type to generate ({', '.join(dataset.class_to_idx.keys())}): ")
num_images = 5
for ui_type in dataset.class_to_idx.keys():
    generated_images = generate_images(model, timesteps, device, num_images, ui_type, dataset.class_to_idx)

    # Save and Display Images
    generated_images = (generated_images + 1) / 2
    for i, img in enumerate(generated_images):
        save_path = f"generated_{ui_type}_{i + 1}.png"
        plt.figure()
        plt.axis('off')
        plt.imshow(img.permute(1, 2, 0).cpu().numpy())
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
