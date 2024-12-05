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
channels = 3
timesteps = 1000
epochs = 50
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

# Enhanced Conditional UNet with Self-Attention and Spectral Norm
class EnhancedSelfAttentionUNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(EnhancedSelfAttentionUNet, self).__init__()
        self.num_classes = num_classes

        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, image_size * image_size)

        # Encoder
        self.encoder = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels + 1, 64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)),
            nn.ReLU()
        )

        # Self-Attention Layer
        self.self_attention = SelfAttention(128)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.Dropout(0.2),  # Regularization
            nn.utils.spectral_norm(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, t, labels):
        # Embed labels
        label_embed = self.label_embedding(labels).view(-1, 1, image_size, image_size)
        x = torch.cat([x, label_embed], dim=1)

        # Encoder
        x = self.encoder(x)

        # Self-Attention
        x = self.self_attention(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.decoder(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, channels, height, width = x.size()
        query = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, height * width)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        value = self.value(x).view(batch, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch, channels, height, width)
        return self.gamma * out + x


# Noise Schedule (Cosine)
def cosine_noise_schedule(timesteps):
    return 0.02 * (1 - torch.cos(torch.linspace(0, torch.pi, timesteps)))


# Forward Diffusion Process
def q_sample(x_start, t, noise):
    batch_size = x_start.size(0)
    alphas_t = alphas_cumprod[t].view(batch_size, 1, 1, 1)
    return x_start * torch.sqrt(1 - alphas_t) + noise * torch.sqrt(alphas_t)


# Reverse Diffusion Process
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
            x = (x - pred_noise * torch.sqrt(beta_t) / torch.sqrt(1 - alpha_t)) / torch.sqrt(alpha_cumprod_prev_t / alpha_cumprod_t)
            x = x + noise * torch.sqrt(beta_t)
        else:
            x = (x - pred_noise * torch.sqrt(beta[t])) / torch.sqrt(1 - beta[t])
    return x


# Training Loop
def train_diffusion_model(model, dataloader, optimizer, timesteps, device, class_to_idx):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(epochs):
        total_loss = 0
        for batch, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            batch, labels = batch.to(device), labels.to(device)
            noise = torch.randn_like(batch)
            t = torch.randint(0, timesteps, (batch.size(0),), device=device).long()
            x_noisy = q_sample(batch, t, noise)
            pred_noise = model(x_noisy, t, labels)
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")


# Prepare Dataset
root_dir = "./data/train/Akin_SAGAN_500/semantic"
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
dataset = UISemanticDataset(root_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize Model, Optimizer, and Noise Schedule
num_classes = len(os.listdir(root_dir))
model = EnhancedSelfAttentionUNet(channels, channels, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
beta = cosine_noise_schedule(timesteps).to(device)
alphas = 1.0 - beta
alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

# Train the Model
train_diffusion_model(model, dataloader, optimizer, timesteps, device, dataset.class_to_idx)

# Generate UI Wireframes
model.eval()
num_images = 2
for ui_type in dataset.class_to_idx.keys():
    generated_images = generate_images(model, timesteps, device, num_images, ui_type, dataset.class_to_idx)

    # Save and Display Images
    generated_images = (generated_images + 1) / 2
    for i, img in enumerate(generated_images):
        save_path = f"diff/generated_{ui_type}_{i + 1}.png"
        plt.figure()
        plt.axis('off')
        plt.imshow(img.permute(1, 2, 0).cpu().numpy())
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
