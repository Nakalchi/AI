#pip install torch torchvision matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
class Generator(nn.Module):
    def __init__(self, noise_dim, img_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )
    def forward(self, x):
        return self.gen(x)
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.disc(x)
noise_dim = 100
img_dim = 28 * 28
batch_size = 64
lr = 0.0002
epochs = 50
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = datasets.MNIST(
    root="data",
    train=True,
    transform=transform,
    download=True
)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)
gen = Generator(noise_dim, img_dim).to(device)
disc = Discriminator(img_dim).to(device)
criterion = nn.BCELoss()
opt_gen = optim.Adam(gen.parameters(), lr=lr)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
print("Starting Training...")
for epoch in range(epochs):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.view(-1, img_dim).to(device)
        current_batch_size = real.size(0)
        noise = torch.randn(current_batch_size, noise_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        disc_fake = disc(fake.detach()).view(-1)
        loss_disc = (
            criterion(disc_real, torch.ones_like(disc_real))
            + criterion(disc_fake, torch.zeros_like(disc_fake))
        )
        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()
        output = disc(fake).view(-1)
        loss_gen = criterion(
            output,
            torch.ones_like(output)
        )
        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    print(
        f"Epoch [{epoch + 1}/{epochs}] | "
        f"Loss D: {loss_disc:.4f}, "
        f"Loss G: {loss_gen:.4f}"
    )
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            fake_images = gen(
                torch.randn(16, noise_dim).to(device)
            ).view(-1, 1, 28, 28)
            plt.figure(figsize=(4, 4))
            for i in range(16):
                plt.subplot(4, 4, i + 1)
                plt.imshow(
                    fake_images[i][0].cpu(),
                    cmap="gray"

