import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            1,
            16,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            2,
            2
        )

        self.fc1 = nn.Linear(
            16 * 14 * 14,
            64
        )

        self.fc2 = nn.Linear(
            64,
            10
        )

    def forward(self, x):

        x = self.pool(
            torch.relu(
                self.conv(x)
            )
        )

        x = x.view(
            -1,
            16 * 14 * 14
        )

        x = torch.relu(
            self.fc1(x)
        )

        x = self.fc2(x)

        return x

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 3

losses = []

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    losses.append(running_loss)

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {running_loss:.4f}"
    )

correct = 0
total = 0

model.eval()

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs.data,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

accuracy = 100 * correct / total

print(
    f"\nTest Accuracy: "
    f"{accuracy:.2f}%"
)

plt.figure(figsize=(8, 5))
plt.plot(
    range(1, epochs + 1),
    losses,
    marker="o"
)

plt.title("Training Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
