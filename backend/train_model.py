# train_model.py → FINAL CLEAN & WORKING VERSION (Windows + CUDA)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm   # pip install tqdm

# -------------------------------
# CONFIG
# -------------------------------
DATASET_DIR     = "datasets/sleep_dataset"
IMG_SIZE        = (128, 128)
BATCH_SIZE      = 64
EPOCHS          = 50
MODEL_SAVE_PATH = "models/sleep_model.pth"

# -------------------------------
# Only main process prints (no spam!)
# -------------------------------
def print_once(*args, **kwargs):
    if torch.utils.data.get_worker_info() is None:
        print(*args, **kwargs)

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_once(f"Using device: {device}")

# -------------------------------
# Transforms
# -------------------------------
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=8, translate=(0.1,0.1), scale=(0.9,1.15)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# -------------------------------
# Dataset
# -------------------------------
def is_valid_file(f):
    return f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff','.webp'))

full_dataset = ImageFolder(root=DATASET_DIR, transform=train_transform, is_valid_file=is_valid_file)

print_once("Detected classes:")
for i, c in enumerate(full_dataset.classes):
    print_once(f"  {i}: {c}")

train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

print_once(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# -------------------------------
# Model
# -------------------------------
class SleepCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),
            nn.Linear(128*4*4, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

model = SleepCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

# -------------------------------
# TRAINING FUNCTION
# -------------------------------
def train():
    best_acc = 0.0
    os.makedirs("models", exist_ok=True)

    print_once("\nTraining Started...\n")

    for epoch in range(EPOCHS):
        # ====== TRAIN ======
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d} [Train]", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

            pbar.set_postfix(loss=f"{train_loss/(pbar.n+1):.4f}", acc=f"{100.*correct/total:.2f}%")

        # ====== VALIDATION ======
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1:02d} [Val  ]", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, pred = outputs.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()

        val_acc = 100. * correct / total
        val_loss /= len(val_loader)

        print(f"\nEpoch {epoch+1:02d}/{EPOCHS} → "
              f"Train Acc: {100.*correct/total:.2f}% | "
              f"Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Best model saved! → {val_acc:.2f}%\n")

    print(f"\nTraining finished! Best validation accuracy: {best_acc:.2f}%")
    print(f"Model → {MODEL_SAVE_PATH}")

# ===============================
if __name__ == '__main__':
    train()