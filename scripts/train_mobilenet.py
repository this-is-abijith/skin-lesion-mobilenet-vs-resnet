import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import HAM10000Dataset
from model_mobilenet import get_mobilenet
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# transforms — train has augmentation, val/test don't
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_dirs = ["../data/raw/HAM10000_images_part_1", "../data/raw/HAM10000_images_part_2"]

train_ds = HAM10000Dataset("../data/processed/train.csv", img_dirs, transform=train_transform)
val_ds   = HAM10000Dataset("../data/processed/val.csv", img_dirs, transform=val_transform)

# batch size 16 - safe for 4GB VRAM
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

model = get_mobilenet(freeze_base=True).to(device)

# handle class imbalance: weight malignant class higher (1566 malignant vs 6458 benign in train)
pos_weight = torch.tensor([6458 / 1566]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    start = time.time()

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        elapsed = time.time() - start
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {correct/total:.4f} | Time: {elapsed:.1f}s")

    # save checkpoint every epoch - survive crash
    torch.save(model.state_dict(), f"../models/mobilenet_epoch{epoch+1}.pth")

# final save after all epochs done (outside loop, no indent)
torch.save(model.state_dict(), "../models/mobilenet_binary_final.pth")
print("Final model saved to models/mobilenet_binary_final.pth")