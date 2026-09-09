from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import HAM10000Dataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

img_dirs = ["data/raw/HAM10000_images_part_1", "data/raw/HAM10000_images_part_2"]

train_ds = HAM10000Dataset("data/processed/train.csv", img_dirs, transform=transform)
print("Dataset size:", len(train_ds))

loader = DataLoader(train_ds, batch_size=8, shuffle=True)
images, labels = next(iter(loader))
print("Batch image shape:", images.shape)
print("Batch labels:", labels)