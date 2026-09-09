import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from dataset import HAM10000Dataset
from model_mobilenet import get_mobilenet
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_dirs = ["../data/raw/HAM10000_images_part_1", "../data/raw/HAM10000_images_part_2"]
test_ds = HAM10000Dataset("../data/processed/test.csv", img_dirs, transform=test_transform)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

model = get_mobilenet(freeze_base=True).to(device)
model.load_state_dict(torch.load("../models/mobilenet_binary_final.pth"))
model.eval()

all_labels = []
all_preds = []
all_probs = []

# measure inference latency too
start = time.time()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

elapsed = time.time() - start
per_image_ms = (elapsed / len(test_ds)) * 1000

acc = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)
cm = confusion_matrix(all_labels, all_preds)

model_size_mb = os.path.getsize("../models/mobilenet_binary_final.pth") / (1024 * 1024)

print(f"Test Accuracy:  {acc:.4f}")
print(f"Precision:      {precision:.4f}")
print(f"Recall:         {recall:.4f}  <- important, false negative = missed cancer")
print(f"F1 Score:       {f1:.4f}")
print(f"AUC-ROC:        {auc:.4f}")
print(f"\nConfusion Matrix:\n{cm}")
print(f"  (rows=actual, cols=predicted | [0,0]=TN [0,1]=FP / [1,0]=FN [1,1]=TP)")
print(f"\nInference time: {per_image_ms:.2f} ms/image")
print(f"Model size: {model_size_mb:.2f} MB")