# Mobile-Deployable Skin Lesion Classification

## Problem Statement

Skin cancer, particularly melanoma, is highly treatable when detected early but often 
diagnosed late due to reliance on specialist dermatologists — a major barrier in rural 
and low-resource healthcare settings. Deep CNNs (ResNet, DenseNet) achieve high accuracy 
in automated lesion classification, but their size and compute cost make them impractical 
for mobile/edge deployment — exactly where early screening is most needed.

This project evaluates whether a lightweight architecture (MobileNetV2) can match a 
heavier benchmark (ResNet50) on diagnostic performance, while being small and fast enough 
for real-world mobile deployment.

## Dataset

HAM10000 — 10,015 dermoscopic images across 7 lesion types, binarized into 
benign vs malignant for this experiment.

- Malignant: melanoma (mel), basal cell carcinoma (bcc), actinic keratoses (akiec)
- Benign: nevus (nv), benign keratosis (bkl), dermatofibroma (df), vascular (vasc)

Split done **at patient (lesion_id) level**, not image level, to prevent data leakage — 
HAM10000 contains multiple images per lesion.

| Split | Images | Unique Lesions |
|---|---|---|
| Train | 8024 | 5976 |
| Val | 1010 | 747 |
| Test | 981 | 747 |

## Method

Both models pretrained on ImageNet, base frozen, custom binary classification head 
added and trained (transfer learning). Class imbalance (~80/20 benign/malignant) 
handled via weighted `BCEWithLogitsLoss`.

## Results (Test Set)

| Metric | MobileNetV2 | ResNet50 |
|---|---|---|
| Accuracy | 0.7788 | 0.7503 |
| Precision | 0.4646 | 0.4304 |
| Recall | 0.8542 | 0.8542 |
| F1 Score | 0.6018 | 0.5724 |
| AUC-ROC | 0.8780 | 0.8750 |
| Model Size | **8.73 MB** | 89.99 MB |
| Inference Time | 17.56 ms | 9.75 ms |

## Conclusion

MobileNetV2 matches or exceeds ResNet50 on all diagnostic metrics (accuracy, precision, 
F1, AUC) while being ~10x smaller in size — supporting the case for lightweight 
architectures in mobile-deployable medical screening tools without sacrificing 
diagnostic performance.

## Tech Stack

PyTorch, torchvision, RTX 3050 (4GB VRAM), HAM10000 dataset.