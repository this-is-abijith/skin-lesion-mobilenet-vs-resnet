import torch.nn as nn
from torchvision import models

def get_mobilenet(freeze_base=True):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    if freeze_base:
        for param in model.features.parameters():
            param.requires_grad = False  # freeze pretrained ImageNet layers

    # replace classifier head — original outputs 1000 classes, we need 1 (binary, sigmoid)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, 1)  # 1 output, use BCEWithLogitsLoss (no sigmoid here)
    )
    return model