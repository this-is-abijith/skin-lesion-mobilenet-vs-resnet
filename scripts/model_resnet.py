import torch.nn as nn
from torchvision import models

def get_resnet(freeze_base=True):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False  # freeze all pretrained layers first

    # replace final fc layer - original outputs 1000 classes, we need 1 (binary)
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.fc.in_features, 1)
    )
    # fc layer new-created, requires_grad=True by default - only this trains
    return model