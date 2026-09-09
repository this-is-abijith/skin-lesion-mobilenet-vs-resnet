import torch
from model_resnet import get_resnet

model = get_resnet()
dummy_input = torch.randn(2, 3, 224, 224)
output = model(dummy_input)
print("Output shape:", output.shape)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / Total: {total:,}")