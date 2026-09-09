import torch
from model_mobilenet import get_mobilenet

model = get_mobilenet()
dummy_input = torch.randn(2, 3, 224, 224)  # fake batch of 2
output = model(dummy_input)
print("Output shape:", output.shape)  # expect [2, 1]

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / Total: {total:,}")